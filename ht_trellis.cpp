// CPU Viterbi for Halogen's scalar 16-bit / four-bit-shift trellis.
// Build: c++ -O3 -march=native -fopenmp -shared -fPIC ht_trellis.cpp -o ht_trellis.so
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {
constexpr int states = 65536, histories = 4096, length = 256;
struct Workspace {
    std::vector<float> a, b, minima;
    std::vector<uint8_t> trace;
    explicit Workspace(bool enabled = true)
        : a(enabled ? states : 0), b(a.size()), minima(enabled ? histories : 0),
          trace(enabled ? length * histories : 0) {}
};

// Fixed boundary means the last three nibbles equal the input history. With
// that history held fixed, this is an exact Viterbi optimum, not beam search.
float search(const float *x, const float *cb, int n, int boundary,
             uint8_t *codes, Workspace &w) {
    std::fill(w.a.begin(), w.a.end(), boundary < 0 ? 0.f : INFINITY);
    if (boundary >= 0)
        for (int d = 0; d < 16; ++d) w.a[d * histories + boundary] = 0.f;
    for (int t = 0; t < n; ++t) {
        float *prev = w.a.data(), *next = w.b.data();
        uint8_t *back = w.trace.data() + t * histories;
        for (int h = 0; h < histories; ++h) {
            float best = prev[h];
            uint8_t arg = 0;
            for (int d = 1; d < 16; ++d) {
                float candidate = prev[d * histories + h];
                if (candidate < best) { best = candidate; arg = d; }
            }
            w.minima[h] = best;
            back[h] = arg;
        }
        for (int s = 0; s < states; ++s) {
            float error = x[t] - cb[s];
            next[s] = w.minima[s >> 4] + error * error;
        }
        w.a.swap(w.b);
    }
    int end = 0;
    if (boundary < 0) {
        end = std::min_element(w.a.begin(), w.a.end()) - w.a.begin();
    } else {
        end = boundary;
        for (int d = 1; d < 16; ++d) {
            int s = d * histories + boundary;
            if (w.a[s] < w.a[end]) end = s;
        }
    }
    float cost = w.a[end];
    for (int t = n - 1; t >= 0; --t) {
        codes[t] = end & 15;
        end = (end >> 4) | (int(w.trace[t * histories + (end >> 4)]) << 12);
    }
    return cost;
}

struct Path { float cost; uint16_t state, parent; };
struct BeamWorkspace {
    std::vector<Path> active, candidates;
    std::vector<float> best;
    std::vector<uint16_t> parent, trace, codes;
    explicit BeamWorkspace(int width)
        : active(width ? histories : 0), candidates(active.size()), best(active.size()),
          parent(active.size()), trace(length * width), codes(trace.size()) {}
};

void beam_search(const float *x, const float *cb, int boundary, int width,
                 uint8_t *codes, BeamWorkspace &w) {
    int count = boundary < 0 ? histories : 1;
    for (int h = 0; h < count; ++h)
        w.active[h] = {0.f, uint16_t(boundary < 0 ? h : boundary), 0};
    for (int t = 0; t < length; ++t) {
        std::fill(w.best.begin(), w.best.end(), INFINITY);
        int first = 0, last = 16;
        if (boundary >= 0 && t >= length - 3) {
            first = (boundary >> (4 * (length - 1 - t))) & 15;
            last = first + 1;
        }
        for (int j = 0; j < count; ++j) {
            const Path p = w.active[j];
            for (int d = first; d < last; ++d) {
                int state = (int(p.state) << 4) | d;
                float delta = x[t] - cb[state];
                float cost = p.cost + delta * delta;
                int h = state & (histories - 1);
                if (cost < w.best[h]) {
                    w.best[h] = cost;
                    w.parent[h] = j;
                }
            }
        }
        int available = 0;
        for (int h = 0; h < histories; ++h)
            if (std::isfinite(w.best[h]))
                w.candidates[available++] = {w.best[h], uint16_t(h), w.parent[h]};
        count = std::min(width, available);
        auto order = [](const Path &a, const Path &b) {
            return a.cost < b.cost || (a.cost == b.cost && a.state < b.state);
        };
        if (count < available)
            std::nth_element(w.candidates.begin(), w.candidates.begin() + count,
                             w.candidates.begin() + available, order);
        for (int j = 0; j < count; ++j) {
            w.active[j] = w.candidates[j];
            w.trace[t * width + j] = w.active[j].parent;
            w.codes[t * width + j] = w.active[j].state & 15;
        }
    }
    int end = 0;
    for (int j = 1; j < count; ++j)
        if (w.active[j].cost < w.active[end].cost) end = j;
    for (int t = length - 1; t >= 0; --t) {
        codes[t] = w.codes[t * width + end];
        end = w.trace[t * width + end];
    }
}
}

extern "C" int ht_encode(const float *tiles, const float *codebook,
                          uint8_t *codes, int count, int threads, int beam) {
    if (count < 0 || threads < 1 || beam < 0 || beam > histories) return -1;
    #pragma omp parallel num_threads(threads)
    {
        Workspace w(beam == 0);
        BeamWorkspace bw(beam);
        #pragma omp for schedule(static)
        for (int i = 0; i < count; ++i) {
            const float *x = tiles + i * length;
            uint8_t *q = codes + i * length;
            if (beam) beam_search(x, codebook, -1, beam, q, bw);
            else search(x, codebook, length, -1, q, w);
            int boundary = (int(q[253]) << 8) | (int(q[254]) << 4) | q[255];
            if (beam) beam_search(x, codebook, boundary, beam, q, bw);
            else search(x, codebook, length, boundary, q, w);
        }
    }
    return 0;
}

extern "C" float ht_search_fixed(const float *x, const float *codebook,
                                  int n, int boundary, uint8_t *codes) {
    if (n < 1 || n > length || boundary < 0 || boundary >= histories)
        return std::numeric_limits<float>::quiet_NaN();
    Workspace w;
    return search(x, codebook, n, boundary, codes, w);
}
