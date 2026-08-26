#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace blb_vit {

// Keep a full-width 2*ell intermediate product below SCI's 64-bit limit.
constexpr int kRingBits = 31;

struct LocalStats {
  uint64_t bytes = 0;
  uint64_t rounds = 0;
  double milliseconds = 0.0;
};

inline uint64_t encode_ring(double value, int scale_bits,
                            int ell = kRingBits) {
  const int64_t quantized =
      static_cast<int64_t>(std::llround(value * (1ULL << scale_bits)));
  const uint64_t mask = ell == 64 ? ~0ULL : ((1ULL << ell) - 1);
  return static_cast<uint64_t>(quantized) & mask;
}

inline LocalStats aggregate_parties(const LocalStats &left,
                                    const LocalStats &right) {
  return {left.bytes + right.bytes, left.rounds + right.rounds,
          std::max(left.milliseconds, right.milliseconds)};
}

} // namespace blb_vit
