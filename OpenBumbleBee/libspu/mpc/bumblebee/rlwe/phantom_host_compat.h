#pragma once

// Phantom's public headers contain a few CUDA intrinsics in inline bodies.
// Keep the adapter in an ordinary C++ translation unit by supplying their
// exact host equivalents. CUDA translation units use the native definitions.
#if defined(OPENBUMBLEBEE_ENABLE_PHANTOM) && !defined(__CUDACC__)

#include <cstdint>

inline std::uint32_t __brev(std::uint32_t value) noexcept {
  value = ((value & 0x55555555U) << 1U) | ((value >> 1U) & 0x55555555U);
  value = ((value & 0x33333333U) << 2U) | ((value >> 2U) & 0x33333333U);
  value = ((value & 0x0F0F0F0FU) << 4U) | ((value >> 4U) & 0x0F0F0F0FU);
  value = (value << 24U) | ((value & 0xFF00U) << 8U) |
          ((value >> 8U) & 0xFF00U) | (value >> 24U);
  return value;
}

inline int __popc(std::uint32_t value) noexcept {
  return __builtin_popcount(value);
}

inline std::uint64_t __umul64hi(std::uint64_t lhs,
                                std::uint64_t rhs) noexcept {
  using uint128 = unsigned __int128;
  return static_cast<std::uint64_t>(
      (static_cast<uint128>(lhs) * static_cast<uint128>(rhs)) >> 64U);
}

inline int __clzll(std::uint64_t value) noexcept {
  return value == 0 ? 64 : __builtin_clzll(value);
}

namespace phantom::arith {
template <typename T>
constexpr const T& min(const T& lhs, const T& rhs) noexcept {
  return rhs < lhs ? rhs : lhs;
}
}  // namespace phantom::arith

#endif
