#pragma once

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ours_vit {

// Split at the public fractional bit so no 192-bit intermediate is needed.
// Combine the two low products before shifting to preserve the rounding carry.
inline std::pair<__uint128_t, __uint128_t> multiply_fft_root(
    __uint128_t real, __uint128_t imag, int64_t root_real,
    int64_t root_imag, int width, int scale) {
    if (width <= 1 || width >= 127 || scale <= 0 || scale > 62)
        throw std::invalid_argument("invalid FFT product dimensions");
    const __int128 unit = __int128{1} << scale;
    if (root_real < -unit || root_real > unit ||
        root_imag < -unit || root_imag > unit)
        throw std::invalid_argument("FFT root is outside the unit square");
    const __uint128_t mask = (__uint128_t{1} << width) - 1;
    const __uint128_t low_mask = (__uint128_t{1} << scale) - 1;
    real &= mask;
    imag &= mask;
    const auto high = [=](__uint128_t value) -> __int128 {
        return static_cast<__int128>(value >> scale) -
            ((value >> (width - 1)) ? (__int128{1} << (width - scale)) : 0);
    };
    const auto floor_shift = [=](__int128 value) -> __int128 {
        return value >= 0 ? value >> scale : -((-value + unit - 1) >> scale);
    };
    if (width <= scale) {
        const auto signed_value = [=](__uint128_t value) -> __int128 {
            return static_cast<__int128>(value) -
                ((value >> (width - 1)) ? (__int128{1} << width) : 0);
        };
        const __int128 a = signed_value(real), b = signed_value(imag);
        return {static_cast<__uint128_t>(floor_shift(a * root_real - b * root_imag)) & mask,
                static_cast<__uint128_t>(floor_shift(a * root_imag + b * root_real)) & mask};
    }
    const __int128 a = high(real), b = high(imag);
    const __int128 lo_a = real & low_mask, lo_b = imag & low_mask;
    const __int128 out_real = a * root_real - b * root_imag +
        floor_shift(lo_a * root_real - lo_b * root_imag);
    const __int128 out_imag = a * root_imag + b * root_real +
        floor_shift(lo_a * root_imag + lo_b * root_real);
    return {static_cast<__uint128_t>(out_real) & mask,
            static_cast<__uint128_t>(out_imag) & mask};
}

}  // namespace ours_vit
