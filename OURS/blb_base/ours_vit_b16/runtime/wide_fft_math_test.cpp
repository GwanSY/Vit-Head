#include "wide_fft_math.h"
#include <iostream>

int main() {
    int width, scale;
    uint64_t ah, al, bh, bl;
    int64_t c, d;
    while (std::cin >> width >> scale >> ah >> al >> bh >> bl >> c >> d) {
        const auto result = ours_vit::multiply_fft_root(
            (__uint128_t(ah) << 64) | al, (__uint128_t(bh) << 64) | bl,
            c, d, width, scale);
        std::cout << uint64_t(result.first >> 64) << ' ' << uint64_t(result.first)
                  << ' ' << uint64_t(result.second >> 64) << ' '
                  << uint64_t(result.second) << '\n';
    }
}
