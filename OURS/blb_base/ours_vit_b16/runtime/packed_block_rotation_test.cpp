#include "packed_block_rotation.h"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

using Slots = std::vector<int64_t>;

static void add_rotation(Slots &x, int shift) {
    const Slots source = x;
    const int size = x.size();
    for (int i = 0; i < size; ++i)
        x[i] += source[(i + shift + size) % size];
}

int main() {
    // Distinct signed values in both head halves detect wrap-around mixing.
    for (int blocks : {1, 2, 4, 8, 16, 32, 64}) {
        for (int rows : {1, 64, 128, 256}) {
            Slots input(2 * blocks * rows);
            for (int i = 0; i < static_cast<int>(input.size()); ++i)
                input[i] = (i * 7919 % 100003) - 50001;
            for (int block = 0; block < blocks; ++block) {
                Slots reduced = input;
                Slots broadcast(input.size(), 0);
                for (int half = 0; half < 2; ++half)
                    for (int token = 0; token < rows; ++token) {
                        const int i = (half * blocks + block) * rows + token;
                        broadcast[i] = input[i];
                    }
                for (int stride = 1; stride < blocks; stride <<= 1) {
                    add_rotation(reduced, ours_vit::packed_block_rotation(
                        block, stride, rows, false));
                    add_rotation(broadcast, ours_vit::packed_block_rotation(
                        block, stride, rows, true));
                }
                for (int half = 0; half < 2; ++half)
                    for (int token = 0; token < rows; ++token) {
                        const int target = (half * blocks + block) * rows + token;
                        int64_t expected = 0;
                        for (int feature = 0; feature < blocks; ++feature) {
                            const int i = (half * blocks + feature) * rows + token;
                            expected += input[i];
                            assert(broadcast[i] == input[target]);
                        }
                        assert(reduced[target] == expected);
                    }
            }
        }
    }
    std::cout << "PACKED_BLOCK_ROTATION_TEST status=PASS\n";
}
