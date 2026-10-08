#pragma once

namespace ours_vit {

// CKKS positive rotations read the slot to the right. For a power-of-two
// half containing B blocks, the reduction shifts sum to [-b, B-1-b], so
// output block b receives exactly its own head's blocks [0,B). The opposite
// directions broadcast a masked block b to that same half. No alignment
// rotation is needed and the two packed heads never mix at retained slots.
inline int packed_block_rotation(int block, int stride, int rows,
                                 bool broadcast) {
    const int reduction = (block & stride) ? -stride : stride;
    return (broadcast ? -reduction : reduction) * rows;
}

} // namespace ours_vit
