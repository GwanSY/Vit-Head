#pragma once

#include "FloatingPoint/fixed-point.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ours_vit {

// Batch all independent comparisons at a tree level into one SCI call.
// This fork's LE(lhs, rhs) selects lhs when lhs >= rhs; do not substitute GT.
inline FixArray secure_row_max(FixOp &fix, int party,
                                    const uint64_t *input, int rows, int keys,
                                    int ell, int scale) {
  if (rows <= 0 || keys <= 0 ||
      keys > std::numeric_limits<int>::max() / rows)
    throw std::invalid_argument("invalid secure row maximum shape");
  std::vector<uint64_t> current(static_cast<size_t>(rows) * keys);
  for (int key = 0; key < keys; ++key)
    for (int row = 0; row < rows; ++row)
      current[static_cast<size_t>(key) * rows + row] =
          input[static_cast<size_t>(row) * keys + key];
  while (keys > 1) {
    const int pairs = keys / 2;
    FixArray lhs(party, pairs * rows, true, ell, scale);
    FixArray rhs(party, pairs * rows, true, ell, scale);
    for (int pair = 0; pair < pairs; ++pair) {
      std::copy_n(current.data() + static_cast<size_t>(2 * pair) * rows,
                  rows, lhs.data + static_cast<size_t>(pair) * rows);
      std::copy_n(current.data() + static_cast<size_t>(2 * pair + 1) * rows,
                  rows, rhs.data + static_cast<size_t>(pair) * rows);
    }
    auto select_lhs = fix.LE(lhs, rhs);
    auto selected = fix.if_else(select_lhs, lhs, rhs);
    std::copy_n(selected.data, pairs * rows, current.data());
    if (keys % 2)
      std::copy_n(current.data() + static_cast<size_t>(keys - 1) * rows,
                  rows, current.data() + static_cast<size_t>(pairs) * rows);
    keys = pairs + keys % 2;
  }
  FixArray result(party, rows, true, ell, scale);
  std::copy_n(current.data(), rows, result.data);
  return result;
}

}  // namespace ours_vit
