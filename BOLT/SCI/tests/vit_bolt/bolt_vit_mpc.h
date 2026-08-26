#pragma once

#include "FloatingPoint/fp-math.h"
#include "OT/ot_pack.h"
#include "utils/io_pack.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// This is the BERT NonLinear execution strategy expressed for the ViT
// protocol.  It owns the same 32 alternating-party MPC channels and only
// partitions independent rows/elements; it does not change any approximation
// or fixed-point contract.
class BoltVitMPC {
public:
  static constexpr int kThreads = 32;

  BoltVitMPC(int party, const std::string &address, int first_port);
  ~BoltVitMPC() = default;

  std::uint64_t communication() const;
  std::uint64_t rounds() const;

  std::vector<FixArray> layer_norm(
      const std::vector<FixArray> &rows,
      std::uint64_t epsilon_lsb = 0);
  std::vector<FixArray> softmax(const std::vector<FixArray> &rows);
  FixArray gelu(const FixArray &input);
  FixArray gt_p_sub(const FixArray &input, std::uint64_t p,
                    int output_scale);
  FixArray right_shift(const FixArray &input, int amount);
  FixArray reduce(const FixArray &input, int output_ell);
  FixArray signed_extend_to_64(const FixArray &input);

private:
  int party_;
  std::vector<std::unique_ptr<sci::IOPack>> io_;
  std::vector<std::unique_ptr<sci::OTPack>> ot_;
  std::vector<std::unique_ptr<FPMath>> fp_;
};
