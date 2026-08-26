#pragma once

#include "FloatingPoint/fp-math.h"

#include <memory>
#include <string>
#include <vector>

namespace blb_vit {

class CkksLinear {
public:
  CkksLinear(int party, sci::NetIO *io, int dim);
  ~CkksLinear();

  CkksLinear(const CkksLinear &) = delete;
  CkksLinear &operator=(const CkksLinear &) = delete;

  FixArray project_private(FPMath &fp, const std::vector<double> &weights,
                           const std::vector<double> &input, int rows,
                           int dim, int output_scale_bits);
  const char *backend_name() const { return "seal_ckks"; }
  const char *device_name() const { return "cpu"; }
  void synchronize() const {}

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace blb_vit
