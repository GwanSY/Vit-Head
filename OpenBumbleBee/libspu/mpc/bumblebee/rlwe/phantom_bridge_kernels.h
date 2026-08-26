#pragma once

#include <cstddef>
#include <cstdint>

#include "cuda_runtime_api.h"

namespace spu::mpc::bumblebee {

// Canonicalize SEAL's valid lazy [0, 4q) RNS representation before invoking
// Phantom kernels, which require [0, q). This is a representation bridge, not
// an HE operation or a change to the polynomial arithmetic.
cudaError_t CanonicalizePhantomRns(std::uint64_t* data,
                                   const void* phantom_moduli,
                                   std::size_t polynomial_count,
                                   std::size_t modulus_size,
                                   std::size_t degree,
                                   cudaStream_t stream);

cudaError_t DyadicProductPhantomRns(std::uint64_t* cipher,
                                    const std::uint64_t* plain,
                                    const void* phantom_moduli,
                                    std::size_t polynomial_count,
                                    std::size_t modulus_size,
                                    std::size_t degree,
                                    cudaStream_t stream);

cudaError_t AddSubCipherPhantomRns(
    std::uint64_t* lhs, const std::uint64_t* rhs,
    const void* phantom_moduli, std::size_t polynomial_count,
    std::size_t modulus_size, std::size_t degree, bool subtract,
    cudaStream_t stream);

cudaError_t NegateCipherPhantomRns(
    std::uint64_t* cipher, const void* phantom_moduli,
    std::size_t polynomial_count, std::size_t modulus_size,
    std::size_t degree, cudaStream_t stream);

cudaError_t AddSubPlainPhantomRns(std::uint64_t* cipher,
                                  const std::uint64_t* plain,
                                  const void* phantom_moduli,
                                  std::size_t modulus_size,
                                  std::size_t degree, bool subtract,
                                  cudaStream_t stream);

cudaError_t NegacyclicRightShiftPhantomRns(
    const std::uint64_t* input, std::uint64_t* output,
    const void* phantom_moduli, std::size_t polynomial_count,
    std::size_t modulus_size, std::size_t degree, std::size_t shift,
    cudaStream_t stream);

}  // namespace spu::mpc::bumblebee
