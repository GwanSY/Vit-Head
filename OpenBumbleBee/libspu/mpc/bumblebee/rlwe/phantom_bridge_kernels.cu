#include "libspu/mpc/bumblebee/rlwe/phantom_bridge_kernels.h"

#include "ntt.cuh"
#include "uintmodmath.cuh"

namespace spu::mpc::bumblebee {
namespace {

__global__ void CanonicalizeRnsKernel(std::uint64_t* data,
                                      const DModulus* moduli,
                                      std::size_t count,
                                      std::size_t modulus_size,
                                      std::size_t degree) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += blockDim.x * gridDim.x) {
    const auto modulus_index = (index / degree) % modulus_size;
    const auto q = moduli[modulus_index].value();
    auto value = data[index];
    // SEAL normally bounds lazy Harvey values by 4q. Keep the loop robust to
    // seeded/encryption buffers with a wider yet valid unsigned range.
    while (value >= q) value -= q;
    data[index] = value;
  }
}

__global__ void DyadicProductRnsKernel(std::uint64_t* cipher,
                                       const std::uint64_t* plain,
                                       const DModulus* moduli,
                                       std::size_t count,
                                       std::size_t modulus_size,
                                       std::size_t degree) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += blockDim.x * gridDim.x) {
    const auto modulus_index = (index / degree) % modulus_size;
    const auto modulus = moduli[modulus_index];
    const auto plain_index = modulus_index * degree + index % degree;
    cipher[index] = phantom::arith::multiply_and_barrett_reduce_uint64(
        cipher[index], plain[plain_index], modulus.value(),
        modulus.const_ratio());
  }
}

__global__ void AddSubPlainRnsKernel(std::uint64_t* cipher,
                                     const std::uint64_t* plain,
                                     const DModulus* moduli,
                                     std::size_t count, std::size_t degree,
                                     bool subtract) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += blockDim.x * gridDim.x) {
    const auto modulus_index = index / degree;
    const auto q = moduli[modulus_index].value();
    const auto lhs = cipher[index];
    const auto rhs = plain[index];
    if (subtract) {
      cipher[index] = lhs >= rhs ? lhs - rhs : lhs + q - rhs;
    } else {
      auto sum = lhs + rhs;
      if (sum >= q) sum -= q;
      cipher[index] = sum;
    }
  }
}

__global__ void AddSubCipherRnsKernel(std::uint64_t* lhs,
                                      const std::uint64_t* rhs,
                                      const DModulus* moduli,
                                      std::size_t count,
                                      std::size_t modulus_size,
                                      std::size_t degree, bool subtract) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += blockDim.x * gridDim.x) {
    const auto modulus_index = (index / degree) % modulus_size;
    const auto q = moduli[modulus_index].value();
    const auto left = lhs[index];
    const auto right = rhs[index];
    if (subtract) {
      lhs[index] = left >= right ? left - right : left + q - right;
    } else {
      auto sum = left + right;
      if (sum >= q) sum -= q;
      lhs[index] = sum;
    }
  }
}

__global__ void NegateCipherRnsKernel(std::uint64_t* cipher,
                                      const DModulus* moduli,
                                      std::size_t count,
                                      std::size_t modulus_size,
                                      std::size_t degree) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += blockDim.x * gridDim.x) {
    const auto modulus_index = (index / degree) % modulus_size;
    const auto q = moduli[modulus_index].value();
    const auto value = cipher[index];
    cipher[index] = value == 0 ? 0 : q - value;
  }
}

__global__ void NegacyclicRightShiftRnsKernel(
    const std::uint64_t* input, std::uint64_t* output,
    const DModulus* moduli, std::size_t count, std::size_t modulus_size,
    std::size_t degree, std::size_t shift) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count; index += blockDim.x * gridDim.x) {
    const auto coefficient = index % degree;
    const auto modulus_index = (index / degree) % modulus_size;
    const auto q = moduli[modulus_index].value();
    if (coefficient < shift) {
      const auto source = index + degree - shift;
      const auto value = input[source];
      output[index] = value == 0 ? 0 : q - value;
    } else {
      output[index] = input[index - shift];
    }
  }
}

}  // namespace

cudaError_t CanonicalizePhantomRns(std::uint64_t* data,
                                   const void* phantom_moduli,
                                   std::size_t polynomial_count,
                                   std::size_t modulus_size,
                                   std::size_t degree,
                                   cudaStream_t stream) {
  const auto count = polynomial_count * modulus_size * degree;
  constexpr std::size_t threads = 256;
  const auto blocks = static_cast<unsigned>((count + threads - 1) / threads);
  CanonicalizeRnsKernel<<<blocks, threads, 0, stream>>>(
      data, static_cast<const DModulus*>(phantom_moduli), count, modulus_size,
      degree);
  return cudaGetLastError();
}

cudaError_t DyadicProductPhantomRns(
    std::uint64_t* cipher, const std::uint64_t* plain,
    const void* phantom_moduli, std::size_t polynomial_count,
    std::size_t modulus_size, std::size_t degree, cudaStream_t stream) {
  const auto count = polynomial_count * modulus_size * degree;
  constexpr std::size_t threads = 256;
  const auto blocks = static_cast<unsigned>((count + threads - 1) / threads);
  DyadicProductRnsKernel<<<blocks, threads, 0, stream>>>(
      cipher, plain, static_cast<const DModulus*>(phantom_moduli), count,
      modulus_size, degree);
  return cudaGetLastError();
}

cudaError_t AddSubCipherPhantomRns(
    std::uint64_t* lhs, const std::uint64_t* rhs,
    const void* phantom_moduli, std::size_t polynomial_count,
    std::size_t modulus_size, std::size_t degree, bool subtract,
    cudaStream_t stream) {
  const auto count = polynomial_count * modulus_size * degree;
  constexpr std::size_t threads = 256;
  const auto blocks = static_cast<unsigned>((count + threads - 1) / threads);
  AddSubCipherRnsKernel<<<blocks, threads, 0, stream>>>(
      lhs, rhs, static_cast<const DModulus*>(phantom_moduli), count,
      modulus_size, degree, subtract);
  return cudaGetLastError();
}

cudaError_t NegateCipherPhantomRns(
    std::uint64_t* cipher, const void* phantom_moduli,
    std::size_t polynomial_count, std::size_t modulus_size,
    std::size_t degree, cudaStream_t stream) {
  const auto count = polynomial_count * modulus_size * degree;
  constexpr std::size_t threads = 256;
  const auto blocks = static_cast<unsigned>((count + threads - 1) / threads);
  NegateCipherRnsKernel<<<blocks, threads, 0, stream>>>(
      cipher, static_cast<const DModulus*>(phantom_moduli), count,
      modulus_size, degree);
  return cudaGetLastError();
}

cudaError_t AddSubPlainPhantomRns(
    std::uint64_t* cipher, const std::uint64_t* plain,
    const void* phantom_moduli, std::size_t modulus_size,
    std::size_t degree, bool subtract, cudaStream_t stream) {
  const auto count = modulus_size * degree;
  constexpr std::size_t threads = 256;
  const auto blocks = static_cast<unsigned>((count + threads - 1) / threads);
  AddSubPlainRnsKernel<<<blocks, threads, 0, stream>>>(
      cipher, plain, static_cast<const DModulus*>(phantom_moduli), count,
      degree, subtract);
  return cudaGetLastError();
}

cudaError_t NegacyclicRightShiftPhantomRns(
    const std::uint64_t* input, std::uint64_t* output,
    const void* phantom_moduli, std::size_t polynomial_count,
    std::size_t modulus_size, std::size_t degree, std::size_t shift,
    cudaStream_t stream) {
  const auto count = polynomial_count * modulus_size * degree;
  constexpr std::size_t threads = 256;
  const auto blocks = static_cast<unsigned>((count + threads - 1) / threads);
  NegacyclicRightShiftRnsKernel<<<blocks, threads, 0, stream>>>(
      input, output, static_cast<const DModulus*>(phantom_moduli), count,
      modulus_size, degree, shift);
  return cudaGetLastError();
}

}  // namespace spu::mpc::bumblebee
