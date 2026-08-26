#include <seal/seal.h>

#include "blb_ckks_linear.h"
#include "blb_vit_common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace blb_vit {
namespace {

constexpr size_t kPolyModulusDegree = 8192;
constexpr int kCkksScaleBits = 40;

template <class T>
void send_he_object(sci::NetIO *io, const T &object) {
  std::ostringstream stream(std::ios::binary | std::ios::out);
  object.save(stream, seal::compr_mode_type::none);
  const std::string bytes = stream.str();
  const uint64_t size = bytes.size();
  io->send_data(&size, sizeof(size));
  io->send_data(bytes.data(), bytes.size());
}

template <class T>
T recv_he_object(sci::NetIO *io, const seal::SEALContext &context) {
  uint64_t size = 0;
  io->recv_data(&size, sizeof(size));
  if (size == 0 || size > (1ULL << 30))
    throw std::runtime_error("invalid serialized CKKS object size");
  std::string bytes(size, '\0');
  io->recv_data(bytes.data(), bytes.size());
  std::istringstream stream(bytes, std::ios::binary | std::ios::in);
  T object;
  object.load(context, stream);
  return object;
}

} // namespace

struct CkksLinear::Impl {
  Impl(int role, sci::NetIO *channel, int matrix_dim)
      : party(role), io(channel), dim(matrix_dim) {
    if (dim <= 0) throw std::runtime_error("CKKS dimension must be positive");

    seal::EncryptionParameters parms(seal::scheme_type::ckks);
    parms.set_poly_modulus_degree(kPolyModulusDegree);
    parms.set_coeff_modulus(
        seal::CoeffModulus::Create(kPolyModulusDegree, {60, 40, 60}));
    context = std::make_shared<seal::SEALContext>(
        parms, true, seal::sec_level_type::tc128);
    if (!context->parameters_set())
      throw std::runtime_error("SEAL rejected CKKS parameters");
    encoder = std::make_unique<seal::CKKSEncoder>(*context);
    evaluator = std::make_unique<seal::Evaluator>(*context);

    seal::PublicKey public_key;
    if (party == sci::BOB) {
      seal::KeyGenerator keygen(*context);
      secret_key = keygen.secret_key();
      keygen.create_public_key(public_key);
      keygen.create_galois_keys(rotation_steps(dim), galois_keys);
      send_he_object(io, public_key);
      send_he_object(io, galois_keys);
      decryptor = std::make_unique<seal::Decryptor>(*context, secret_key);
    } else {
      public_key = recv_he_object<seal::PublicKey>(io, *context);
      galois_keys = recv_he_object<seal::GaloisKeys>(io, *context);
    }
    encryptor = std::make_unique<seal::Encryptor>(*context, public_key);
    io->flush();
  }

  static std::vector<int> rotation_steps(int matrix_dim) {
    std::vector<int> steps;
    for (int k = 1; k < matrix_dim; ++k) {
      steps.push_back(k);
      steps.push_back(k - matrix_dim);
    }
    std::sort(steps.begin(), steps.end());
    steps.erase(std::unique(steps.begin(), steps.end()), steps.end());
    return steps;
  }

  std::vector<double> packed_input(const std::vector<double> &input,
                                   int rows, int matrix_dim) const {
    const size_t logical_size = static_cast<size_t>(rows) * matrix_dim;
    if (logical_size > encoder->slot_count())
      throw std::runtime_error("rows*dim exceeds CKKS slot count");
    std::vector<double> slots(encoder->slot_count(), 0.0);
    if (party == sci::BOB)
      std::copy(input.begin(), input.end(), slots.begin());
    return slots;
  }

  std::vector<double> weight_mask(const std::vector<double> &weights,
                                  int rows, int matrix_dim, int k,
                                  bool wrap) const {
    std::vector<double> mask(encoder->slot_count(), 0.0);
    for (int row = 0; row < rows; ++row)
      for (int out = 0; out < matrix_dim; ++out) {
        const bool needs_wrap = out + k >= matrix_dim;
        if (needs_wrap != wrap) continue;
        const int in = (out + k) % matrix_dim;
        mask[static_cast<size_t>(row) * matrix_dim + out] =
            weights[static_cast<size_t>(out) * matrix_dim + in];
      }
    return mask;
  }

  seal::Ciphertext apply_linear(const seal::Ciphertext &input,
                                const std::vector<double> &weights,
                                int rows, int matrix_dim) const {
    seal::Ciphertext output;
    bool initialized = false;
    for (int k = 0; k < matrix_dim; ++k) {
      seal::Ciphertext rotated = input;
      if (k != 0)
        evaluator->rotate_vector(input, k, galois_keys, rotated);
      seal::Plaintext direct;
      encoder->encode(weight_mask(weights, rows, matrix_dim, k, false),
                      std::ldexp(1.0, kCkksScaleBits), direct);
      seal::Ciphertext term;
      evaluator->multiply_plain(rotated, direct, term);
      if (!initialized) {
        output = std::move(term);
        initialized = true;
      } else {
        evaluator->add_inplace(output, term);
      }
      if (k != 0) {
        seal::Ciphertext wrapped;
        evaluator->rotate_vector(input, k - matrix_dim, galois_keys, wrapped);
        seal::Plaintext wrap_plain;
        encoder->encode(weight_mask(weights, rows, matrix_dim, k, true),
                        std::ldexp(1.0, kCkksScaleBits), wrap_plain);
        evaluator->multiply_plain_inplace(wrapped, wrap_plain);
        evaluator->add_inplace(output, wrapped);
      }
    }
    evaluator->rescale_to_next_inplace(output);
    return output;
  }

  FixArray masked_slots_to_shares(FPMath &fp, seal::Ciphertext cipher,
                                  int logical_size,
                                  int output_scale_bits) const {
    const uint64_t ring_mask = (1ULL << kRingBits) - 1;
    const uint64_t sign_bit = 1ULL << (kRingBits - 1);
    std::vector<uint64_t> shares(logical_size, 0);
    if (party == sci::ALICE) {
      sci::PRG128 prg;
      std::vector<uint64_t> random_mask(logical_size);
      prg.random_data(random_mask.data(),
                      random_mask.size() * sizeof(uint64_t));
      std::vector<double> slot_mask(encoder->slot_count(), 0.0);
      const double output_scale = std::ldexp(1.0, output_scale_bits);
      for (int i = 0; i < logical_size; ++i) {
        const uint64_t residue = random_mask[i] & ring_mask;
        const int64_t centered =
            residue >= sign_bit
                ? static_cast<int64_t>(residue) -
                      static_cast<int64_t>(1ULL << kRingBits)
                : static_cast<int64_t>(residue);
        slot_mask[i] = static_cast<double>(centered) / output_scale;
        shares[i] = (-residue) & ring_mask;
      }
      seal::Plaintext mask_plain;
      encoder->encode(slot_mask, cipher.parms_id(), cipher.scale(),
                      mask_plain);
      evaluator->add_plain_inplace(cipher, mask_plain);
      send_he_object(io, cipher);
      io->flush();
    } else {
      cipher = recv_he_object<seal::Ciphertext>(io, *context);
      seal::Plaintext plain;
      decryptor->decrypt(cipher, plain);
      std::vector<double> slots;
      encoder->decode(plain, slots);
      const double output_scale = std::ldexp(1.0, output_scale_bits);
      for (int i = 0; i < logical_size; ++i) {
        const int64_t quantized =
            static_cast<int64_t>(std::llround(slots[i] * output_scale));
        shares[i] = static_cast<uint64_t>(quantized) & ring_mask;
      }
    }
    FixArray output(fp.party, logical_size, true, kRingBits,
                    output_scale_bits);
    std::copy(shares.begin(), shares.end(), output.data);
    return output;
  }

  FixArray project_chunk(FPMath &fp, const std::vector<double> &weights,
                         const std::vector<double> &input, int rows,
                         int matrix_dim, int output_scale_bits) {
    seal::Ciphertext encrypted;
    if (party == sci::BOB) {
      seal::Plaintext plain;
      encoder->encode(packed_input(input, rows, matrix_dim),
                      std::ldexp(1.0, kCkksScaleBits), plain);
      encryptor->encrypt(plain, encrypted);
      send_he_object(io, encrypted);
      io->flush();
    } else {
      encrypted = recv_he_object<seal::Ciphertext>(io, *context);
      encrypted = apply_linear(encrypted, weights, rows, matrix_dim);
    }
    return masked_slots_to_shares(
        fp, std::move(encrypted), rows * matrix_dim, output_scale_bits);
  }

  FixArray project_private(FPMath &fp, const std::vector<double> &weights,
                           const std::vector<double> &input, int rows,
                           int matrix_dim, int output_scale_bits) {
    if (matrix_dim != dim ||
        input.size() != static_cast<size_t>(rows) * matrix_dim ||
        weights.size() != static_cast<size_t>(matrix_dim) * matrix_dim)
      throw std::runtime_error("bad CKKS projection shape");
    const int rows_per_cipher =
        static_cast<int>(encoder->slot_count()) / matrix_dim;
    FixArray output(fp.party, rows * matrix_dim, true, kRingBits,
                    output_scale_bits);
    for (int offset = 0; offset < rows; offset += rows_per_cipher) {
      const int chunk_rows = std::min(rows_per_cipher, rows - offset);
      const size_t begin = static_cast<size_t>(offset) * matrix_dim;
      const size_t count = static_cast<size_t>(chunk_rows) * matrix_dim;
      std::vector<double> chunk(input.begin() + begin,
                                input.begin() + begin + count);
      FixArray chunk_output = project_chunk(
          fp, weights, chunk, chunk_rows, matrix_dim, output_scale_bits);
      std::copy(chunk_output.data, chunk_output.data + chunk_output.size,
                output.data + begin);
    }
    return output;
  }

  int party;
  sci::NetIO *io;
  int dim;
  std::shared_ptr<seal::SEALContext> context;
  std::unique_ptr<seal::CKKSEncoder> encoder;
  std::unique_ptr<seal::Evaluator> evaluator;
  std::unique_ptr<seal::Encryptor> encryptor;
  std::unique_ptr<seal::Decryptor> decryptor;
  seal::SecretKey secret_key;
  seal::GaloisKeys galois_keys;
};

CkksLinear::CkksLinear(int party, sci::NetIO *io, int dim)
    : impl_(std::make_unique<Impl>(party, io, dim)) {}

CkksLinear::~CkksLinear() = default;

FixArray CkksLinear::project_private(
    FPMath &fp, const std::vector<double> &weights,
    const std::vector<double> &input, int rows, int dim,
    int output_scale_bits) {
  return impl_->project_private(fp, weights, input, rows, dim,
                                output_scale_bits);
}

} // namespace blb_vit
