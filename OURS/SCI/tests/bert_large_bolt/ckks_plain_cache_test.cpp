#include "ckks_plain_cache.h"

#include "LinearHE/evaluator-wrapper.h"

#include <ckks.h>
#include <seal/seal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_same_ciphertext(const seal::Ciphertext &expected,
                             const seal::Ciphertext &actual,
                             const char *label) {
  require(expected.parms_id() == actual.parms_id(),
          "host/device ciphertext parms_id mismatch");
  require(expected.size() == actual.size(),
          "host/device ciphertext size mismatch");
  require(expected.coeff_modulus_size() == actual.coeff_modulus_size(),
          "host/device ciphertext modulus-size mismatch");
  require(expected.is_ntt_form() == actual.is_ntt_form(),
          "host/device ciphertext NTT-state mismatch");
  require(expected.scale() == actual.scale(),
          "host/device ciphertext scale mismatch");

  const auto count = expected.size() * expected.coeff_modulus_size() *
                     expected.poly_modulus_degree();
  if (!std::equal(expected.data(), expected.data() + count, actual.data())) {
    throw std::runtime_error(std::string(label) +
                             " host/device RNS coefficients differ");
  }
}

double decoded_max_abs_error(const seal::SEALContext &context,
                             const seal::SecretKey &secret_key,
                             const seal::Ciphertext &expected,
                             const seal::Ciphertext &actual) {
  seal::Decryptor decryptor(context, secret_key);
  seal::CKKSEncoder encoder(context);
  seal::Plaintext expected_plain;
  seal::Plaintext actual_plain;
  decryptor.decrypt(expected, expected_plain);
  decryptor.decrypt(actual, actual_plain);
  std::vector<double> expected_slots;
  std::vector<double> actual_slots;
  encoder.decode(expected_plain, expected_slots);
  encoder.decode(actual_plain, actual_slots);
  require(expected_slots.size() == actual_slots.size(),
          "host/device decoded slot-count mismatch");
  double max_error = 0.0;
  for (std::size_t i = 0; i < expected_slots.size(); ++i) {
    max_error =
        std::max(max_error, std::abs(expected_slots[i] - actual_slots[i]));
  }
  return max_error;
}

} // namespace

int main() {
  try {
    constexpr std::size_t degree = 32768;
    // The formal he1 profile is intentional: its QKV rescale leaves about 56
    // scale bits.  Testing at a synthetic 30-bit output exaggerates the known
    // SEAL/Phantom rounding-policy difference and is not the cache's runtime
    // schedule.
    const std::vector<int> modulus_bits{60, 34, 60, 60, 60, 60, 60};

    seal::EncryptionParameters host_parms(seal::scheme_type::ckks);
    host_parms.set_poly_modulus_degree(degree);
    host_parms.set_coeff_modulus(
        seal::CoeffModulus::Create(degree, modulus_bits));
    seal::SEALContext host_context(host_parms, true,
                                   seal::sec_level_type::tc128);

    phantom::EncryptionParameters device_parms(phantom::scheme_type::ckks);
    device_parms.set_poly_modulus_degree(degree);
    device_parms.set_coeff_modulus(
        phantom::arith::CoeffModulus::Create(degree, modulus_bits));
    PhantomContext device_context(device_parms);

    const auto levels =
        validate_ckks_host_device_contexts(host_context, device_context);
    require(levels.size() == 6, "unexpected CKKS data-level count");

    seal::CKKSEncoder encoder(host_context);
    seal::Evaluator host_evaluator(host_context);
    seal::KeyGenerator keygen(host_context);
    const seal::SecretKey secret_key = keygen.secret_key();
    seal::PublicKey public_key;
    keygen.create_public_key(public_key);
    seal::Encryptor encryptor(host_context, public_key);
    PhantomEvaluator device_evaluator(device_context);
    PhantomCKKSEncoder device_encoder(device_context);

    std::vector<double> input(encoder.slot_count());
    std::vector<double> weight(encoder.slot_count());
    for (std::size_t i = 0; i < input.size(); ++i) {
      input[i] = static_cast<double>(static_cast<int>(i % 17) - 8) / 32.0;
      weight[i] = static_cast<double>(static_cast<int>(i % 11) - 5) / 16.0;
    }

    for (std::size_t index = 0; index < levels.size(); ++index) {
      const double input_scale =
          std::ldexp(1.0, index == 0 ? 72 : 20);
      const double weight_scale =
          std::ldexp(1.0, index == 0 ? 44 : 20);
      seal::Plaintext input_plain;
      seal::Plaintext weight_plain;
      encoder.encode(input, levels[index].seal_parms_id, input_scale,
                     input_plain);
      encoder.encode(weight, levels[index].seal_parms_id, weight_scale,
                     weight_plain);

      seal::Ciphertext host_input;
      encryptor.encrypt(input_plain, host_input);
      seal::Ciphertext expected_product;
      host_evaluator.multiply_plain(host_input, weight_plain,
                                    expected_product);

      PhantomPlaintext device_weight;
      UnifiedPlaintext::to_device(host_context, weight_plain, device_context,
                                  device_weight);
      if (index == 0) {
        std::vector<double> device_decoded_weight;
        device_encoder.decode(device_context, device_weight,
                              device_decoded_weight);
        double upload_decode_error = 0.0;
        for (std::size_t i = 0; i < weight.size(); ++i) {
          upload_decode_error = std::max(
              upload_decode_error,
              std::abs(weight[i] - device_decoded_weight[i]));
        }
        std::cout << "CKKS_PLAIN_CACHE_UPLOAD level=" << index
                  << " max_abs_error=" << upload_decode_error << std::endl;
        if (!std::isfinite(upload_decode_error) ||
            upload_decode_error > 1e-6) {
          throw std::runtime_error(
              "raw SEAL plaintext is not a valid Phantom NTT plaintext");
        }
      }
      UnifiedCiphertext device_input(host_input);
      device_input.to_device(host_context, device_context);
      UnifiedCiphertext device_product(LOCATION::DEVICE);
      device_evaluator.multiply_plain(device_input.dcipher(), device_weight,
                                      device_product.dcipher());
      device_product.to_host(host_context, device_context);
      require_same_ciphertext(expected_product, device_product.hcipher(),
                              "ct-pt multiply");

      if (index == 0) {
        host_evaluator.rescale_to_next_inplace(expected_product);

        UnifiedCiphertext device_rescaled(host_input);
        device_rescaled.to_device(host_context, device_context);
        device_evaluator.multiply_plain_inplace(device_rescaled.dcipher(),
                                                device_weight);
        device_evaluator.rescale_to_next_inplace(device_rescaled.dcipher());
        device_rescaled.to_host(host_context, device_context);
        const double error = decoded_max_abs_error(
            host_context, secret_key, expected_product,
            device_rescaled.hcipher());
        std::cout << "CKKS_PLAIN_CACHE_RESCALE level=" << index
                  << " max_abs_error=" << error << std::endl;
        if (!std::isfinite(error) || error > 1e-9) {
          throw std::runtime_error(
              "formal QKV multiply/rescale decoded error exceeds 1e-9");
        }
      }
    }

    std::cout << "CKKS_PLAIN_CACHE_TEST status=PASS levels=" << levels.size()
              << std::endl;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "CKKS_PLAIN_CACHE_TEST status=FAIL reason=" << error.what()
              << std::endl;
    return 1;
  }
}
