#include "libspu/mpc/bumblebee/rlwe/phantom_evaluator.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "gtest/gtest.h"
#include "seal/seal.h"
#include "seal/util/ntt.h"
#include "seal/util/polyarithsmallmod.h"

namespace spu::mpc::bumblebee::test {
namespace {

void ExpectCipherCoefficientsEqual(const seal::SEALContext& context,
                                   const seal::Ciphertext& expected,
                                   const seal::Ciphertext& actual) {
  ASSERT_EQ(expected.parms_id(), actual.parms_id());
  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.coeff_modulus_size(), actual.coeff_modulus_size());
  ASSERT_EQ(expected.poly_modulus_degree(), actual.poly_modulus_degree());
  ASSERT_EQ(expected.is_ntt_form(), actual.is_ntt_form());
  const auto context_data = context.get_context_data(expected.parms_id());
  ASSERT_NE(context_data, nullptr);
  const auto& moduli = context_data->parms().coeff_modulus();
  const auto degree = expected.poly_modulus_degree();
  for (std::size_t poly = 0; poly < expected.size(); ++poly) {
    for (std::size_t modulus = 0; modulus < moduli.size(); ++modulus) {
      const auto q = moduli[modulus].value();
      for (std::size_t coeff = 0; coeff < degree; ++coeff) {
        ASSERT_EQ(expected.data(poly)[modulus * degree + coeff] % q,
                  actual.data(poly)[modulus * degree + coeff] % q)
            << "poly=" << poly << " modulus=" << modulus
            << " coefficient=" << coeff;
      }
    }
  }
}

void SealExactModSwitchToNextInplace(const seal::SEALContext& context,
                                     seal::Ciphertext& cipher) {
  const auto current = context.get_context_data(cipher.parms_id());
  ASSERT_NE(current, nullptr);
  const auto next = current->next_context_data();
  ASSERT_NE(next, nullptr);
  const auto pool = seal::MemoryManager::GetPool();
  const auto* rns_tool = current->rns_tool();
  const auto* ntt_tables = current->small_ntt_tables();
  if (cipher.is_ntt_form()) {
    SEAL_ITERATE(seal::util::iter(cipher), cipher.size(), [&](auto item) {
      rns_tool->divide_and_round_q_last_ntt_inplace(item, ntt_tables, pool);
    });
  } else {
    SEAL_ITERATE(seal::util::iter(cipher), cipher.size(), [&](auto item) {
      rns_tool->divide_and_round_q_last_inplace(item, pool);
    });
  }

  seal::Ciphertext switched(pool);
  switched.resize(context, next->parms_id(), cipher.size());
  SEAL_ITERATE(seal::util::iter(cipher, switched), cipher.size(),
               [&](auto item) {
                 seal::util::set_poly(
                     std::get<0>(item), cipher.poly_modulus_degree(),
                     cipher.coeff_modulus_size() - 1, std::get<1>(item));
               });
  switched.is_ntt_form() = cipher.is_ntt_form();
  switched.scale() = cipher.scale();
  switched.correction_factor() = cipher.correction_factor();
  std::swap(cipher, switched);
}

TEST(PhantomEvaluatorTest, BfvOperatorsMatchSealCoefficientByCoefficient) {
  ASSERT_EQ(setenv("SPU_BUMBLEBEE_HE_BACKEND", "gpu", 1), 0);
  ASSERT_EQ(setenv("SPU_BUMBLEBEE_VERIFY_GPU", "1", 1), 0);
  ValidateBumblebeeHeBackend();

  constexpr std::size_t degree = 4096;
  seal::EncryptionParameters parameters(seal::scheme_type::bfv);
  parameters.set_use_special_prime(false);
  parameters.set_poly_modulus_degree(degree);
  parameters.set_coeff_modulus(
      seal::CoeffModulus::Create(degree, {60, 52}));
  parameters.set_plain_modulus(seal::PlainModulus::Batching(degree, 20));
  seal::SEALContext context(parameters, true, seal::sec_level_type::none);
  seal::KeyGenerator keygen(context);
  seal::Encryptor encryptor(context, keygen.secret_key());
  seal::BatchEncoder encoder(context);

  std::vector<std::uint64_t> values(degree);
  std::vector<std::uint64_t> multiplier_values(degree);
  for (std::size_t i = 0; i < degree; ++i) {
    values[i] = (17 * i + 3) % parameters.plain_modulus().value();
    multiplier_values[i] = (5 * i + 1) % parameters.plain_modulus().value();
  }
  seal::Plaintext plain;
  seal::Plaintext multiplier;
  encoder.encode(values, plain);
  encoder.encode(multiplier_values, multiplier);
  ASSERT_NE(multiplier.nonzero_coeff_count(), 1U)
      << "unexpected SEAL monomial fast path";
  seal::Ciphertext cipher;
  encryptor.encrypt_symmetric(plain, cipher);
  seal::Evaluator cpu(context);
  seal::Ciphertext expected_manual = cipher;
  const auto context_data = context.get_context_data(cipher.parms_id());
  ASSERT_NE(context_data, nullptr);
  const auto& moduli = context_data->parms().coeff_modulus();
  const auto* tables = context_data->small_ntt_tables();
  seal::Plaintext lifted;
  lifted.resize(degree * moduli.size());
  const auto threshold = context_data->plain_upper_half_threshold();
  const auto increments = context_data->plain_upper_half_increment();
  for (std::size_t modulus = 0; modulus < moduli.size(); ++modulus) {
    for (std::size_t coeff = 0; coeff < multiplier.coeff_count(); ++coeff) {
      const auto value = multiplier[coeff];
      lifted[modulus * degree + coeff] =
          value >= threshold ? value + increments[modulus] : value;
    }
    seal::util::ntt_negacyclic_harvey(
        lifted.data() + modulus * degree, tables[modulus]);
  }
  for (std::size_t poly = 0; poly < expected_manual.size(); ++poly) {
    for (std::size_t modulus = 0; modulus < moduli.size(); ++modulus) {
      auto* component = expected_manual.data(poly) + modulus * degree;
      seal::util::ntt_negacyclic_harvey_lazy(component, tables[modulus]);
      seal::util::dyadic_product_coeffmod(
          component, lifted.data() + modulus * degree, degree,
          moduli[modulus], component);
      seal::util::inverse_ntt_negacyclic_harvey(component, tables[modulus]);
    }
  }
  seal::Ciphertext expected_seal = cipher;
  cpu.multiply_plain_inplace(expected_seal, multiplier);
  ExpectCipherCoefficientsEqual(context, expected_seal, expected_manual);

  PhantomEvaluator gpu(context);
  ASSERT_TRUE(gpu.using_gpu());
  seal::Ciphertext sum = cipher;
  seal::Ciphertext expected_sum = cipher;
  cpu.add_inplace(expected_sum, cipher);
  gpu.add_inplace(sum, cipher);
  ExpectCipherCoefficientsEqual(context, expected_sum, sum);
  gpu.sub_inplace(sum, cipher);
  ExpectCipherCoefficientsEqual(context, cipher, sum);

  seal::Ciphertext negated;
  seal::Ciphertext expected_negated;
  cpu.negate(cipher, expected_negated);
  gpu.negate(cipher, negated);
  ExpectCipherCoefficientsEqual(context, expected_negated, negated);

  seal::Ciphertext ntt_probe = cipher;
  gpu.ntt_inplace(ntt_probe);
  gpu.inverse_ntt_inplace(ntt_probe);
  gpu.add_plain_inplace(cipher, multiplier);
  gpu.sub_plain_inplace(cipher, multiplier);
  gpu.multiply_plain_inplace(cipher, multiplier);
  gpu.multiply_sub_plain_inplace(cipher, multiplier, plain);
}

TEST(PhantomEvaluatorTest, ExactCkksNttAndModSwitchMatchSeal) {
  ASSERT_EQ(setenv("SPU_BUMBLEBEE_HE_BACKEND", "gpu", 1), 0);
  ASSERT_EQ(setenv("SPU_BUMBLEBEE_VERIFY_GPU", "1", 1), 0);
  ValidateBumblebeeHeBackend();

  constexpr std::size_t degree = 4096;
  seal::EncryptionParameters parameters(seal::scheme_type::ckks);
  parameters.set_use_special_prime(false);
  parameters.set_poly_modulus_degree(degree);
  parameters.set_coeff_modulus(
      seal::CoeffModulus::Create(degree, {59, 49, 45}));
  seal::SEALContext context(parameters, true, seal::sec_level_type::none);
  const auto first = context.first_context_data();
  ASSERT_NE(first, nullptr);

  seal::Ciphertext cipher;
  cipher.resize(context, first->parms_id(), 2);
  cipher.scale() = 1.0;
  cipher.correction_factor() = 1;
  cipher.is_ntt_form() = false;
  const auto& moduli = first->parms().coeff_modulus();
  for (std::size_t poly = 0; poly < cipher.size(); ++poly) {
    for (std::size_t modulus = 0; modulus < moduli.size(); ++modulus) {
      const auto q = moduli[modulus].value();
      for (std::size_t coeff = 0; coeff < degree; ++coeff) {
        cipher.data(poly)[modulus * degree + coeff] =
            (19 * coeff + 23 * modulus + 29 * poly + 7) % q;
      }
    }
  }

  PhantomEvaluator gpu(context);
  gpu.ntt_inplace(cipher);
  const auto ntt_form = cipher;
  gpu.inverse_ntt_inplace(cipher);

  seal::Ciphertext expected_ntt = cipher;
  seal::Evaluator cpu(context);
  cpu.transform_to_ntt_inplace(expected_ntt);
  ExpectCipherCoefficientsEqual(context, expected_ntt, ntt_form);

  seal::Ciphertext expected_switched = ntt_form;
  SealExactModSwitchToNextInplace(context, expected_switched);
  seal::Ciphertext actual_switched = ntt_form;
  gpu.mod_switch_to_inplace(actual_switched,
                            expected_switched.parms_id());
  ExpectCipherCoefficientsEqual(context, expected_switched, actual_switched);

  seal::Ciphertext expected_coeff_switched = cipher;
  SealExactModSwitchToNextInplace(context, expected_coeff_switched);
  seal::Ciphertext actual_coeff_switched = cipher;
  gpu.mod_switch_to_inplace(actual_coeff_switched,
                            expected_coeff_switched.parms_id());
  ExpectCipherCoefficientsEqual(context, expected_coeff_switched,
                                actual_coeff_switched);

  seal::Plaintext plain;
  plain.resize(degree * moduli.size());
  plain.parms_id() = first->parms_id();
  plain.scale() = 1.0;
  for (std::size_t modulus = 0; modulus < moduli.size(); ++modulus) {
    const auto q = moduli[modulus].value();
    for (std::size_t coeff = 0; coeff < degree; ++coeff) {
      plain[modulus * degree + coeff] =
          (31 * coeff + 11 * modulus + 5) % q;
    }
  }

  seal::Ciphertext expected_added = cipher;
  for (std::size_t modulus = 0; modulus < moduli.size(); ++modulus) {
    const auto q = moduli[modulus].value();
    for (std::size_t coeff = 0; coeff < degree; ++coeff) {
      auto sum = expected_added.data(0)[modulus * degree + coeff] +
                 plain[modulus * degree + coeff];
      if (sum >= q) {
        sum -= q;
      }
      expected_added.data(0)[modulus * degree + coeff] = sum;
    }
  }
  seal::Ciphertext actual_added = cipher;
  gpu.add_plain_inplace(actual_added, plain);
  ExpectCipherCoefficientsEqual(context, expected_added, actual_added);
  gpu.sub_plain_inplace(actual_added, plain);
  ExpectCipherCoefficientsEqual(context, cipher, actual_added);

  gpu.ntt_inplace(plain);
  seal::Ciphertext expected_product = ntt_form;
  cpu.multiply_plain_inplace(expected_product, plain);
  seal::Ciphertext actual_product = ntt_form;
  gpu.multiply_plain_inplace(actual_product, plain);
  ExpectCipherCoefficientsEqual(context, expected_product, actual_product);

  const std::array<seal::Ciphertext, 2> sequence_ciphers = {ntt_form,
                                                            ntt_form};
  const std::array<seal::Plaintext, 2> sequence_plains = {plain, plain};
  seal::Ciphertext expected_sequence;
  cpu.multiply_plain(sequence_ciphers[0], sequence_plains[0],
                     expected_sequence);
  seal::Ciphertext sequence_term;
  cpu.multiply_plain(sequence_ciphers[1], sequence_plains[1], sequence_term);
  cpu.add_inplace(expected_sequence, sequence_term);
  seal::Ciphertext actual_sequence;
  gpu.multiply_add_sequence(sequence_ciphers.data(), sequence_plains.data(),
                            sequence_ciphers.size(), actual_sequence);
  ExpectCipherCoefficientsEqual(context, expected_sequence, actual_sequence);

  gpu.prepare_multiply_add_inputs(
      sequence_ciphers.data(), sequence_ciphers.size(),
      sequence_plains.data(), sequence_plains.size());
  seal::Ciphertext resident_sequence;
  gpu.multiply_add_sequence(sequence_ciphers.data(), sequence_plains.data(),
                            sequence_ciphers.size(), resident_sequence);
  gpu.clear_multiply_add_inputs();
  ExpectCipherCoefficientsEqual(context, expected_sequence,
                                resident_sequence);

  seal::Ciphertext expected_finished = expected_sequence;
  cpu.transform_from_ntt_inplace(expected_finished);
  SealExactModSwitchToNextInplace(context, expected_finished);
  seal::Ciphertext finished_sequence;
  gpu.multiply_add_sequence(sequence_ciphers.data(), sequence_plains.data(),
                            sequence_ciphers.size(), finished_sequence,
                            /*inverse_ntt_output=*/true,
                            expected_finished.parms_id());
  ExpectCipherCoefficientsEqual(context, expected_finished,
                                finished_sequence);
}

}  // namespace
}  // namespace spu::mpc::bumblebee::test
