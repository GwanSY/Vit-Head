#include "ciphertext.h"
#include "context.cuh"
#include "secretkey.h"

#include <dlfcn.h>
#include <execinfo.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>

namespace {

using BinaryOperation = void (*)(const PhantomContext &, PhantomCiphertext &,
                                 const PhantomCiphertext &);

std::atomic<bool> first_mismatch_seen{false};
std::atomic<unsigned> trace_event_count{0};
std::atomic<unsigned> mismatch_count{0};
std::mutex log_mutex;

std::string output_path() {
    const char *base = std::getenv("BLB_VIT_CIPHERTEXT_DIAGNOSTIC_LOG");
    if (base == nullptr || *base == '\0') return {};
    return std::string(base) + "." + std::to_string(getpid()) + ".log";
}

bool environment_flag(const char *name) {
    const char *value = std::getenv(name);
    if (value == nullptr) return false;
    return std::string(value) == "1" || std::string(value) == "true" ||
           std::string(value) == "yes";
}

double scale_bits(double scale) {
    if (!std::isfinite(scale) || scale <= 0.0)
        return std::numeric_limits<double>::quiet_NaN();
    return std::log2(scale);
}

std::string context_fingerprint(const PhantomContext &context) {
    // PhantomCiphertext deliberately does not carry an owner context or a
    // SEAL parms_id.  Fingerprint every available context level so that a
    // context created with a different prime schedule is still visible in
    // the diagnostic record.
    try {
        std::uint64_t hash = 1469598103934665603ULL;
        std::ostringstream summary;
        summary << "levels=" << context.context_data_.size();
        for (std::size_t chain = 0; chain < context.context_data_.size();
             ++chain) {
            const auto &parms = context.get_context_data(chain).parms();
            hash ^= static_cast<std::uint64_t>(chain);
            hash *= 1099511628211ULL;
            hash ^= static_cast<std::uint64_t>(parms.poly_modulus_degree());
            hash *= 1099511628211ULL;
            hash ^= static_cast<std::uint64_t>(parms.scheme());
            hash *= 1099511628211ULL;
            for (const auto &modulus : parms.coeff_modulus()) {
                hash ^= modulus.value();
                hash *= 1099511628211ULL;
            }
        }
        summary << " fnv64=" << std::hex << hash << std::dec;
        return summary.str();
    } catch (const std::exception &error) {
        return std::string("unavailable=") + error.what();
    } catch (...) {
        return "unavailable=unknown";
    }
}

std::string parameter_fingerprint(const PhantomContext &context,
                                  std::size_t chain) {
    // Phantom stores a chain index rather than SEAL's 256-bit parms_id.  The
    // modulus tuple is the exact parameter identity available at this device
    // boundary; include it as a stable, explicit fingerprint.
    try {
        const auto &moduli =
            context.get_context_data(chain).parms().coeff_modulus();
        std::uint64_t hash = 1469598103934665603ULL;
        std::ostringstream tuple;
        tuple << "[";
        for (std::size_t i = 0; i < moduli.size(); ++i) {
            if (i) tuple << ",";
            const std::uint64_t value = moduli[i].value();
            tuple << value;
            hash ^= value;
            hash *= 1099511628211ULL;
        }
        tuple << "]";
        std::ostringstream result;
        result << "fnv64=" << std::hex << hash << std::dec
               << " modulus_tuple=" << tuple.str();
        return result.str();
    } catch (const std::exception &error) {
        return std::string("unavailable=") + error.what();
    } catch (...) {
        return "unavailable=unknown";
    }
}

void write_operand(std::FILE *file, const char *name,
                   const PhantomCiphertext &cipher,
                   const PhantomContext &context) {
    std::fprintf(
        file,
        "%s object=%p data=%p context=%p context_parameter_id=%s "
        "parameter_id=unavailable_at_phantom_boundary chain=%zu size=%zu "
        "poly_degree=%zu remaining_moduli=%zu scale=%.17g scale_bits=%.17g "
        "ntt=%u noise_scale_degree=%zu parameter_fingerprint=%s\n",
        name, static_cast<const void *>(&cipher),
        static_cast<const void *>(cipher.data()),
        static_cast<const void *>(&context), context_fingerprint(context).c_str(),
        cipher.chain_index(), cipher.size(), cipher.poly_modulus_degree(),
        cipher.coeff_modulus_size(), cipher.scale(), scale_bits(cipher.scale()),
        static_cast<unsigned>(cipher.is_ntt_form()), cipher.GetNoiseScaleDeg(),
        parameter_fingerprint(context, cipher.chain_index()).c_str());
}

void write_context(std::FILE *file, const PhantomContext &context) {
    std::fprintf(
        file,
        "context_record context=%p context_parameter_id=%s "
        "per_operand_owner=not_stored_by_phantom "
        "seal_parms_id=unavailable_at_phantom_boundary\n",
        static_cast<const void *>(&context), context_fingerprint(context).c_str());
}

void write_stack(std::FILE *file) {
    void *frames[32];
    const int count = backtrace(frames, 32);
    char **symbols = backtrace_symbols(frames, count);
    if (symbols == nullptr) return;
    for (int i = 0; i < count; ++i)
        std::fprintf(file, "stack[%d]=%s\n", i, symbols[i]);
    std::free(symbols);
}

void trace_unary(const char *operation, const PhantomContext &context,
                 const PhantomCiphertext &before,
                 const PhantomCiphertext &after) {
    // Keep the trace focused on the upper native-native levels.  The first
    // mismatch is at chain 7/8; lower setup traffic is irrelevant and can
    // otherwise make the trace enormous.
    if (!environment_flag("BLB_VIT_CIPHERTEXT_DIAGNOSTIC_TRACE_ALL") &&
        before.chain_index() < 4 && after.chain_index() < 4)
        return;
    const unsigned ordinal = trace_event_count.fetch_add(1);
    if (ordinal >= 512) return;
    const std::string path = output_path();
    std::FILE *file = path.empty() ? stderr : std::fopen(path.c_str(), "a");
    if (file == nullptr) file = stderr;
    std::lock_guard<std::mutex> guard(log_mutex);
    std::fprintf(
        file,
        "OURS_NATIVE_CIPHERTEXT_TRACE sequence=%u operation=%s "
        "before_object=%p "
        "after_object=%p before_chain=%zu "
        "before_limbs=%zu before_scale=%.17g before_scale_bits=%.17g "
        "before_ntt=%u after_chain=%zu after_limbs=%zu "
        "after_scale=%.17g after_scale_bits=%.17g after_ntt=%u "
        "context=%p context_parameter_id=%s before_parameter=%s "
        "after_parameter=%s\n",
        ordinal, operation, static_cast<const void *>(&before),
        static_cast<const void *>(&after), before.chain_index(),
        before.coeff_modulus_size(), before.scale(), scale_bits(before.scale()),
        static_cast<unsigned>(before.is_ntt_form()), after.chain_index(),
        after.coeff_modulus_size(), after.scale(), scale_bits(after.scale()),
        static_cast<unsigned>(after.is_ntt_form()),
        static_cast<const void *>(&context),
        context_fingerprint(context).c_str(),
        parameter_fingerprint(context, before.chain_index()).c_str(),
        parameter_fingerprint(context, after.chain_index()).c_str());
    if (ordinal < 24) write_stack(file);
    std::fflush(file);
    if (file != stderr) std::fclose(file);
}

void trace_plain(const char *operation, const PhantomContext &context,
                 const PhantomCiphertext &cipher,
                 const PhantomPlaintext &plain) {
    if (!environment_flag("BLB_VIT_CIPHERTEXT_DIAGNOSTIC_TRACE_ALL") &&
        cipher.chain_index() < 4 && plain.chain_index() < 4)
        return;
    const unsigned ordinal = trace_event_count.fetch_add(1);
    if (ordinal >= 512) return;
    const std::string path = output_path();
    std::FILE *file = path.empty() ? stderr : std::fopen(path.c_str(), "a");
    if (file == nullptr) file = stderr;
    std::lock_guard<std::mutex> guard(log_mutex);
    std::fprintf(
        file,
        "OURS_NATIVE_CIPHERTEXT_TRACE sequence=%u operation=%s "
        "cipher_object=%p "
        "plain_object=%p cipher_chain=%zu cipher_limbs=%zu "
        "cipher_scale=%.17g cipher_scale_bits=%.17g cipher_ntt=%u "
        "plain_chain=%zu plain_coefficients=%zu plain_scale=%.17g "
        "context=%p context_parameter_id=%s cipher_parameter=%s "
        "plain_parameter=%s\n",
        ordinal, operation, static_cast<const void *>(&cipher),
        static_cast<const void *>(&plain), cipher.chain_index(),
        cipher.coeff_modulus_size(), cipher.scale(), scale_bits(cipher.scale()),
        static_cast<unsigned>(cipher.is_ntt_form()), plain.chain_index(),
        plain.coeff_count(), plain.scale(),
        static_cast<const void *>(&context),
        context_fingerprint(context).c_str(),
        parameter_fingerprint(context, cipher.chain_index()).c_str(),
        parameter_fingerprint(context, plain.chain_index()).c_str());
    if (ordinal < 24) write_stack(file);
    std::fflush(file);
    if (file != stderr) std::fclose(file);
}

void trace_binary(const char *operation, const PhantomContext &context,
                  const PhantomCiphertext &first,
                  const PhantomCiphertext &second) {
    // The object addresses let the offline reader follow the two generated
    // paths through copies, public-one products and level switches.  Keep the
    // default trace focused on the native native levels; an explicit environment
    // flag enables the complete operation stream when needed.
    if (!environment_flag("BLB_VIT_CIPHERTEXT_DIAGNOSTIC_TRACE_ALL") &&
        first.chain_index() < 4 && second.chain_index() < 4)
        return;
    const unsigned ordinal = trace_event_count.fetch_add(1);
    if (ordinal >= 512) return;
    const std::string path = output_path();
    std::FILE *file = path.empty() ? stderr : std::fopen(path.c_str(), "a");
    if (file == nullptr) file = stderr;
    std::lock_guard<std::mutex> guard(log_mutex);
    std::fprintf(file,
                 "OURS_NATIVE_CIPHERTEXT_TRACE sequence=%u operation=%s "
                 "phase=before first_context_equals_second_argument=1\n",
                 ordinal, operation);
    write_operand(file, "first", first, context);
    write_operand(file, "second", second, context);
    if (ordinal < 24) write_stack(file);
    std::fflush(file);
    if (file != stderr) std::fclose(file);
}

void check_and_log(const char *operation, const PhantomContext &context,
                   const PhantomCiphertext &first,
                   const PhantomCiphertext &second) {
    const bool chain_mismatch = first.chain_index() != second.chain_index();
    const bool ntt_mismatch = first.is_ntt_form() != second.is_ntt_form();
    const bool limb_mismatch =
        first.coeff_modulus_size() != second.coeff_modulus_size();
    const bool poly_mismatch =
        first.poly_modulus_degree() != second.poly_modulus_degree();
    const bool size_mismatch = first.size() != second.size();
    const double scale_reference =
        std::max(std::abs(first.scale()), std::abs(second.scale()));
    const bool scale_mismatch =
        !std::isfinite(first.scale()) || !std::isfinite(second.scale()) ||
        scale_reference == 0.0 ||
        std::abs(first.scale() - second.scale()) /
                std::max(scale_reference, 1.0) >
            1e-12;
    if (!chain_mismatch && !ntt_mismatch && !limb_mismatch &&
        !poly_mismatch && !size_mismatch && !scale_mismatch)
        return;
    const bool is_first = !first_mismatch_seen.exchange(true);
    ++mismatch_count;
    if (!is_first &&
        !environment_flag("BLB_VIT_CIPHERTEXT_DIAGNOSTIC_LOG_ALL_MISMATCHES"))
        return;

    const std::string path = output_path();
    std::FILE *file = path.empty() ? stderr : std::fopen(path.c_str(), "a");
    if (file == nullptr) file = stderr;
    std::lock_guard<std::mutex> guard(log_mutex);
    std::fprintf(file,
                 "OURS_NATIVE_CIPHERTEXT_MISMATCH operation=%s pid=%d "
                 "first=%d mismatch_index=%u context_argument_shared=1\n",
                 operation, static_cast<int>(getpid()), is_first,
                 mismatch_count.load());
    std::fprintf(file,
                 "reason=chain:%d ntt:%d remaining_moduli:%d poly:%d size:%d "
                 "scale:%d parameter_identity=unavailable "
                 "per_operand_context=not_available_at_phantom_boundary\n",
                 chain_mismatch, ntt_mismatch, limb_mismatch, poly_mismatch,
                 size_mismatch, scale_mismatch);
    write_context(file, context);
    write_operand(file, "operand1", first, context);
    write_operand(file, "operand2", second, context);
    write_stack(file);
    std::fflush(file);
    if (file != stderr) std::fclose(file);
}

BinaryOperation resolve(const char *symbol) {
    void *address = dlsym(RTLD_NEXT, symbol);
    if (address == nullptr) {
        std::fprintf(stderr, "native diagnostic interposer cannot resolve %s: %s\n",
                     symbol, dlerror());
        std::abort();
    }
    return reinterpret_cast<BinaryOperation>(address);
}

}  // namespace

extern "C" void phantom_add_inplace_diagnostic(
    const PhantomContext &context, PhantomCiphertext &first,
    const PhantomCiphertext &second)
    __asm__("_ZN7phantom11add_inplaceERK14PhantomContextR17PhantomCiphertextRKS3_");

extern "C" void phantom_add_inplace_diagnostic(
    const PhantomContext &context, PhantomCiphertext &first,
    const PhantomCiphertext &second) {
    trace_binary("add", context, first, second);
    check_and_log("add", context, first, second);
    static BinaryOperation real = resolve(
        "_ZN7phantom11add_inplaceERK14PhantomContextR17PhantomCiphertextRKS3_");
    real(context, first, second);
}

extern "C" void phantom_multiply_inplace_diagnostic(
    const PhantomContext &context, PhantomCiphertext &first,
    const PhantomCiphertext &second)
    __asm__("_ZN7phantom16multiply_inplaceERK14PhantomContextR17PhantomCiphertextRKS3_");

extern "C" void phantom_multiply_inplace_diagnostic(
    const PhantomContext &context, PhantomCiphertext &first,
    const PhantomCiphertext &second) {
    trace_binary("multiply", context, first, second);
    check_and_log("multiply", context, first, second);
    static BinaryOperation real = resolve(
        "_ZN7phantom16multiply_inplaceERK14PhantomContextR17PhantomCiphertextRKS3_");
    real(context, first, second);
}

extern "C" void phantom_multiply_and_relin_inplace_diagnostic(
    const PhantomContext &context, PhantomCiphertext &first,
    const PhantomCiphertext &second, const PhantomRelinKey &)
    __asm__("_ZN7phantom26multiply_and_relin_inplaceERK14PhantomContextR17PhantomCiphertextRKS3_RK15PhantomRelinKey");

extern "C" void phantom_multiply_and_relin_inplace_diagnostic(
    const PhantomContext &context, PhantomCiphertext &first,
    const PhantomCiphertext &second, const PhantomRelinKey &key) {
    trace_binary("multiply_and_relin", context, first, second);
    check_and_log("multiply_and_relin", context, first, second);
    using Operation = void (*)(const PhantomContext &, PhantomCiphertext &,
                               const PhantomCiphertext &,
                               const PhantomRelinKey &);
    void *address = dlsym(
        RTLD_NEXT,
        "_ZN7phantom26multiply_and_relin_inplaceERK14PhantomContextR17PhantomCiphertextRKS3_RK15PhantomRelinKey");
    if (address == nullptr) {
        std::fprintf(stderr,
                     "native diagnostic interposer cannot resolve multiply_and_relin: %s\n",
                     dlerror());
        std::abort();
    }
    reinterpret_cast<Operation>(address)(context, first, second, key);
}

extern "C" PhantomCiphertext phantom_mod_switch_to_next_diagnostic(
    const PhantomContext &context, const PhantomCiphertext &cipher)
    __asm__("_ZN7phantom18mod_switch_to_nextERK14PhantomContextRK17PhantomCiphertext");

extern "C" PhantomCiphertext phantom_mod_switch_to_next_diagnostic(
    const PhantomContext &context, const PhantomCiphertext &cipher) {
    using Operation = PhantomCiphertext (*)(const PhantomContext &,
                                             const PhantomCiphertext &);
    static Operation real = reinterpret_cast<Operation>(resolve(
        "_ZN7phantom18mod_switch_to_nextERK14PhantomContextRK17PhantomCiphertext"));
    PhantomCiphertext result = real(context, cipher);
    trace_unary("mod_switch_to_next", context, cipher, result);
    return result;
}

extern "C" PhantomCiphertext phantom_rescale_to_next_diagnostic(
    const PhantomContext &context, const PhantomCiphertext &cipher)
    __asm__("_ZN7phantom15rescale_to_nextERK14PhantomContextRK17PhantomCiphertext");

extern "C" PhantomCiphertext phantom_rescale_to_next_diagnostic(
    const PhantomContext &context, const PhantomCiphertext &cipher) {
    using Operation = PhantomCiphertext (*)(const PhantomContext &,
                                             const PhantomCiphertext &);
    static Operation real = reinterpret_cast<Operation>(resolve(
        "_ZN7phantom15rescale_to_nextERK14PhantomContextRK17PhantomCiphertext"));
    PhantomCiphertext result = real(context, cipher);
    trace_unary("rescale_to_next", context, cipher, result);
    return result;
}

extern "C" void phantom_multiply_plain_inplace_diagnostic(
    const PhantomContext &context, PhantomCiphertext &cipher,
    const PhantomPlaintext &plain)
    __asm__("_ZN7phantom22multiply_plain_inplaceERK14PhantomContextR17PhantomCiphertextRK16PhantomPlaintext");

extern "C" void phantom_multiply_plain_inplace_diagnostic(
    const PhantomContext &context, PhantomCiphertext &cipher,
    const PhantomPlaintext &plain) {
    trace_plain("multiply_plain_inplace", context, cipher, plain);
    using Operation = void (*)(const PhantomContext &, PhantomCiphertext &,
                               const PhantomPlaintext &);
    static Operation real = reinterpret_cast<Operation>(resolve(
        "_ZN7phantom22multiply_plain_inplaceERK14PhantomContextR17PhantomCiphertextRK16PhantomPlaintext"));
    real(context, cipher, plain);
}
