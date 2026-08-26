#include "FloatingPoint/fp-math.h"
#include "OT/ot_pack.h"
#include "utils/ArgMapping/ArgMapping.h"
#include "utils/io_pack.h"

#include <cstdint>
#include <iostream>
#include <vector>

using namespace sci;

std::uint64_t prime_mod = 536903681ULL;
std::int32_t bitlength = 38;
std::int32_t num_threads = 1;

namespace {

int party = 0;
int port = 47430;
std::string address = "127.0.0.1";

FixArray local_array(const std::vector<std::int64_t> &clear, int row,
                    int width) {
  FixArray result(party, width, true, 38, 12);
  for (int col = 0; col < width; ++col) {
    const std::int64_t value =
        party == ALICE ? clear[row * width + col] : 0;
    result.data[col] = static_cast<std::uint64_t>(value) & result.ell_mask();
  }
  return result;
}

void reveal_and_print(FPMath &fp, const std::vector<FixArray> &values,
                      const char *name) {
  for (const auto &value : values) {
    FixArray public_value = fp.fix->output(BOB, value);
    if (party == BOB) {
      std::cout << name;
      for (int index = 0; index < public_value.size; ++index)
        std::cout << ' ' << signed_val(public_value.data[index],
                                       public_value.ell);
      std::cout << '\n';
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  ArgMapping args;
  args.arg("r", party, "Role: ALICE=1, BOB=2");
  args.arg("p", port, "TCP port");
  args.arg("ip", address, "ALICE address (BOB only)");
  args.parse(argc, argv);

  IOPack io(party, port, address);
  OTPack ot(&io, party);
  FPMath fp(party, &io, &ot);

  constexpr int ln_rows = 2;
  constexpr int ln_width = 8;
  const std::vector<std::int64_t> ln_clear{
      1000, -500, 200, 700, -300, 50, 900, -1200,
      -700, 300, 1100, -200, 400, -800, 600, 100};
  std::vector<FixArray> ln_input;
  for (int row = 0; row < ln_rows; ++row)
    ln_input.push_back(local_array(ln_clear, row, ln_width));
  FixArray zero_weight(party, ln_rows * ln_width, true, 38, 12);
  FixArray zero_bias(party, ln_rows * ln_width, true, 38, 12);
  std::fill(zero_weight.data, zero_weight.data + zero_weight.size, 0);
  std::fill(zero_bias.data, zero_bias.data + zero_bias.size, 0);
  reveal_and_print(fp, fp.layer_norm_fix(ln_input, zero_weight, zero_bias, 0),
                   "LN");

  constexpr int gelu_width = 14;
  const std::vector<std::int64_t> gelu_clear{
      -16384, -8192, -4096, -1, 0, 1, 4096,
      8192, 16384, 32768, 44032, 65536, 110592, 196608};
  FixArray gelu_input = local_array(gelu_clear, 0, gelu_width);
  FixArray gelu_scale11 = fp.fix->right_shift(gelu_input, 1);
  FixArray gelu_ell20 = fp.fix->reduce(gelu_scale11, 20);
  reveal_and_print(fp, {fp.gelu_approx_2(gelu_ell20)}, "GE");

  const std::vector<std::int64_t> sqrt_clear{
      1, 2, 4096, 8192, 65536, 1000000, 10000000, 100000000};
  FixArray sqrt_input = local_array(sqrt_clear, 0, 8);
  reveal_and_print(fp, {fp.sqrt(sqrt_input, true)}, "SQ");

  constexpr int sm_rows = 2;
  constexpr int sm_width = 5;
  const std::vector<std::int64_t> sm_clear{
      0, -4096, -8192, 2048, -12288,
      4096, 0, -4096, -16384, 1024};
  const std::vector<std::int64_t> sm_shift_clear{
      0, -4096, -8192, 2048, -12288,
      0, -4096, -8192, -20480, -3072};
  std::vector<FixArray> sm_input;
  std::vector<FixArray> sm_shift_input;
  for (int row = 0; row < sm_rows; ++row)
    sm_input.push_back(local_array(sm_clear, row, sm_width));
  for (int row = 0; row < sm_rows; ++row)
    sm_shift_input.push_back(local_array(sm_shift_clear, row, sm_width));
  FixArray exp_input = concat(sm_shift_input);
  FixArray exponent;
  FixArray exponent_shift;
  std::tie(exponent, exponent_shift) = fp.exp4(exp_input);
  FixArray public_exponent = fp.fix->output(BOB, exponent);
  if (party == BOB) {
    std::cout << "EXP";
    for (int index = 0; index < public_exponent.size; ++index)
      std::cout << ' ' << signed_val(public_exponent.data[index],
                                     public_exponent.ell);
    std::cout << '\n';
  }
  FixArray ignored;
  std::vector<FixArray> sm_output;
  std::tie(sm_output, ignored) = fp.softmax_fix(sm_input);
  reveal_and_print(fp, sm_output, "SM");
  return 0;
}
