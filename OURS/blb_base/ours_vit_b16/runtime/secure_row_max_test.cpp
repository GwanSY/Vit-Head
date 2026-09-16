#include "ckks_bert.h"
#include "secure_row_max.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

int bitlength = 37;
void max_thread(int, int, uint64_t *, uint64_t *, int, int, int, int,
                FPMath *);

int main(int argc, char **argv) {
  int party = 0, port = 0;
  std::string address = "127.0.0.1";
  ArgMapping args;
  args.arg("r", party, "party");
  args.arg("p", port, "port");
  args.parse(argc, argv);
  if ((party != 1 && party != 2) || port <= 0) return 2;
  sci::IOPack io(party, port, address);
  sci::OTPack ot(&io, party);
  FPMath math(party, &io, &ot);
  constexpr int ell = 37, scale = 12;
  constexpr uint64_t mask = (uint64_t{1} << ell) - 1;
  uint64_t bad = 0;
  for (int rows : {1, 3, 17}) {
    for (int keys : {1, 2, 3, 5, 128, 197, 256}) {
      std::vector<uint64_t> shares(rows * keys), output(rows);
      std::vector<int64_t> expected(rows, -(int64_t{1} << 36));
      for (int row = 0; row < rows; ++row) {
        for (int key = 0; key < keys; ++key) {
          const int index = row * keys + key;
          const int64_t value = (row == 0 ? -7 :
              ((row * 71 + key * 13) % 97 - 48)) * (int64_t{1} << scale);
          expected[row] = std::max(expected[row], value);
          const uint64_t alice = (uint64_t{0x987654321} +
              uint64_t(index + 1) * 313371337) & mask;
          shares[index] = party == 1 ? alice : (uint64_t(value) - alice) & mask;
        }
      }
      const auto started = std::chrono::steady_clock::now();
      max_thread(0, party, shares.data(), output.data(), rows, keys, ell, scale, &math);
      FixArray local(party, rows, true, ell, scale);
      std::copy(output.begin(), output.end(), local.data);
      auto opened = math.fix->output(sci::PUBLIC, local);
      for (int row = 0; row < rows; ++row)
        bad += (opened.data[row] & mask) != (uint64_t(expected[row]) & mask);
      std::cout << "ROW_MAX_TEST rows=" << rows << " keys=" << keys
                << " cumulative_bad=" << bad << " wall_s="
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
                << std::endl;
    }
  }
  std::cout << "ROW_MAX_TEST status=" << (bad == 0 ? "PASS" : "FAIL")
            << " fixture=public_deterministic" << std::endl;
  return bad == 0 ? 0 : 1;
}
