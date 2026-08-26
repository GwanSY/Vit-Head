#include "bolt_vit_mpc.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <stdexcept>
#include <thread>

using namespace sci;

namespace {

std::pair<std::size_t, std::size_t> chunk(std::size_t size, int tid) {
  return {size * static_cast<std::size_t>(tid) /
              BoltVitMPC::kThreads,
          size * static_cast<std::size_t>(tid + 1) /
              BoltVitMPC::kThreads};
}

int alternating_party(int party, int tid) {
  return (tid & 1) ? 3 - party : party;
}

} // namespace

BoltVitMPC::BoltVitMPC(int party, const std::string &address, int first_port)
    : party_(party), io_(kThreads), ot_(kThreads), fp_(kThreads) {
  for (int tid = 0; tid < kThreads; ++tid) {
    io_[tid] = std::make_unique<IOPack>(party, first_port + tid, address);
    const int protocol_party = alternating_party(party, tid);
    ot_[tid] = std::make_unique<OTPack>(io_[tid].get(), protocol_party);
    fp_[tid] = std::make_unique<FPMath>(protocol_party, io_[tid].get(),
                                        ot_[tid].get());
  }
}

std::uint64_t BoltVitMPC::communication() const {
  std::uint64_t result = 0;
  for (const auto &channel : io_) result += channel->get_comm();
  return result;
}

std::uint64_t BoltVitMPC::rounds() const {
  std::uint64_t result = 0;
  for (const auto &channel : io_) result += channel->get_rounds();
  return result;
}

std::vector<FixArray> BoltVitMPC::layer_norm(
    const std::vector<FixArray> &rows, std::uint64_t epsilon_lsb) {
  if (rows.empty()) return {};
  const int dim = rows.front().size;
  std::vector<FixArray> result(rows.size());
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int tid = 0; tid < kThreads; ++tid) {
    threads.emplace_back([&, tid] {
      const auto [begin, end] = chunk(rows.size(), tid);
      if (begin == end) return;
      const int protocol_party = alternating_party(party_, tid);
      std::vector<FixArray> input;
      input.reserve(end - begin);
      for (std::size_t row = begin; row < end; ++row) {
        if (rows[row].size != dim)
          throw std::invalid_argument("BOLT ViT MPC LayerNorm row mismatch");
        input.push_back(fp_[tid]->fix->input(
            protocol_party, dim, rows[row].data, true, rows[row].ell,
            rows[row].s));
      }
      FixArray zero_weight(protocol_party,
                           static_cast<int>((end - begin) * dim), true,
                           rows[begin].ell, rows[begin].s);
      FixArray zero_bias(protocol_party,
                         static_cast<int>((end - begin) * dim), true,
                         rows[begin].ell, rows[begin].s);
      std::fill(zero_weight.data,
                zero_weight.data + zero_weight.size, std::uint64_t(0));
      std::fill(zero_bias.data, zero_bias.data + zero_bias.size,
                std::uint64_t(0));
      const std::vector<FixArray> normalized =
          fp_[tid]->layer_norm_fix(input, zero_weight, zero_bias,
                                   epsilon_lsb);
      for (std::size_t i = 0; i < normalized.size(); ++i) {
        result[begin + i] = FixArray(party_, dim, true, normalized[i].ell,
                                     normalized[i].s);
        std::memcpy(result[begin + i].data, normalized[i].data,
                    dim * sizeof(std::uint64_t));
      }
    });
  }
  for (auto &thread : threads) thread.join();
  return result;
}

std::vector<FixArray> BoltVitMPC::softmax(
    const std::vector<FixArray> &rows) {
  if (rows.empty()) return {};
  const int width = rows.front().size;
  std::vector<FixArray> result(rows.size());
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int tid = 0; tid < kThreads; ++tid) {
    threads.emplace_back([&, tid] {
      const auto [begin, end] = chunk(rows.size(), tid);
      if (begin == end) return;
      const int protocol_party = alternating_party(party_, tid);
      std::vector<FixArray> input;
      input.reserve(end - begin);
      for (std::size_t row = begin; row < end; ++row) {
        if (rows[row].size != width)
          throw std::invalid_argument("BOLT ViT MPC Softmax row mismatch");
        input.push_back(fp_[tid]->fix->input(
            protocol_party, width, rows[row].data, true, rows[row].ell,
            rows[row].s));
      }
      FixArray exponent_shift;
      std::vector<FixArray> output;
      std::tie(output, exponent_shift) = fp_[tid]->softmax_fix(input);
      for (std::size_t i = 0; i < output.size(); ++i) {
        result[begin + i] =
            FixArray(party_, width, true, output[i].ell, output[i].s);
        std::memcpy(result[begin + i].data, output[i].data,
                    width * sizeof(std::uint64_t));
      }
    });
  }
  for (auto &thread : threads) thread.join();
  return result;
}

FixArray BoltVitMPC::gelu(const FixArray &input) {
  FixArray result(party_, input.size, true, input.ell, input.s);
  std::atomic<int> output_ell(input.ell);
  std::atomic<int> output_scale(input.s);
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int tid = 0; tid < kThreads; ++tid) {
    threads.emplace_back([&, tid] {
      const auto [begin, end] = chunk(input.size, tid);
      if (begin == end) return;
      const int protocol_party = alternating_party(party_, tid);
      FixArray local = fp_[tid]->fix->input(
          protocol_party, static_cast<int>(end - begin), input.data + begin,
          true, input.ell, input.s);
      FixArray output = fp_[tid]->gelu_approx_2(local);
      // SCI's shared BERT GELU helper materializes its historical output in
      // Z2^37.  ViT uses the wider Z2^38 ring required by its p=45b HE
      // boundary; this is only a sign-preserving representation extension,
      // and leaves the BOLT GELU approximation and scales unchanged.
      if (output.ell < 38)
        output = fp_[tid]->fix->extend(output, 38);
      std::memcpy(result.data + begin, output.data,
                  (end - begin) * sizeof(std::uint64_t));
      output_ell.store(output.ell, std::memory_order_relaxed);
      output_scale.store(output.s, std::memory_order_relaxed);
    });
  }
  for (auto &thread : threads) thread.join();
  result.ell = output_ell.load(std::memory_order_relaxed);
  result.s = output_scale.load(std::memory_order_relaxed);
  return result;
}

FixArray BoltVitMPC::gt_p_sub(const FixArray &input, std::uint64_t p,
                              int output_scale) {
  FixArray result(party_, input.size, true, input.ell, output_scale);
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int tid = 0; tid < kThreads; ++tid) {
    threads.emplace_back([&, tid] {
      const auto [begin, end] = chunk(input.size, tid);
      if (begin == end) return;
      const int protocol_party = alternating_party(party_, tid);
      const int count = static_cast<int>(end - begin);
      FixArray local = fp_[tid]->fix->input(
          protocol_party, count, input.data + begin, true, input.ell,
          input.s);
      FixArray p_array = fp_[tid]->fix->input(PUBLIC, count, p, true,
                                              input.ell, input.s);
      FixArray half_array = fp_[tid]->fix->input(
          PUBLIC, count, (p - 1) / 2, true, input.ell, input.s);
      FixArray converted = fp_[tid]->gt_p_sub(local, p_array);
      converted = fp_[tid]->fix->sub(converted, half_array);
      if (input.s > output_scale)
        converted = fp_[tid]->fix->right_shift(
            converted, input.s - output_scale);
      else if (input.s < output_scale)
        converted = fp_[tid]->fix->mul(
            converted, 1 << (output_scale - input.s));
      std::memcpy(result.data + begin, converted.data,
                  (end - begin) * sizeof(std::uint64_t));
    });
  }
  for (auto &thread : threads) thread.join();
  return result;
}

FixArray BoltVitMPC::right_shift(const FixArray &input, int amount) {
  if (amount <= 0) return input;
  FixArray result(party_, input.size, true, input.ell, input.s - amount);
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int tid = 0; tid < kThreads; ++tid) {
    threads.emplace_back([&, tid] {
      const auto [begin, end] = chunk(input.size, tid);
      if (begin == end) return;
      const int protocol_party = alternating_party(party_, tid);
      FixArray local = fp_[tid]->fix->input(
          protocol_party, static_cast<int>(end - begin), input.data + begin,
          true, input.ell, input.s);
      FixArray shifted = fp_[tid]->fix->right_shift(local, amount);
      std::memcpy(result.data + begin, shifted.data,
                  (end - begin) * sizeof(std::uint64_t));
    });
  }
  for (auto &thread : threads) thread.join();
  return result;
}

FixArray BoltVitMPC::reduce(const FixArray &input, int output_ell) {
  FixArray result(party_, input.size, true, output_ell, input.s);
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int tid = 0; tid < kThreads; ++tid) {
    threads.emplace_back([&, tid] {
      const auto [begin, end] = chunk(input.size, tid);
      if (begin == end) return;
      const int protocol_party = alternating_party(party_, tid);
      FixArray local = fp_[tid]->fix->input(
          protocol_party, static_cast<int>(end - begin), input.data + begin,
          true, input.ell, input.s);
      FixArray reduced = fp_[tid]->fix->reduce(local, output_ell);
      std::memcpy(result.data + begin, reduced.data,
                  (end - begin) * sizeof(std::uint64_t));
    });
  }
  for (auto &thread : threads) thread.join();
  return result;
}

FixArray BoltVitMPC::signed_extend_to_64(const FixArray &input) {
  if (input.party != party_ || !input.signed_ || input.ell <= 0 ||
      input.ell >= 64)
    throw std::invalid_argument(
        "BOLT ViT signed ring extension requires a local signed share");
  FixArray result(party_, input.size, true, 64, input.s);
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int tid = 0; tid < kThreads; ++tid) {
    threads.emplace_back([&, tid] {
      const auto [begin, end] = chunk(input.size, tid);
      if (begin == end) return;
      const int protocol_party = alternating_party(party_, tid);
      FixArray local = fp_[tid]->fix->input(
          protocol_party, static_cast<int>(end - begin), input.data + begin,
          true, input.ell, input.s);
      FixArray extended = fp_[tid]->fix->extend(local, 64);
      std::memcpy(result.data + begin, extended.data,
                  (end - begin) * sizeof(std::uint64_t));
    });
  }
  for (auto &thread : threads) thread.join();
  return result;
}
