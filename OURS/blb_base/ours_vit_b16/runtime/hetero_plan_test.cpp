#include "hetero_plan.h"

#include <iostream>
#include <sstream>
#include <stdexcept>

int main() {
  try {
    auto a0 = ours_vit::HeteroPlan::all_softmax_a0();
    a0.validate(ours_vit::Ablation::A0);

    // A1 deliberately keeps original head order while interleaving all three
    // paths.  This is the manual mixed assignment used to validate path
    // semantics independently from A2's physical grouping.
    std::ostringstream a1_text;
    a1_text << "OURS_VIT_HEAD_PLAN_V1\n"
         << "layers=12,heads=12,logical_tokens=197,physical_tokens=256\n"
         << "layer,physical_head,original_head,protocol,alpha\n";
    for (int layer = 0; layer < 12; ++layer) {
      for (int head = 0; head < 12; ++head) {
        const char *protocol = head % 3 == 0 ? "softmax_mpc" :
                               head % 3 == 1 ? "relu_norm_mpc" : "he_linear";
        a1_text << layer << ',' << head << ',' << head << ',' << protocol
                << ',' << (head % 3 == 2 ? 0.015625 : 1.0) << '\n';
      }
    }
    std::istringstream a1_input(a1_text.str());
    auto a1 = ours_vit::HeteroPlan::parse(a1_input, "a1-manual-mixed-test");
    a1.validate(ours_vit::Ablation::A1);

    // A2 represents the same original-head assignment in three contiguous
    // physical groups. original_head is therefore a synchronized permutation,
    // not a dynamic inference decision.
    std::ostringstream a2_text;
    a2_text << "OURS_VIT_HEAD_PLAN_V1\n"
            << "layers=12,heads=12,logical_tokens=197,physical_tokens=256\n"
            << "layer,physical_head,original_head,protocol,alpha\n";
    const int grouped_order[12] = {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11};
    for (int layer = 0; layer < 12; ++layer) {
      for (int physical = 0; physical < 12; ++physical) {
        const int original = grouped_order[physical];
        const char *protocol = original % 3 == 0 ? "softmax_mpc" :
                               original % 3 == 1 ? "relu_norm_mpc" :
                                                   "he_linear";
        a2_text << layer << ',' << physical << ',' << original << ','
                << protocol << ','
                << (original % 3 == 2 ? 0.015625 : 1.0) << '\n';
      }
    }
    std::istringstream a2_input(a2_text.str());
    auto a2 = ours_vit::HeteroPlan::parse(a2_input, "a2-grouped-test");
    a2.validate(ours_vit::Ablation::A2);
    a2.validate(ours_vit::Ablation::A4);

    // A4 must permit calibration to keep an entire sensitive layer on MPC.
    // Only the complete fixed plan, rather than every layer, needs a native
    // CKKS Linear head.
    std::ostringstream sparse_a4_text;
    sparse_a4_text
        << "OURS_VIT_HEAD_PLAN_V1\n"
        << "layers=12,heads=12,logical_tokens=197,physical_tokens=256\n"
        << "layer,physical_head,original_head,protocol,alpha\n";
    for (int layer = 0; layer < 12; ++layer) {
      for (int head = 0; head < 12; ++head) {
        const bool linear = layer == 0 && head == 11;
        sparse_a4_text << layer << ',' << head << ',' << head << ','
                       << (linear ? "he_linear" : "softmax_mpc") << ','
                       << (linear ? 0.005 : 1.0) << '\n';
      }
    }
    std::istringstream sparse_a4_input(sparse_a4_text.str());
    auto sparse_a4 = ours_vit::HeteroPlan::parse(
        sparse_a4_input, "a4-sparse-linear-test");
    sparse_a4.validate(ours_vit::Ablation::A4);

    bool rejected = false;
    try {
      a0.validate(ours_vit::Ablation::A4);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("A4 accepted a plan with no native Linear head");

    // A3 is a complete grouped MPC conversion stage only when every head uses
    // a secure MPC normalization. Native Linear is completed by A4, never by
    // an MPC surrogate hidden under the A3 label.
    rejected = false;
    try {
      a2.validate(ours_vit::Ablation::A3);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("A3 accepted an unimplemented Linear surrogate");

    rejected = false;
    try {
      a1.validate(ours_vit::Ablation::A0);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    if (!rejected) throw std::runtime_error("A0 accepted heterogeneous plan");

    std::ostringstream noncontiguous_text;
    noncontiguous_text
        << "OURS_VIT_HEAD_PLAN_V1\n"
        << "layers=12,heads=12,logical_tokens=197,physical_tokens=256\n"
        << "layer,physical_head,original_head,protocol,alpha\n";
    for (int layer = 0; layer < 12; ++layer) {
      for (int head = 0; head < 12; ++head) {
        const char *protocol = head % 2 == 0 ? "softmax_mpc" :
                                               "relu_norm_mpc";
        noncontiguous_text << layer << ',' << head << ',' << head << ','
                           << protocol << ",1\n";
      }
    }
    std::istringstream noncontiguous_input(noncontiguous_text.str());
    auto noncontiguous = ours_vit::HeteroPlan::parse(
        noncontiguous_input, "noncontiguous-test");
    rejected = false;
    try {
      noncontiguous.validate(ours_vit::Ablation::A2);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("A2 accepted non-contiguous protocol groups");
    std::cout << "OURS_HETERO_PLAN_TEST status=PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "OURS_HETERO_PLAN_TEST status=FAIL reason=" << error.what()
              << '\n';
    return 1;
  }
}
