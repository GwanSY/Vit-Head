#pragma once

#include <iosfwd>
#include <string>
#include <vector>

namespace ours_vit {

constexpr int kPlanLayers = 12;
constexpr int kPlanHeads = 12;
constexpr int kPlanLogicalTokens = 197;
constexpr int kPlanPhysicalTokens = 256;

enum class Ablation { A0, A1, A2, A3, A4 };
enum class HeadProtocol { SoftmaxMpc, ReluNormMpc, HeLinear };

struct HeadSpec {
  int original_head = -1;
  HeadProtocol protocol = HeadProtocol::SoftmaxMpc;
  double alpha = 1.0;
};

struct LayerPlan {
  std::vector<HeadSpec> heads;
};

class HeteroPlan {
 public:
  static HeteroPlan all_softmax_a0();
  static HeteroPlan load_file(const std::string &path);
  static HeteroPlan parse(std::istream &input, const std::string &source);

  void validate(Ablation ablation) const;
  void write_audit_csv(const std::string &path, Ablation ablation) const;

  const std::vector<LayerPlan> &layers() const { return layers_; }

 private:
  std::vector<LayerPlan> layers_;
};

Ablation parse_ablation(const std::string &value);
const char *to_string(Ablation value);
const char *to_string(HeadProtocol value);

}  // namespace ours_vit
