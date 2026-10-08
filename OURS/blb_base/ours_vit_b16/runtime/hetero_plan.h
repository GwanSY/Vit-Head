#pragma once

#include <iosfwd>
#include <string>
#include <vector>

namespace ours_vit {

constexpr int kPlanLayers = 12;
constexpr int kPlanHeads = 12;
constexpr int kPlanLogicalTokens = 197;
constexpr int kPlanPhysicalTokens = 256;

enum class Version { B, H, HA, HP, HAP, J, BP };

inline bool uses_packing(Version v) {
  return v == Version::HP || v == Version::HAP || v == Version::J || v == Version::BP;
}
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
  static HeteroPlan all_softmax();
  static HeteroPlan load_file(const std::string &path);
  static HeteroPlan parse(std::istream &input, const std::string &source);

  bool has_linear() const;
  void validate(Version version) const;
  void write_audit_csv(const std::string &path, Version version) const;

  const std::vector<LayerPlan> &layers() const { return layers_; }

 private:
  std::vector<LayerPlan> layers_;
};

Version parse_version(const std::string &value);
const char *to_string(Version value);
const char *to_string(HeadProtocol value);

}  // namespace ours_vit
