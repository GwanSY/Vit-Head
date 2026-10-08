#include "hetero_plan.h"
#include <iostream>
#include <sstream>
#include <stdexcept>
using namespace ours_vit;
template<class F> void rejects(F f) {
  try { f(); } catch (const std::exception &) { return; }
  throw std::runtime_error("invalid plan/version accepted");
}
HeteroPlan mixed(bool duplicate = false, bool split = false) {
  std::ostringstream out;
  out << "OURS_VIT_HEAD_PLAN_V1\nlayers=12,heads=12,logical_tokens=197,physical_tokens=256\n"
      << "layer,physical_head,original_head,protocol,alpha\n";
  for (int l=0;l<12;++l) for (int h=0;h<12;++h)
    out << l << ',' << h << ',' << (duplicate ? 0 : h) << ','
        << (h<4 || (split && h==11) ? "softmax_mpc,1" : h<8 ? "relu_norm_mpc,1" : "he_linear,0.125") << '\n';
  std::istringstream in(out.str());return HeteroPlan::parse(in,"test");
}
int main() {
  try {
    const auto softmax=HeteroPlan::all_softmax(), plan=mixed();
    for (auto v : {Version::B,Version::H,Version::HA,Version::HP,Version::HAP,Version::J,Version::BP}) {
      if (parse_version(to_string(v))!=v) throw std::runtime_error("version roundtrip");
      softmax.validate(v); // all-Softmax allocator fallback is legal
      if (v==Version::B || v==Version::BP) rejects([&]{plan.validate(v);});
      else plan.validate(v);
      rejects([&]{mixed(true).validate(v);});
      rejects([&]{mixed(false,true).validate(v);});
    }
    if (softmax.has_linear() || !plan.has_linear()) throw std::runtime_error("native capability");
    for (const auto *old : {"a0","A1","a2","A3","a4","unknown"}) rejects([&]{parse_version(old);});
    if (uses_packing(Version::B) || uses_packing(Version::HA) || !uses_packing(Version::HAP))
      throw std::runtime_error("packing matrix");
    std::cout << "VERSION_PLAN_TEST PASS\n";
  } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
