"""Build optional interval tracing without replacing the stage-one CPU library."""
import argparse
import difflib
import json
import shutil
import subprocess
from pathlib import Path
from runtime import ROOT, digest, write


def patch(path, transform):
    backup = path.with_name(path.name + ".stage2-original")
    if not backup.exists():
        shutil.copy2(path, backup)
    before = backup.read_text()
    after = transform(before)
    path.write_text(after)
    return "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                    fromfile=str(path.name), tofile=str(path.name)))


def replace(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one patch anchor: {old[:100]}")
    return text.replace(old, new)


def patch_yacl(path):
    def transform(s):
        s = replace(s, '#include <iostream>', '#include <iostream>\n#include <cstdint>')
        hook = '''// Optional SPU observer. No protocol or message order is changed.
extern "C" void SpuStage2RecordLinkEvent(const char*, const char*, const char*,
    size_t, size_t, int64_t, int64_t) __attribute__((weak));
static int64_t Stage2LinkNow() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}
'''
        s = replace(s, 'namespace yacl::link {', hook + '\nnamespace yacl::link {')
        calls = [
            ('channels_[dst_rank]->SendAsync(key, value);', 'send', 'dst_rank', 'value.size()'),
            ('channels_[dst_rank]->SendAsync(key, std::move(value));', 'send', 'dst_rank', 'value_length'),
            ('channels_[dst_rank]->SendAsyncThrottled(key, value);', 'send', 'dst_rank', 'value.size()'),
            ('channels_[dst_rank]->SendAsyncThrottled(key, std::move(value));', 'send', 'dst_rank', 'value_length'),
            ('channels_[dst_rank]->Send(key, value);', 'send', 'dst_rank', 'value.size()'),
            ('auto value = channels_[src_rank]->Recv(key);', 'recv', 'src_rank', 'value.size()'),
        ]
        for call, kind, peer, size in calls:
            new = f'''const auto stage2_start = SpuStage2RecordLinkEvent ? Stage2LinkNow() : 0;
  {call}
  if (SpuStage2RecordLinkEvent) {{
    SpuStage2RecordLinkEvent("{kind}", desc_.id.c_str(), key.c_str(),
        {peer}, {size}, stage2_start, Stage2LinkNow());
  }}'''
            s = replace(s, call, new)
        return s
    return patch(path, transform)


def prepare():
    src = ROOT / "build/src"
    diffs = {}
    header = src / "libspu/mpc/bumblebee/rlwe/bumblebee_metrics.h"
    def header_transform(s):
        s = replace(s, '#include <cstdint>', '#include <cstdint>\n#include <string>\n#include <vector>')
        extra = '''// Stage-two intervals use the same clock as SPU ActionRecord.
struct Stage2Event {
  std::string kind, name;
  int64_t start, end;
  size_t thread, peer, bytes;
};
bool Stage2Enabled();
int64_t Stage2Now();
void RecordStage2Event(const char* kind, const std::string& name,
    int64_t start, int64_t end, size_t peer = 0, size_t bytes = 0);
std::vector<Stage2Event> GetStage2Events();
class ScopedStage2Span {
  const char *kind_, *name_;
  int64_t start_;
 public:
  ScopedStage2Span(const char* kind, const char* name)
      : kind_(kind), name_(name), start_(Stage2Enabled() ? Stage2Now() : 0) {}
  ~ScopedStage2Span() {
    if (start_) RecordStage2Event(kind_, name_, start_, Stage2Now());
  }
};

'''
        s = replace(s, 'class ScopedBumblebeeMetric final {', extra + 'class ScopedBumblebeeMetric final {')
        s = replace(s, 'std::size_t bytes = 0)\n      : kind_(kind), bytes_(bytes), start_(Clock::now()) {}',
            '''std::size_t bytes = 0, const char* label = "backend_conversion_region")
      : kind_(kind), bytes_(bytes), start_(Clock::now()), label_(label),
        event_start_(kind == BumblebeeMetricKind::kConversion && Stage2Enabled()
                     ? Stage2Now() : 0) {}''')
        s = replace(s, 'Clock::time_point start_;', 'Clock::time_point start_;\n  const char* label_;\n  int64_t event_start_;')
        return s
    diffs[str(header.relative_to(src))] = patch(header, header_transform)
    metrics = src / "libspu/mpc/bumblebee/rlwe/bumblebee_metrics.cc"
    def metrics_transform(s):
        s = replace(s, '#include <chrono>', '#include <chrono>\n#include <cstdlib>\n#include <mutex>\n#include <thread>')
        s = replace(s, 'std::array<AtomicMetric, kMetricCount> metrics;',
            'std::array<AtomicMetric, kMetricCount> metrics;\n  std::mutex event_mutex;\n  std::vector<Stage2Event> events;')
        s = replace(s, 'void ResetBumblebeeMetrics() {\n  auto& state = State();',
            'void ResetBumblebeeMetrics() {\n  auto& state = State();\n  { std::lock_guard<std::mutex> lock(state.event_mutex); state.events.clear(); }')
        extra = '''bool Stage2Enabled() {
  static const bool enabled = std::getenv("SPU_BB_STAGE2_TRACE") != nullptr;
  return enabled;
}
int64_t Stage2Now() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}
void RecordStage2Event(const char* kind, const std::string& name,
    int64_t start, int64_t end, size_t peer, size_t bytes) {
  if (!Stage2Enabled()) return;
  auto& state = State();
  std::lock_guard<std::mutex> lock(state.event_mutex);
  state.events.push_back({kind, name, start, end,
      std::hash<std::thread::id>{}(std::this_thread::get_id()), peer, bytes});
}
std::vector<Stage2Event> GetStage2Events() {
  auto& state = State();
  std::lock_guard<std::mutex> lock(state.event_mutex);
  return state.events;
}

'''
        s = replace(s, 'ScopedBumblebeeMetric::~ScopedBumblebeeMetric() {',
            extra + '''ScopedBumblebeeMetric::~ScopedBumblebeeMetric() {
  if (event_start_) RecordStage2Event("conversion_region", label_, event_start_, Stage2Now());''')
        s += '''
extern "C" void SpuStage2RecordLinkEvent(const char* kind, const char* context,
    const char* key, size_t peer, size_t bytes, int64_t start, int64_t end) {
  using namespace spu::mpc::bumblebee;
  if (Stage2Enabled()) RecordStage2Event(kind, std::string(context) + ":" + key,
                                        start, end, peer, bytes);
}
'''
        return s
    diffs[str(metrics.relative_to(src))] = patch(metrics, metrics_transform)
    dot = src / "libspu/mpc/bumblebee/arith/bumblebee_dot.cc"
    def dot_transform(s):
        s = replace(s, 'ScopedBumblebeeMetric conversion(BumblebeeMetricKind::kConversion);',
                    'ScopedBumblebeeMetric conversion(BumblebeeMetricKind::kConversion, 0, "dot_H2A");')
        s = replace(s, 'yacl::ElapsedTimer _timer;\n    const auto &this_galois_context',
                    'ScopedStage2Span stage2_packing("packing", "RLWE_pack");\n    yacl::ElapsedTimer _timer;\n    const auto &this_galois_context') if s.count('yacl::ElapsedTimer _timer;\n    const auto &this_galois_context') == 1 else s.replace('yacl::ElapsedTimer _timer;', 'ScopedStage2Span stage2_packing("packing", "cipher_pack");\n    yacl::ElapsedTimer _timer;')
        s = replace(s, '  SPU_ENFORCE(batch_size > 0);\n  int next_rank = conn->NextRank();',
                    '  ScopedStage2Span stage2_conversion("conversion", "dot_decrypt_moddown");\n  SPU_ENFORCE(batch_size > 0);\n  int next_rank = conn->NextRank();')
        # The sender input stage covers encoding/encryption and message submission.
        anchor = '  size_t num_ct;\n'
        # Existing functions remain intact; output H2A and decrypt are sufficient
        # to identify actual protocol boundaries without guessing input phases.
        return s
    diffs[str(dot.relative_to(src))] = patch(dot, dot_transform)
    mul = src / "libspu/mpc/bumblebee/arith/bumblebee_mul_impl.cc"
    def mul_transform(s):
        s = s.replace('ScopedBumblebeeMetric conversion(BumblebeeMetricKind::kConversion);',
                      'ScopedBumblebeeMetric conversion(BumblebeeMetricKind::kConversion, 0, __func__);')
        return s
    diffs[str(mul.relative_to(src))] = patch(mul, mul_transform)
    api = src / "libspu/device/api.cc"
    def api_transform(s):
        s = replace(s, '#include <chrono>', '#include <chrono>\n#include <atomic>\n#include <cstdlib>')
        extra = '''void dumpStage2(spu::SPUContext* sctx, int64_t start, int64_t end) {
  using namespace mpc::bumblebee;
  const char* prefix = std::getenv("SPU_BB_STAGE2_TRACE");
  if (!prefix) return;
  static std::atomic<size_t> counter{0};
  auto path = std::string(prefix) + "-p" + std::to_string(sctx->lctx()->Rank()) +
              "-e" + std::to_string(++counter) + ".tsv";
  std::ofstream out(path);
  out << "kind\\tname\\tstart_ns\\tend_ns\\tthread\\tpeer\\tbytes\\trecv_bytes\\n";
  out << "execution\\ttotal\\t" << start << "\\t" << end << "\\t0\\t0\\t0\\t0\\n";
  for (const auto& rec : GET_TRACER(sctx)->getProfState()->getRecords()) {
    const char* kind = (rec.flag & TR_HLO) ? "HLO" : (rec.flag & TR_HAL) ? "HAL" : "MPC";
    auto ns = [](auto t) { return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count(); };
    out << kind << "\\t" << rec.name << "\\t" << ns(rec.start) << "\\t" << ns(rec.end)
        << "\\t0\\t0\\t" << (rec.send_bytes_end - rec.send_bytes_start)
        << "\\t" << (rec.recv_bytes_end - rec.recv_bytes_start) << "\\n";
  }
  for (const auto& event : GetStage2Events()) {
    out << event.kind << "\\t" << event.name << "\\t" << event.start << "\\t" << event.end
        << "\\t" << event.thread << "\\t" << event.peer << "\\t" << event.bytes << "\\t0\\n";
  }
  out.close();
  SPDLOG_INFO("[Stage2Profile] file={} start_ns={} end_ns={}", path, start, end);
}

'''
        s = replace(s, 'void SPUErrorHandler(void *use_data,', extra + 'void SPUErrorHandler(void *use_data,')
        s = replace(s, '  // execution\n  std::vector<spu::Value> outputs;',
                    '  // execution\n  int64_t stage2_start = 0, stage2_end = 0;\n  std::vector<spu::Value> outputs;')
        s = replace(s, '    TimeitGuard timeit(exec_stats.execution_time);',
                    '    TimeitGuard timeit(exec_stats.execution_time);\n    if (is_bumblebee) stage2_start = mpc::bumblebee::Stage2Now();')
        s = replace(s, '  // sync output to environment.',
                    '  if (is_bumblebee) stage2_end = mpc::bumblebee::Stage2Now();\n\n  // sync output to environment.')
        s = replace(s, '    printBumblebeeBackendMetrics(sctx, exec_stats, comm_stats);',
                    '    printBumblebeeBackendMetrics(sctx, exec_stats, comm_stats);\n    dumpStage2(sctx, stage2_start, stage2_end);')
        return s
    diffs[str(api.relative_to(src))] = patch(api, api_transform)
    return diffs


def build():
    diffs = prepare()
    cache = ROOT / "build/cache"
    external = Path("91451686bbf5ebe79baf6c2b1919d0ed/external/yacl/yacl/link/context.cc")
    config = json.loads((ROOT / "config.json").read_text())
    subprocess.run(["docker", "run", "--rm", "-v", f"{ROOT}:/task:ro", "-v", f"{cache}:/cache",
        config["image_base"], "python", "/task/profile_build.py", "--patch-yacl", str(Path('/cache') / external)], check=True)
    diffs[str(external)] = ''.join(difflib.unified_diff(
        (cache / external).with_name('context.cc.stage2-original').read_text().splitlines(True),
        (cache / external).read_text().splitlines(True), fromfile='yacl/context.cc', tofile='yacl/context.cc'))
    write(ROOT / "build/stage2-patches.json", diffs)
    base = json.loads((ROOT / "build/cpu.json").read_text())
    args = base["command"]
    args[args.index('--name')+1] = 'ours-bb-stage2-build'
    manifest = dict(status="building", backend="cpu", command=args,
        original_library_sha256=digest(ROOT / "build/cpu.so"),
        sources={str(p.relative_to(ROOT / 'build/src')):digest(p)
                 for p in (ROOT / 'build/src').rglob('*') if p.is_file() and not p.name.endswith('.stage2-original')},
        yacl_context_sha256=digest(cache / external), patch_sha256=digest(ROOT / "build/stage2-patches.json"))
    write(ROOT / "build/stage2.json", manifest)
    print("Building stage2.so with optional interval recording", flush=True)
    with (ROOT / "build/stage2.log").open('w') as log:
        result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        manifest.update(status="failed", returncode=result.returncode)
        write(ROOT / "build/stage2.json", manifest)
        raise RuntimeError("See build/stage2.log")
    product = cache / "91451686bbf5ebe79baf6c2b1919d0ed/execroot/spulib/bazel-out/k8-opt/bin/spu/libspu.so"
    shutil.copy2(product, ROOT / "build/stage2.so")
    manifest.update(status="built", library_sha256=digest(ROOT / "build/stage2.so"))
    write(ROOT / "build/stage2.json", manifest)
    print("Stage-two tracing library built; cpu.so preserved", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--patch-yacl', type=Path)
    args = parser.parse_args()
    if args.patch_yacl:
        patch_yacl(args.patch_yacl)
    else:
        build()
