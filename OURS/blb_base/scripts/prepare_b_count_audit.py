"""在输出目录构造 BERT/ViT 的只计数副本，不修改正式 B 或供体。"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex

ROOT = Path(__file__).resolve().parents[1]
OPS = {'rotate_vector', 'rotate_vector_inplace', 'multiply', 'multiply_inplace',
       'multiply_plain', 'multiply_plain_inplace', 'square', 'square_inplace',
       'rescale_to_next', 'rescale_to_next_inplace', 'relinearize_inplace',
       'complex_conjugate', 'complex_conjugate_inplace'}


def masked(source):
    # Preserve offsets while excluding comments and literals from C++ matching.
    return re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
                  lambda m: ''.join('\n' if c == '\n' else ' ' for c in m[0]), source)


def closing(code, start):
    depth = 1
    for i in range(start + 1, len(code)):
        depth += (code[i] == '(') - (code[i] == ')')
        if depth == 0:
            return i
    raise ValueError('unbalanced C++ call')


def instrument(source):
    code = masked(source)
    insertions = {}
    def insert(i, text):
        insertions[i] = insertions.get(i, '') + text
    calls = 0
    for m in re.finditer(r'\b\w+(?:(?:->|\.)\w+)*->(' + '|'.join(sorted(OPS)) + r')\s*\(', code):
        end = closing(code, m.end() - 1)
        insert(m.start(), '(blb_count::hit("' + m[1] + '"), ')
        insert(end + 1, ')')
        calls += 1
    for m in re.finditer(r'\b(?:Bert_ckks|Linear_ckks)::(\w+)\s*\(', code):
        end = closing(code, m.end() - 1)
        brace = end + 1
        while brace < len(code) and code[brace].isspace():
            brace += 1
        if brace < len(code) and code[brace] == '{':
            insert(brace + 1, '\nblb_count::Scope audit_scope("' + m[1] + '");\n')
    for pos, text in sorted(insertions.items(), reverse=True):
        source = source[:pos] + text + source[pos:]
    return '#include "count_audit.h"\n' + source, calls


HEADER = r'''#pragma once
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
namespace blb_count {
struct State { std::mutex mutex; bool enabled=false; std::string stage="block";
 std::map<std::pair<std::string,std::string>,unsigned long long> counts; };
inline State &state() { static State s; return s; }
inline void hit(const char *op) { auto &s=state(); std::lock_guard<std::mutex> lock(s.mutex);
 if(s.enabled) ++s.counts[{s.stage,op}]; }
struct Scope { std::string before; Scope(const char *name) { auto &s=state();
 std::lock_guard<std::mutex> lock(s.mutex); before=s.stage; s.stage += std::string("/")+name; }
 ~Scope() { auto &s=state(); std::lock_guard<std::mutex> lock(s.mutex); s.stage=before; } };
inline void dump() { auto &s=state(); std::lock_guard<std::mutex> lock(s.mutex);
 for(auto &entry:s.counts) std::cout << "BLB_COUNT stage=" << entry.first.first
 << " op=" << entry.first.second << " count=" << entry.second << std::endl; }
inline void start() { auto &s=state(); s.enabled=true; s.counts.clear(); std::atexit(dump); }
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vit', action='store_true', help='count frozen ViT and time exp safety passes')
    parser.add_argument('--source', type=Path, help='optional preserved runtime source directory')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    src = out / 'source'; src.mkdir()
    (src / 'count_audit.h').write_text(HEADER)
    manifest = {}
    donor = args.source or ROOT / ('ours_vit_b16/runtime' if args.vit else 'gpu/SCI/tests/bert_large_bolt')
    for p in donor.iterdir():
        if p.suffix not in {'.cpp', '.h'}:
            continue
        original = p.read_text()
        text, count = instrument(original)
        if p.name == 'ckks_bert_large_main.cpp':
            text = text.replace('cout << "begin test" << endl;',
                                'blb_count::start();\n  cout << "begin test" << endl;')
        if args.vit and p.name == 'ckks_vit_b16_main.cpp':
            text = text.replace('const auto online_start =', 'blb_count::start();\n    const auto online_start =')
        if args.vit and p.name == 'ckks_bert_basic.cpp' and 'FPMath *exp_math =' in original:
            text = text.replace('BoolArray underflow;', 'auto audit_exp_pre = high_resolution_clock::now();\n    BoolArray underflow;')
            text = text.replace('// BLB Algorithm 5: exactly six squares',
                'cout << "EXP_SAFETY_PHASE phase=input_clip ms=" << chrono::duration<double,milli>(high_resolution_clock::now()-audit_exp_pre).count() << endl;\n    // BLB Algorithm 5: exactly six squares')
            text = text.replace('ret = exp_math->fix->if_else(underflow, uint64_t(0), ret);',
                'auto audit_exp_tail = high_resolution_clock::now();\n        ret = exp_math->fix->if_else(underflow, uint64_t(0), ret);')
            text = text.replace('ret = exp_math->fix->if_else(negative_output, uint64_t(0), ret);',
                'ret = exp_math->fix->if_else(negative_output, uint64_t(0), ret);\n        cout << "EXP_SAFETY_PHASE phase=output_mux_and_clamp ms=" << chrono::duration<double,milli>(high_resolution_clock::now()-audit_exp_tail).count() << endl;')
        (src / p.name).write_text(text)
        manifest[p.name] = dict(original_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                               counted_call_sites=count)
    (out/'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    build = ROOT/'build/container-gpu/tests/bert_large_bolt/CMakeFiles'
    flags = (build/('LIB_VIT_OURS_CKKS.dir/flags.make' if args.vit else 'LIB_BERT_CKKS.dir/flags.make')).read_text()
    def flag(key): return re.search(r'^'+key+r' = (.*)$', flags, re.M)[1]
    container_out = Path('/workspace/OURS')/out.relative_to(ROOT)
    files = ['ckks_bert_large_main.cpp', 'he.cpp', 'nonlinear.cpp',
             'linear_ckks.cpp', 'ckks_bert.cpp', 'bert_utils.cpp']
    if args.vit:
        files = ['ckks_vit_b16_main.cpp', 'he.cpp', 'linear_ckks.cpp', 'ckks_bert.cpp']
        (out/'vit_model.h').write_bytes((ROOT/'ours_vit_b16/vit_model.h').read_bytes())
    commands = ['#!/usr/bin/env bash', 'set -euo pipefail', 'cd '+shlex.quote(str(container_out)),
                'mkdir -p build/bin build/objects', 'pids=()']
    for name in files:
        commands += ['/usr/bin/c++ '+flag('CXX_DEFINES')+' '+flag('CXX_INCLUDES')+' '+flag('CXX_FLAGS')+
                     ' -c source/'+name+' -o build/objects/'+name+'.o > build/'+name+'.log 2>&1 &',
                     'pids+=("$!")']
    commands += ['for pid in "${pids[@]}"; do wait "$pid"; done']
    target = 'ours_vit_b16_main' if args.vit else 'ckks_bert_large_main'
    main_object = ('CMakeFiles/ours_vit_b16_main.dir/workspace/OURS/ours_vit_b16/runtime/ckks_vit_b16_main.cpp.o'
                   if args.vit else 'CMakeFiles/ckks_bert_large_main.dir/ckks_bert_large_main.cpp.o')
    link = (build/(target+'.dir/link.txt')).read_text().strip()
    link = link.replace(main_object,
                        ' '.join(str(container_out/'build/objects'/ (n+'.o')) for n in files))
    link = link.replace('../../bin/'+target, str(container_out/'build/bin'/target))
    link = link.replace('../../lib/libLIB_BERT_CKKS.a', '')
    commands += ['cd /workspace/OURS/build/container-gpu/tests/bert_large_bolt', link]
    (out/'build.sh').write_text('\n'.join(commands)+'\n')
    print(out)


if __name__ == '__main__':
    main()
