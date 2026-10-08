"""构造独立的 rotation/online send-only 审计副本；不修改协议源码。"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from prepare_b_count_audit import ROOT, masked, closing

HEADER = r'''#pragma once
#include <array>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
#include <stdexcept>
namespace blb_wire {
using Bytes=std::array<uint64_t,2>; // main HE transport, SCI channels
struct Scope;
struct State {
 bool active=false; int block=0; std::thread::id owner;
 std::function<Bytes()> read; std::vector<Scope*> stack;
 std::mutex lock; std::string stage="online";
 std::map<std::tuple<std::string,std::string,int>,uint64_t> rotations;
 std::map<std::pair<std::string,std::string>,uint64_t> wire;
};
inline State &state(){ static State s; return s; }
template<class T> void bind(T &b) {
 state().read=[&b](){ Bytes r{0,0}; std::set<const void*> seen;
  auto add=[&](auto *io,int channel){if(io && seen.insert(io).second) r[channel]+=io->counter;};
  add(b.io,0);
  for(auto *p:b.nonlinear.iopackArr) if(p){add(p->io,1);add(p->io_rev,1);add(p->io_GC,1);}
  return r;
 };
}
struct Scope {
 bool on=false; Bytes before{0,0},children{0,0}; std::string prev,path,name;
 Scope(const char *n):name(n){auto &s=state();
  if(!s.active || std::this_thread::get_id()!=s.owner) return;
  on=true; before=s.read();
  {std::lock_guard<std::mutex> guard(s.lock);prev=s.stage;path=prev+"/"+name;s.stage=path;}
  s.stack.push_back(this);
 }
 ~Scope(){if(!on)return;auto &s=state();auto end=s.read();Bytes delta;
  for(int i=0;i<2;++i){if(end[i]<before[i] || end[i]-before[i]<children[i]) std::terminate();delta[i]=end[i]-before[i];}
  auto main=delta[0]-children[0],mpc=delta[1]-children[1];
  if(name=="ckks_to_mpc" || name=="mpc_to_ckks") s.wire[{path,name}]+=main+mpc;
  else if(name=="ciphertext_transfer") s.wire[{path,"ciphertext_transfer"}]+=main+mpc;
  else {s.wire[{path,"nonlinear_MPC"}]+=mpc;s.wire[{path,"other"}]+=main;}
  s.stack.pop_back();if(!s.stack.empty())for(int i=0;i<2;++i)s.stack.back()->children[i]+=delta[i];
  {std::lock_guard<std::mutex> guard(s.lock);s.stage=prev;}
 }
};
inline std::unique_ptr<Scope> &root(){static std::unique_ptr<Scope> r;return r;}
inline Bytes &initial(){static Bytes b;return b;}
inline void start(int block){auto &s=state();if(s.active)throw std::runtime_error("nested audit start");
 s.owner=std::this_thread::get_id();s.block=block;s.rotations.clear();s.wire.clear();s.stage="block";
 initial()=s.read();s.active=true;root().reset(new Scope("online"));
 std::cout<<"WIRE_BEGIN block="<<block<<" main="<<initial()[0]<<" sci="<<initial()[1]<<std::endl;
}
inline void stop(){auto &s=state();root().reset();auto end=s.read();s.active=false;
 uint64_t sum=0;for(auto &x:s.wire){sum+=x.second; if(x.second)std::cout<<"WIRE_COUNT block="<<s.block<<" stage="<<x.first.first<<" category="<<x.first.second<<" bytes="<<x.second<<std::endl;}
 uint64_t total=end[0]-initial()[0]+end[1]-initial()[1];
 std::cout<<"WIRE_TOTAL block="<<s.block<<" main_sent="<<end[0]-initial()[0]<<" sci_sent="<<end[1]-initial()[1]<<" sent="<<total<<" categorized="<<sum<<" conservation="<<(sum==total)<<std::endl;
 if(sum!=total)throw std::runtime_error("wire conservation failure");
 for(auto &x:s.rotations)std::cout<<"ROT_TRACE block="<<s.block<<" stage="<<std::get<0>(x.first)<<" site="<<std::get<1>(x.first)<<" step="<<std::get<2>(x.first)<<" count="<<x.second<<std::endl;
}
template<class F> void rotation(const char *site,int step,F run){auto &s=state();
 if(s.active){std::lock_guard<std::mutex> guard(s.lock);++s.rotations[{s.stage,site,step}];}run(step);
}
template<class F> void send(const char *site,size_t cts,F run){auto &s=state();
 if(!s.active){run();return;}auto before=s.read();std::string stage=s.stage;
 {Scope scope("ciphertext_transfer");run();}
 auto after=s.read();std::cout<<"WIRE_SEND block="<<s.block<<" stage="<<stage<<" site="<<site<<" cts="<<cts<<" bytes="<<after[0]-before[0]<<std::endl;
}
}
'''


def arguments(code, start, end):
    depth = 0
    cuts = [start+1]
    for i in range(start+1, end):
        if code[i] in '([{': depth += 1
        elif code[i] in ')]}': depth -= 1
        elif code[i] == ',' and depth == 0: cuts.append(i+1)
    return [(a, b-1 if j+1 < len(cuts) else end)
            for j, (a,b) in enumerate(zip(cuts, cuts[1:]+[end]))]


def instrument(source, name):
    code = masked(source)
    changes = []
    for m in re.finditer(r'\b\w+(?:(?:->|\.)\w+)*->(rotate_vector(?:_inplace)?)\s*\(', code):
        end = closing(code, m.end()-1)
        args = arguments(code, m.end()-1, end)
        a,b = args[1]
        step = source[a:b]
        call = source[m.start():a]+' audit_step '+source[b:end+1]
        site = f'{name}:{source.count(chr(10),0,m.start())+1}:{m[1]}'
        new = f'blb_wire::rotation("{site}", {step}, [&](int audit_step){{ {call}; }})'
        changes.append((m.start(), end+1, new))
    for m in re.finditer(r'\bsend_encrypted_vector\s*\(', code):
        end = closing(code,m.end()-1)
        args = arguments(code,m.end()-1,end)
        if len(args) != 2: raise ValueError('unexpected send signature')
        vec = source[slice(*args[1])].strip()
        site=f'{name}:{source.count(chr(10),0,m.start())+1}'
        call=source[m.start():end+1]
        changes.append((m.start(),end+1,f'blb_wire::send("{site}", ({vec}).size(), [&](){{ {call}; }})'))
    for m in re.finditer(r'\b\w+->encrypt_send\s*\(', code):
        end = closing(code,m.end()-1)
        args = arguments(code,m.end()-1,end)
        if len(args)!=2: raise ValueError('unexpected encrypt_send signature')
        vec=source[slice(*args[0])].strip()
        site=f'{name}:{source.count(chr(10),0,m.start())+1}:encrypt_send'
        call=source[m.start():end+1]
        changes.append((m.start(),end+1,f'blb_wire::send("{site}", ({vec}).size(), [&](){{ {call}; }})'))
    for m in re.finditer(r'\b(?:Bert_ckks|Linear_ckks)::(\w+)\s*\(', code):
        end=closing(code,m.end()-1)+1
        while end<len(code) and code[end].isspace():end+=1
        if code[end:end+1]=='{':changes.append((end+1,end+1,f'\nblb_wire::Scope wire_scope("{m[1]}");\n'))
    for a,b,text in sorted(changes,reverse=True):source=source[:a]+text+source[b:]
    return '#include "wire_audit.h"\n'+source


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--vit',action='store_true')
    args=parser.parse_args()
    out=args.output.resolve()
    cmd=[sys.executable,str(ROOT/'scripts/prepare_b_count_audit.py'),'--output',str(out)]
    if args.vit:cmd+=['--vit']
    subprocess.run(cmd,check=True)
    donor=ROOT/('ours_vit_b16/runtime' if args.vit else 'gpu/SCI/tests/bert_large_bolt')
    manifest={}
    for p in donor.iterdir():
        if p.suffix not in {'.h','.cpp'}:continue
        text=instrument(p.read_text(),p.name)
        if p.name=='ckks_bert_large_main.cpp':
            text=text.replace('cout << "begin test" << endl;', 'blb_wire::bind(bert_ckks); blb_wire::start(0);\n  cout << "begin test" << endl;')
            text=text.replace('auto end = high_resolution_clock::now();\n  cout << "--------linear part 4 done--------"', 'blb_wire::stop();\n  auto end = high_resolution_clock::now();\n  cout << "--------linear part 4 done--------"')
        if p.name=='ckks_vit_b16_main.cpp':
            text=text.replace('const uint64_t comm0 = bert.current_comm();', 'blb_wire::bind(bert); blb_wire::start(layer);\n      const uint64_t comm0 = bert.current_comm();')
            text=text.replace('const uint64_t comm1 = bert.current_comm();', 'blb_wire::stop();\n      const uint64_t comm1 = bert.current_comm();')
        (out/'source'/p.name).write_text(text)
        manifest[p.name]=dict(original_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                              instrumented_sha256=hashlib.sha256(text.encode()).hexdigest())
    (out/'source/wire_audit.h').write_text(HEADER)
    (out/'wire_source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Catch stale entry-point anchors before starting an expensive build/run.
    mainfile=out/'source'/('ckks_vit_b16_main.cpp' if args.vit else 'ckks_bert_large_main.cpp')
    text=mainfile.read_text()
    assert text.count('blb_wire::start(')==1 and text.count('blb_wire::stop();')==1


if __name__=='__main__':main()
