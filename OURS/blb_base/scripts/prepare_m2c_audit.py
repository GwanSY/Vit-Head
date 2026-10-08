"""生成隔离的 M2C 等价流程探针；保留供体其余算子，不宣称其为完整 BERT 验证。"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from prepare_b_count_audit import ROOT, masked, closing
from prepare_b_wire_audit import instrument, arguments


def function(text, signature):
    start=text.index(signature)
    code=masked(text)
    brace=code.index('{', start)
    depth=1; end=brace+1
    while depth:
        depth+=(code[end]=='{')-(code[end]=='}'); end+=1
    return start,brace,end


DETAIL=r'''#pragma once
#include "wire_audit.h"
namespace m2c_detail {
struct State { bool active=false; unsigned id=0; std::string phase;
 std::mutex mutex; std::map<std::string,std::array<uint64_t,3>> detail; };
inline State &state(){static State s;return s;}
inline void record(const std::string &phase,uint64_t calls,uint64_t elements,uint64_t bytes){
 auto &s=state(); if(!s.active)return;std::lock_guard<std::mutex> lock(s.mutex);
 auto &v=s.detail[phase];v[0]+=calls;v[1]+=elements;v[2]+=bytes;
}
struct Conversion {
 bool on; blb_wire::Bytes before; unsigned id; size_t cts; int slots,bits,scale;
 Conversion(size_t n,int sl,int b,int sc):on(blb_wire::state().active),cts(n),slots(sl),bits(b),scale(sc){
  if(!on)return;auto &s=state();s.active=true;s.detail.clear();id=++s.id;before=blb_wire::state().read();
 }
 ~Conversion(){if(!on)return;auto &s=state();auto after=blb_wire::state().read();s.active=false;
  std::cout<<"M2C_TOTAL block="<<blb_wire::state().block<<" id="<<id<<" cts="<<cts<<" slots="<<slots
   <<" bits="<<bits<<" scale="<<scale<<" sent="<<after[0]+after[1]-before[0]-before[1]<<std::endl;
  for(auto &v:s.detail)std::cout<<"M2C_DETAIL block="<<blb_wire::state().block<<" id="<<id<<" phase="<<v.first
   <<" calls="<<v.second[0]<<" elements="<<v.second[1]<<" sent="<<v.second[2]<<std::endl;
 }
};
struct Phase {
 std::string name,prev;uint64_t elements;blb_wire::Bytes before;bool on;
 Phase(const char*n,uint64_t count):name(n),elements(count),on(state().active){
  if(on){prev=state().phase;state().phase=name;before=blb_wire::state().read();}}
 ~Phase(){if(on){auto after=blb_wire::state().read();record(name,1,elements,after[0]+after[1]-before[0]-before[1]);state().phase=prev;}}
};
template<class IO,class F> void primitive(IO *io,const char*name,uint64_t elements,F f){
 if(!state().active){f();return;}auto before=io->get_comm();f();
 record(state().phase+"/"+name,1,elements,io->get_comm()-before);
}
}
'''

FIXTURE=r'''
// Public fixture only. Opening its result never opens a model activation.
static void m2c_public_fixture(Bert_ckks &b) {
 auto *he=b.linear.he1;
 for(auto cfg:std::vector<std::pair<int,int>>{{37,12},{40,20}}){
  const int bits=cfg.first,scale=cfg.second,slots=he->num_slots;
  const uint64_t mask=(uint64_t{1}<<bits)-1;
  vector<vector<uint64_t>> shares(1,vector<uint64_t>(slots));
  vector<double> reference(slots);
  for(int i=0;i<slots;++i){
   int64_t raw=(int64_t(i%65)-32)*(int64_t{1}<<scale)/8;
   uint64_t a=(uint64_t(i+1)*0x9e3779b97f4a7c15ULL)&mask;
   shares[0][i]=b.party==sci::ALICE?a:(uint64_t(raw)-a)&mask;
   reference[i]=double(raw)/double(uint64_t{1}<<scale);
  }
  auto pts=equivalent_m2c(b.nonlinear,b.party,shares,slots,he->get_Q(),bits,scale,he->he_scale,he,false,false);
  vector<Ciphertext> wire(1);int passed=1;double max_error=0;
  if(b.party==sci::BOB){he->encryptor->encrypt(pts[0],wire[0]);send_encrypted_vector(b.io,wire);
   recv_encrypted_vector(he->context,b.io,wire);
   Plaintext opened;he->decryptor->decrypt(wire[0],opened);vector<complex<double>> values;he->encoder->decode(opened,values);
   for(int i=0;i<slots;++i){
    if(!std::isfinite(values[i].real()) || !std::isfinite(values[i].imag())) max_error=std::numeric_limits<double>::infinity();
    else max_error=max(max_error,max(abs(values[i].real()-reference[i]),abs(values[i].imag())));
   }
   passed=max_error<=8.0/double(uint64_t{1}<<scale);
   b.io->send_data(&passed,sizeof(passed));
  } else {recv_encrypted_vector(he->context,b.io,wire);he->evaluator->add_plain_inplace(wire[0],pts[0]);send_encrypted_vector(b.io,wire);
   b.io->recv_data(&passed,sizeof(passed));}
  cout<<"M2C_PUBLIC_FIXTURE bits="<<bits<<" scale="<<scale<<" slots="<<slots<<" max_error="<<max_error<<" pass="<<passed<<endl;
  if(!passed)throw runtime_error("M2C signed public fixture failed");
 }
}
'''


def phase(text, begin, end, name, elements):
    a=text.index(begin); b=text.index(end,a)
    return text[:a]+'{ m2c_detail::Phase phase("'+name+'", '+elements+');\n'+text[a:b]+'}\n'+text[b:]


def common_body(runtime):
    text=(runtime/'ckks_bert_basic.cpp').read_text()
    a,_,b=function(text,'vector<Plaintext> Bert_ckks::mpc_to_ckks(')
    text=text[a:b]
    text=text[text.index('    int poly_degree'):text.index('    mpc_to_ckks_conversions +=')]
    text=text.replace('blb_conversion_threads()', '64').replace('    ++ring_to_field_calls;','')
    # Variable declarations stay outside measurement scopes.
    text=phase(text,'    if (known_nonnegative)', '\n    // The input shares arrive', 'slot_ring_extension','slot_elements')
    text=phase(text,'    for (size_t index = 0; index < slot_low.size();', '\n    // Real CKKS slots', 'slot_origin_correction','slot_elements')
    text=phase(text,'#pragma omp parallel for num_threads(thread_num)', '    // Algorithm 2 step 2', 'fft_encode_coefficients','x.size()*poly_degree')
    text=phase(text,'    nonlinear.ring_extend_wide(64, coefficient_ring.data()', '\n    const int scale_shift', 'coefficient_ring_extension','coefficient_count')
    pos=text.index('    vector<Plaintext> y(x.size());')
    prefix=text[:pos];tail=text[pos:]
    tail=phase(tail,'#pragma omp parallel for num_threads(thread_num)', '    cout << "BLB_RING_TO_FIELD_WIDE', 'local_field_reduction_rns_ntt','x.size()*poly_degree*rns_moduli.size()')
    return ('#pragma once\n#include "m2c_detail.h"\n'
        'template<class Nonlinear> vector<Plaintext> equivalent_m2c(Nonlinear &nonlinear,int party,\n'
        'const vector<vector<HE_CKKS::scalar_t>> &x,int slot,uint64_t Q,int bitwidth,int s_in,int s_out,HE_CKKS *he,bool half,bool known_nonnegative){\n'
        'm2c_detail::Conversion audit(x.size(),slot,bitwidth,s_in);\n'+prefix+tail+'return y;\n}\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--vit',action='store_true')
    args=p.parse_args();out=args.output.resolve();runtime=ROOT/'ours_vit_b16/runtime'
    cmd=[sys.executable,str(ROOT/'scripts/prepare_b_wire_audit.py'),'--output',str(out)]
    if args.vit:cmd.append('--vit')
    subprocess.run(cmd,check=True)
    src=out/'source'
    # Both executables include the SAME generated converter, copied from ViT.
    (src/'m2c_detail.h').write_text(DETAIL)
    shared=common_body(runtime);(src/'equivalent_m2c.h').write_text(shared)
    donor=runtime if args.vit else ROOT/'gpu/SCI/tests/bert_large_bolt'
    basic=(donor/'ckks_bert_basic.cpp').read_text()
    a,brace,b=function(basic,'vector<Plaintext> Bert_ckks::mpc_to_ckks(')
    signature=basic[a:brace]
    body='''{
    cout << "in mpc_to_ckks, slot, Q, bitwidth, s_in, s_out:" << slot << "," << Q << "," << bitwidth << "," << s_in << "," << s_out << endl;
    auto result=equivalent_m2c(nonlinear,party,x,slot,Q,bitwidth,s_in,s_out,he,half,KNOWN);
    cout << "mpc_to_ckks done" << endl;
    return result;
}'''.replace('KNOWN','known_nonnegative' if args.vit else 'false')
    # Keep ViT's profiling counters, unrelated to protocol messages.
    if args.vit:
        body=body.replace('    auto result=', '    conversion_ciphertexts += x.size();\n    mpc_to_ckks_conversions += x.size();\n    ++ring_to_field_calls;\n    auto result=')
    basic=basic[:a]+signature+body+basic[b:]
    (src/'ckks_bert_basic.cpp').write_text('#include "equivalent_m2c.h"\n'+instrument(basic,'ckks_bert_basic.cpp'))
    if not args.vit:
        # Numerical adapter dependencies; no donor QKV/QK/AV/Softmax/GELU edits.
        he_h=(donor/'he.h').read_text()
        he_h=he_h.replace('    vector<scalar_t> ckks_decode(', '''    using wide_scalar_t = __uint128_t;
    struct wide_complex_t { wide_scalar_t real=0, imag=0; };
    vector<wide_complex_t> ckks_encode_wide(vector<wide_complex_t> &&x,int degree,int working_ell) const;
    void ckks_ntt_rns(const vector<vector<scalar_t>> &rns_coefficients,double scale,Plaintext &destination);
    vector<scalar_t> ckks_decode(''',1)
        # Correct int32 shift overflow when fft_scale >= 32; same root values
        # as ViT's existing numerical fix, no precision parameter change.
        he_h=he_h.replace('(1 << fft_scale)', '(uint64_t{1} << fft_scale)')
        # pos is a loop-carried recurrence. The donor parallel loop races and
        # may build different slot maps on the two parties. ViT already fixes it.
        he_h=he_h.replace('        #pragma omp parallel for\n        for (size_t i = 0; i < num_slots; i++)',
            '        // Sequential CKKS slot-map recurrence, matching ViT.\n        for (size_t i = 0; i < num_slots; i++)',1)
        (src/'he.h').write_text(instrument(he_h,'he.h'))
        full=(runtime/'he.cpp').read_text()
        helper=full[:full.index('} // namespace')+len('} // namespace')]
        for sig in ['vector<HE_CKKS::wide_complex_t> HE_CKKS::ckks_encode_wide(\n    vector<wide_complex_t> &&', 'void HE_CKKS::ckks_ntt_rns(']:
            a,_,b=function(full,sig);method=full[a:b]
            if 'ckks_encode_wide' in sig:
                # M2C-only roots at the existing ViT precision. Donor C2M
                # retains its own root table/parameters and is not changed.
                brace=method.index('{')
                method=method[:brace+1]+'''
    const int fft_scale=40;
    const int logn=seal::util::get_power_of_two(degree);
    seal::util::ComplexRoots roots(static_cast<size_t>(degree)*2, MemoryManager::GetPool());
    vector<complex<uint64_t>> inv_root_powers_2n_scaled(degree);
    for(int i=1;i<degree;++i){
        auto root=std::conj(roots.get_root(seal::util::reverse_bits(static_cast<size_t>(i-1),logn)+1));
        inv_root_powers_2n_scaled[i]=complex<uint64_t>(
            static_cast<uint64_t>(root.real()*(uint64_t{1}<<fft_scale)),
            static_cast<uint64_t>(root.imag()*(uint64_t{1}<<fft_scale)));
    }
'''+method[brace+1:]
            helper+='\n'+method+'\n'
        (src/'he.cpp').write_text(instrument(helper+'\n'+(donor/'he.cpp').read_text(),'he.cpp'))
        (src/'wide_fft_math.h').write_bytes((runtime/'wide_fft_math.h').read_bytes())
        nl_h=(donor/'nonlinear.h').read_text()
        declarations='''    void ring_extend_wide(int nthreads,uint64_t* input,uint64_t* low,uint64_t* high,int size,int bitwidth,int target);
    void ring_extend_wide_nonnegative(int nthreads,uint64_t* input,uint64_t* low,uint64_t* high,int size,int bitwidth,int target);
'''
        nl_h=nl_h.replace('public:', 'public:\n'+declarations,1)
        (src/'nonlinear.h').write_text(nl_h)
        full=(runtime/'nonlinear.cpp').read_text();extra=''
        for sig in ['static void ring_extend_wide_thread(', 'static void ring_extend_wide_nonnegative_thread(', 'void NonLinear::ring_extend_wide(', 'void NonLinear::ring_extend_wide_nonnegative(']:
            a,_,b=function(full,sig);extra+='\n'+full[a:b]+'\n'
        (src/'nonlinear.cpp').write_text((donor/'nonlinear.cpp').read_text()+extra)
    # Measure extension children on their OWN worker channels. Summing global
    # counters in concurrent workers would double-count bytes.
    xt=(ROOT/'gpu/SCI/src/BuildingBlocks/value-extension.cpp').read_text()
    for method in ['s_extend_wide','z_extend_wide']:
        a,_,b=function(xt,'void XTProtocol::'+method+'(');part=xt[a:b]
        for name,kind in [('MSB_to_Wrap','wrap_known_msb'),('wrap_computation','wrap_compare'),('B2A','share_B2A')]:
            part=re.sub(r'aux->'+name+r'\(([^;]+)\);',lambda m:'m2c_detail::primitive(iopack,"'+kind+'",dim,[&](){'+m[0]+'});',part)
        xt=xt[:a]+part+xt[b:]
    (src/'value-extension.cpp').write_text('#include "m2c_detail.h"\n'+xt)
    mainfile=src/('ckks_vit_b16_main.cpp' if args.vit else 'ckks_bert_large_main.cpp')
    main=mainfile.read_text()
    pos=main.index('using namespace')
    main=main[:pos]+'#include "equivalent_m2c.h"\n'+main[pos:]
    # Place the fixture after using declarations but before the first function.
    pos=main.index('int bitlength')
    main=main[:pos]+FIXTURE+'\n'+main[pos:]
    if args.vit:
        main=main.replace('blb_wire::bind(bert); blb_wire::start(layer);',
            'if(layer==0) m2c_public_fixture(bert);\n      blb_wire::bind(bert); blb_wire::start(layer);')
    else:
        main=main.replace('blb_wire::bind(bert_ckks); blb_wire::start(0);',
            'm2c_public_fixture(bert_ckks);\n  blb_wire::bind(bert_ckks); blb_wire::start(0);')
    mainfile.write_text(main)
    wire=(src/'wire_audit.h').read_text()
    wire=wire.replace(' bool active=false;', ' std::map<std::string,std::array<uint64_t,2>> generation;\n bool active=false;',1)
    wire=wire.replace('s.rotations.clear();','s.rotations.clear();s.generation.clear();')
    wire=wire.replace(' for(auto &x:s.rotations)', ''' for(auto &x:s.generation)std::cout<<"M2C_GENERATE block="<<s.block<<" site="<<x.first<<" calls="<<x.second[0]<<" rns_coefficients="<<x.second[1]<<" bytes=0"<<std::endl;
 for(auto &x:s.rotations)''')
    wire+='''
namespace blb_wire {
template<class F> decltype(auto) generated(const char*site,uint64_t coefficients,F run){
 auto &s=state();if(s.active){std::lock_guard<std::mutex> lock(s.lock);auto &v=s.generation[site];++v[0];v[1]+=coefficients;}
 return run();
}
}
'''
    (src/'wire_audit.h').write_text(wire)
    for p in src.iterdir():
        if p.name not in {'he.h','ckks_bert.cpp','ckks_bert_basic.cpp'}:continue
        text=p.read_text();code=masked(text);changes=[]
        for m in re.finditer(r'\b(?:\w+(?:->|\.))*encryptor->encrypt(?:_symmetric)?\s*\(',code):
            end=closing(code,m.end()-1);call_args=arguments(code,m.end()-1,end)
            plain=text[slice(*call_args[0])].strip();call=text[m.start():end+1]
            site=p.name+':'+str(text.count('\n',0,m.start())+1)
            changes.append((m.start(),end+1,'blb_wire::generated("'+site+'", static_cast<const Plaintext&>('+plain+').coeff_count(), [&]() -> decltype(auto) { return '+call+'; })'))
        for a,b,new in sorted(changes,reverse=True):text=text[:a]+new+text[b:]
        p.write_text(text)
    build=(out/'build.sh').read_text()
    compile_line=next(line for line in build.splitlines() if ' -c source/he.cpp ' in line)
    extra=compile_line.replace('source/he.cpp','source/value-extension.cpp').replace('he.cpp.o','value-extension.cpp.o').replace('build/he.cpp.log','build/value-extension.cpp.log')
    build=build.replace('for pid in "${pids[@]}";',extra+'\npids+=("$!")\nfor pid in "${pids[@]}";')
    # The full object defines all XTProtocol symbols, so the archive object is
    # not pulled in; this preserves one implementation, with observation only.
    container=Path('/workspace/OURS')/out.relative_to(ROOT)
    build=build.replace(' -o '+str(container/'build/bin'), ' '+str(container/'build/objects/value-extension.cpp.o')+' -o '+str(container/'build/bin'))
    (out/'build.sh').write_text(build)
    (out/'m2c_manifest.json').write_text(json.dumps(dict(
        mode='vit_unmodified_converter' if args.vit else 'donor_with_equivalent_signed_converter',
        shared_converter_sha256=hashlib.sha256(shared.encode()).hexdigest(),
        complete_real_bert=False,scope='M2C only; donor placeholders/omissions remain; do not label as full BERT numerical validation',
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in src.iterdir() if p.is_file()}),indent=2)+'\n')


if __name__=='__main__':main()
