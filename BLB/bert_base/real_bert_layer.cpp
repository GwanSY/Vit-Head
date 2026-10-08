// Real post-norm BERT layer, using the same Strict BLB kernels as ViT B.
// Per-party validation shares are saved after timing; no tensor is opened
// by the inference protocol. Offline validation is restricted to this fixture.
#include "ckks_bert.h"
#include "wire_audit.h"
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

int bitlength = 37; // SCI-LinearHE's shared public ring-width symbol.

static std::string required_env(const char *name) {
    const char *v = std::getenv(name);
    if (!v || !*v) throw std::runtime_error(std::string("missing ") + name);
    return v;
}

static std::vector<uint64_t> integers(const std::string &path, size_t n) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("missing fixture " + path);
    std::vector<uint64_t> v;
    std::string line;
    while (std::getline(f,line)) {
        for (char &c : line) if (c == ',') c = ' ';
        std::istringstream row(line);
        int64_t x;
        while (row >> x) v.push_back(static_cast<uint64_t>(x));
    }
    if (v.size()!=n) throw std::runtime_error("fixture shape mismatch " + path);
    return v;
}

int main(int argc, char **argv) {
  try {
    int party=0, port=62000;
    for (int i=1;i<argc;++i) {
        std::string a=argv[i];
        if(a.rfind("r=",0)==0) party=std::stoi(a.substr(2));
        if(a.rfind("p=",0)==0) port=std::stoi(a.substr(2));
    }
    if(party!=sci::ALICE && party!=sci::BOB) throw std::runtime_error("invalid party");
    const auto model=required_env("BLB_REAL_MODEL");
    const auto output=required_env("BLB_REAL_OUTPUT");
    constexpr int tokens=128, dim=768, ell=37, scale=12;
    BertModel weights;
    if(party==sci::ALICE) weights=load_model(model,2,1,false);
    Bert_ckks b(party,port,"127.0.0.1",LOCATION::DEVICE,
                party==sci::ALICE ? &weights : nullptr,tokens,false,1,false);
    b.bert_post_norm=true;
    b.linear.bert_post_norm=true;
    if(party==sci::ALICE) b.linear.weights_preprocess(weights,1);
    b.heterogeneous_group_packing_enabled=false;
    if(b.linear.data_lin1_0.image_size!=tokens) throw std::runtime_error("expected 128 tokens");
    auto *fix=b.nonlinear.fpmath[0]->fix;
    auto mask=integers(model+"/inputs_0_mask.txt",tokens);
    std::vector<int64_t> public_mask(mask.begin(),mask.end());
    b.set_public_attention_mask(public_mask);
    std::vector<uint64_t> values(tokens*dim,0);
    if(party==sci::BOB) values=integers(model+"/inputs_0_data.txt",tokens*dim);
    auto x=fix->input(sci::BOB,values.size(),values.data(),true,ell,scale);
    auto parameter=[&](const std::vector<std::vector<uint64_t>> &p) {
        std::vector<uint64_t> v(tokens*dim,0);
        if(party==sci::ALICE) {
            if(p.size()!=1 || p[0].size()!=dim) throw std::runtime_error("LN parameter shape");
            for(int t=0;t<tokens;++t) std::copy(p[0].begin(),p[0].end(),v.begin()+t*dim);
        }
        return fix->input(sci::ALICE,v.size(),v.data(),true,ell,scale);
    };
    // Model/input sharing is before the online block snapshot in both cases.
    auto w1=parameter(b.linear.w_ln_1), b1=parameter(b.linear.b_ln_1);
    auto w2=parameter(b.linear.w_ln_2), b2=parameter(b.linear.b_ln_2);
    std::vector<std::pair<std::string,FixArray>> stages;
    blb_wire::bind(b); blb_wire::start(0);
    const auto start=std::chrono::steady_clock::now();
    b.attention_input_residual_share=x;
    b.attention_input_residual_ready=true;
    auto x5=fix->truncate_reduce(x,7);
    FixArray column(party,x5.size,true,x5.ell,x5.s);
    for(int t=0;t<tokens;++t) for(int d=0;d<dim;++d) column.data[d*tokens+t]=x5.data[t*dim+d];
    auto qkv=b.attention_from_real_input(column,0);
    FixArray qk(party,12*tokens*tokens,true,ell,scale);
    size_t copied=0;
    for(const auto &ct:qkv) {
        const size_t n=std::min(ct.size(),static_cast<size_t>(qk.size)-copied);
        std::copy_n(ct.data(),n,qk.data+copied); copied+=n;
        if(copied==static_cast<size_t>(qk.size)) break;
    }
    if(copied!=static_cast<size_t>(qk.size)) throw std::runtime_error("incomplete QK");
    FixArray qk_row(party,qk.size,true,ell,scale);
    b.linear.plain_cross_packing_postprocess(qk.data,qk_row.data,false,b.linear.data_lin1_0);
    stages.emplace_back("qk",qk_row);
    auto probability=b.softmax_he(qk);
    stages.emplace_back("softmax",probability);
    auto ln1=b.Softmax_v_linear_layer_norm_part1(probability,0);
    stages.emplace_back("ln1_centered",ln1.at("x_flat_avg"));
    stages.emplace_back("ln1_variance",ln1.at("sigma_flat"));
    ln1["w"]=w1; ln1["b"]=b1;
    ln1=b.layer_norm_nonlinear(ln1);
    auto ff1=b.ln_part2_linear3_gelu_part1(ln1,0);
    stages.emplace_back("pre_gelu",ff1.at("x_array"));
    auto gelu=b.gelu_nonlinear(ff1);
    stages.emplace_back("gelu",gelu);
    auto ln2=b.linear4_add_layernorm_part1(gelu,0);
    stages.emplace_back("ln2_centered",ln2.at("x_flat_avg"));
    stages.emplace_back("ln2_variance",ln2.at("sigma_flat"));
    ln2["w"]=w2; ln2["b"]=b2;
    ln2=b.layer_norm_nonlinear(ln2);
    auto result=b.layer_norm_affine_ckks(ln2);
    stages.emplace_back("output",result);
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    blb_wire::stop();
    std::cout<<"REAL_BERT_CONVERSIONS c2m_ciphertexts="<<b.ckks_to_mpc_conversions
             <<" m2c_plaintexts="<<b.mpc_to_ckks_conversions
             <<" field_to_ring_calls="<<b.field_to_ring_calls
             <<" ring_to_field_calls="<<b.ring_to_field_calls<<std::endl;
    std::ofstream meta(output+"/stages_party"+std::to_string(party)+".csv");
    meta<<"stage,elements,ell,scale\n";
    for(const auto &stage:stages) {
        const auto &a=stage.second;
        std::ofstream f(output+"/"+stage.first+".party"+std::to_string(party)+".bin",std::ios::binary);
        f.write(reinterpret_cast<const char*>(a.data),a.size*sizeof(uint64_t));
        if(!f) throw std::runtime_error("cannot save validation share");
        meta<<stage.first<<','<<a.size<<','<<a.ell<<','<<a.s<<'\n';
    }
    std::cout<<"REAL_BERT_EXECUTION status=COMPLETE layers=1 tokens=128 online_seconds="<<seconds
             <<" numeric_acceptance=PENDING_OFFLINE_CHECK\n";
    return 0;
  } catch(const std::exception &e) {
    std::cerr<<"REAL_BERT_EXECUTION status=FAIL reason="<<e.what()<<std::endl;
    return 2;
  }
}
