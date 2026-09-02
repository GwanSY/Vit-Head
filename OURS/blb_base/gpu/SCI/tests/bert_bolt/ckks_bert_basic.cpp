/*
Authors: Deevashwer Rathee
Copyright:
Copyright (c) 2021 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
// #include "FloatingPoint/fp-math.h"
// #include "FloatingPoint/fixed-point.h"
// #include "BuildingBlocks/aux-protocols.h"
// #include "NonLinear/argmax.h"
#include <fstream>
#include <iostream>
#include <thread>
#include <cmath>
#include <vector>
#include "ckks_bert.h"

using namespace sci;
using namespace std;
using namespace seal;

#define MAX_THREADS 1
// #define SCI_HE
#define CKKS_PERF

#ifdef CKKS_PERF
uint64_t c_offline = 0;
uint64_t r_offline = 0;
#endif
/*
example:
    auto t_load_model = high_resolution_clock::now();
    xxx code
    cout << "> [TIMING]: xxx code takes: " << interval(t_load_model) << "sec" << endl;
*/
inline double interval(chrono::_V2::system_clock::time_point start){
    auto end = high_resolution_clock::now();
    auto interval = (end - start)/1e+9;
    return interval.count();
}

inline void NegacyclicRightShiftInplace(seal::Ciphertext& ct, size_t shift,
                                 const seal::SEALContext& context) {
  if (shift == 0 || ct.size() == 0) {
    // nothing to do
    return;
  }

  auto cntxt = context.get_context_data(ct.parms_id());

  size_t num_coeff = ct.poly_modulus_degree();

  std::vector<uint64_t> tmp(shift);
  //  i < N - s  ai*X^i -> ai*X^{i + s}
  // i >= N - s ai*X^i -> -ai*X^{(i + s) % N}
  const auto& modulus = cntxt->parms().coeff_modulus();
  for (size_t k = 0; k < ct.size(); ++k) {
    uint64_t* dst_ptr = ct.data(k);

    for (const auto& prime : modulus) {
      // save [N-s, N)
      std::copy_n(dst_ptr + num_coeff - shift, shift, tmp.data());

      // X^i for i \in [0, N-s)
      for (size_t i = num_coeff - shift; i > 0; --i) {
        dst_ptr[i - 1 + shift] = dst_ptr[i - 1];
      }

      // i \n [N-s, N)
      for (size_t i = 0; i < shift; ++i) {
        dst_ptr[i] = seal::util::negate_uint_mod(tmp[i], prime);
      }

      dst_ptr += num_coeff;
    }
  }
}

inline void MulImageUnitInplace(seal::Ciphertext& ct, const seal::SEALContext& context){
    const bool is_ntt = ct.is_ntt_form();
    const size_t degree = ct.poly_modulus_degree();
    seal::Evaluator evaluator(context);
    if (is_ntt) {
        evaluator.transform_from_ntt_inplace(ct);
    }

    // * X^{N/2}
    NegacyclicRightShiftInplace(ct, degree / 2, context);

    if (is_ntt) {
        evaluator.transform_to_ntt_inplace(ct);
    }
}

inline vector<int> getRNS(uint64_t plain_mod_bitwidth){
    
}

Bert_ckks::Bert_ckks(int party, int port, string address){
    this->party = party;
    this->address = address;
    this->port = port;
    this->io = new NetIO(party == 1 ? nullptr : address.c_str(), port);

    cout << "> Setup Linear" << endl;
    this->linear = Linear_ckks(party, io);
    cout << "> Setup NonLinear" << endl;
    this->nonlinear = NonLinear(party, address, port+1);

    this->prune = prune;
    if(party == ALICE){
        cout << "> Loading and preprocessing weights on server" << endl;
        #ifdef BERT_PERF
        auto t_load_model = high_resolution_clock::now();
        #endif 
        struct BertModel bm = 
            load_model("../../dataset/quantize/mrpc/weights_txt/", NUM_CLASS);

        #ifdef BERT_PERF
        cout << "> [TIMING]: Loading Model takes: " << interval(t_load_model) << "sec" << endl;
        auto t_model_preprocess = high_resolution_clock::now();
        #endif 

        linear.weights_preprocess(bm);

        #ifdef BERT_PERF
        cout << "> [TIMING]: Model Preprocessing takes: " << interval(t_model_preprocess) << "sec" << endl;
        #endif 
    }
    c_offline += this->get_comm();
    r_offline += this->get_round();
    cout << "c_offline:" << c_offline << ", r_offline:" << r_offline << endl;
    cout << "> Bert intialized done!" << endl << endl;

    cout << "> Bert_ckks intialized done!" << endl << endl;

}

Bert_ckks::~Bert_ckks() {
    
}


/*
	input: 解密后的明文share，对于server，是r，对于client，是ifft(x)-r，这里r是uniform random, size是poly degree.
	step:
		1. 双方各自做intt，并且从[0,Q]转为了[-Q/2,Q/2]
		2. 将scale变成f, 位宽变成bw, bw与f均是人为定义参数, bw>log2Q
		3. 在输入l-bit, scale是f的情况下，本地进行fft变化
        4. (可选) trunc-reduce，可以先做3，2和4合并一次。
*/
vector<vector<HE_CKKS::scalar_t>> Bert_ckks::ckks_to_mpc(vector<Plaintext> pts, int degree, uint64_t Q, int bit_out, int s_out, HE_CKKS* he){
    #ifdef BERT_PERF
    auto t_conversion = high_resolution_clock::now();
    #endif 
    cout << "in ckks_to_mpc, degree, Q, bit_out, s_in, s_out:" << degree << "," << Q << "," << bit_out << ","<< round(log2(pts[0].scale())) << "," << s_out << endl;
    int slot = degree/2;
    // step 1
    vector<vector<std::complex<HE_CKKS::scalar_t>>> x_intt(pts.size());
    cout << "pts.size():" << pts.size() << endl;
    for(int i=0;i<pts.size();i++){
        // cout << "i:" << i << endl;
        x_intt[i] = he->ckks_intt(pts[i]);
    }
    vector<HE_CKKS::scalar_t> x_intt_flatten_field,x_intt_flatten_ring(x_intt.size()*x_intt[0].size());
    for(int i=0;i<x_intt.size();i++){
        for(int j=0;j<x_intt[i].size();j++){
            x_intt_flatten_field.push_back(x_intt[i][j].real());
        }
    }
    cout << "step 1 done" << endl;
	// step 2
    int s_in;
    if (abs(pts[0].scale())<1){
        s_in = 0;
    }
    else{
        s_in = round(log2(pts[0].scale()));
    }
    // from (Q,s_in) to (2^bit_out, s_out), s_in is expected to be 0
    nonlinear.field_to_ring(
        1,
        x_intt_flatten_field.data(),
        Q,
        x_intt_flatten_ring.data(),
        x_intt_flatten_field.size(),
        bit_out,
        s_in,
        s_out
    );
    cout << "step 2 done" << endl;
	// step 3
    vector<vector<HE_CKKS::complex_t>> x_decode(pts.size());
    for(int i=0;i<pts.size();i++){
        for(int j=0;j<degree;j++){
            x_decode[i].push_back(HE_CKKS::complex_t(x_intt_flatten_ring[i*degree+j],0));
        }
    }
    cout << "x_decode[0][0]:" << x_decode[0][0].real() << endl;
    vector<vector<HE_CKKS::scalar_t>> y(pts.size());
    vector<vector<HE_CKKS::scalar_t>> y2(pts.size(),vector<HE_CKKS::scalar_t>(slot,0));
    for(int i=0;i<pts.size();i++){
        y[i] = he->ckks_decode(x_decode[i],degree);
    }
    cout << "decode ok" << endl;
    // uint8_t* msb_x = new uint8_t[y[0].size()];
    // for(int i=0;i<y.size();i++){
    //     nonlinear.fpmath[0]->fix->trunc->truncate_with_lsb(y[i].size(), y[i].data(), y2[i].data(),
    //                      1, bit_out, true,
    //                      msb_x);
    // }
    
    cout << "ckks_to_mpc done" << endl;
	return y;
}

/*
	input: l-bit, scale是s_in的share, 维度是(x, slot)
    output: scale是s_out的share
	step:
		1. 先编码进复数，进行本地encode
		2. 从(l-bit, s_in)变至(Q，s_out), s_out = ckks的delta，在ntt中会构建RNS，因此预先把位宽Ext至Q，可以避免对每个RNS分量都做Ext
           这里s_out > s_in (41 vs 30)
		3. 本地做ntt
*/
vector<Plaintext> Bert_ckks::mpc_to_ckks(vector<vector<HE_CKKS::scalar_t>> x, int slot, uint64_t Q, int bitwidth, int s_in, int s_out, HE_CKKS* he){
    cout << "in mpc_to_ckks, slot, Q, bitwidth, s_in, s_out:" << slot << "," << Q << "," << bitwidth << "," << s_in << "," << s_out << endl;
    // step 1
    int poly_degree = slot * 2;
    // cout << "poly_degree:" << poly_degree << endl;
    vector<vector<HE_CKKS::scalar_t>> x_ifft(x.size());
    vector<vector<HE_CKKS::scalar_t>> x_ifft2(x.size(),vector<HE_CKKS::scalar_t>(poly_degree,0));
    for(int i=0;i<x.size();i++){
        x_ifft[i] = he->ckks_encode(x[i],slot);
    }
    // cout << "encode ok" << endl;
    // uint8_t* msb_x = new uint8_t[x_ifft[0].size()];
    // for(int i=0;i<x_ifft.size();i++){
    //     // TODO, now is fake trunc
    //     nonlinear.fpmath[0]->fix->trunc->truncate_with_lsb(x_ifft[i].size(), x_ifft[i].data(), x_ifft2[i].data(),
    //                      1, bitwidth, true,
    //                      msb_x);
    // }
    // cout << "trunc comm:" << get_comm() << endl;
    // cout << "trunc round:" << get_round() << endl;
    cout<<"step 1 done" << endl;
    // step 2
    vector<HE_CKKS::scalar_t> x_intt_flatten_field(x_ifft.size()*x_ifft[0].size()),x_intt_flatten_ring;
    for(int i=0;i<x_ifft.size();i++){
        for(int j=0;j<x_ifft[i].size();j++){
            x_intt_flatten_ring.push_back(x_ifft[i][j]);
        }
    }
    cout << "num ops:"  << x_intt_flatten_ring.size() << endl;
    nonlinear.ring_to_field(1, x_intt_flatten_ring.data(), Q, x_intt_flatten_field.data(), x_intt_flatten_ring.size(), bitwidth, s_in, s_out);
    cout << "step 2 done" << endl;
    // step 3
    vector<vector<HE_CKKS::scalar_t>> x_encode(poly_degree);
    for(int i=0;i<x.size();i++){
        for(int j=0;j<poly_degree;j++){
            x_encode[i].push_back(x_intt_flatten_field[i*poly_degree+j]);
        }
    }
    vector<Plaintext> y(x.size());
    for(int i=0;i<x.size();i++){
        he->ckks_ntt(x_encode[i],poly_degree,pow(2,s_out),y[i]);
    }
    
    cout << "mpc_to_ckks done" << endl;
    return y;
}

// // evaluate input^2, 输入输出均是mpc share, use CKKS in default
// FixArray Bert_ckks::element_wise_square(int32_t dim, FixArray input){
//     int bitwidth = input.ell; //bw
//     int s = input.s;
//     cout << "in element_wise_square, bitwidth, s:" << bitwidth << "," << s << endl;
//     cout << "should be 0! comm:" << this->get_comm() << ", round:" << this->get_round() << endl;
//     linear.set_HE(poly_modulus_degree,vector<int>{bitwidth, s, bitwidth}, s);
//     c_offline += this->get_comm();
//     r_offline += this->get_round();
//     cout << "c_offline:" << c_offline << ", r_offline:" << r_offline << endl;
//     if (dim!=input.size){
//         cout << "Error: input size does not match" << endl;
//         return input;
//     }
//     uint64_t slot_count = poly_modulus_degree / 2;
//     uint64_t num_cts = dim / slot_count;
//     cout << "slot_count:" << slot_count << endl;
//     cout << "num_cts:" << num_cts << endl;
//     vector<vector<HE_CKKS::scalar_t>> vec(num_cts);
//     vector<Plaintext> pts;
//     vector<Ciphertext> cts(num_cts);
//     vector<Plaintext> output_pts(num_cts);
//     for(int i = 0; i < num_cts; i++){
//         Plaintext pt;
//         Ciphertext ct;
//         for(int j = 0; j < slot_count; ++j){
//             vec[i].push_back(input.data[i*slot_count + j]);
//         }
//     }
//     cout << "Q:" << linear.he->get_Q() << endl;
//     this->get_comm();
//     this->get_round();
//     pts = mpc_to_ckks(vec, slot_count, linear.he->get_Q(), bitwidth, s, s);
//     cout << "mpc_to_ckks comm:" << this->get_comm() << endl;
//     cout << "mpc_to_ckks round:" << this->get_round() << endl;
//     cout << "pts.size():" << pts.size() << endl;
//     vector<Ciphertext> half_cts(num_cts/2);
//     // Alice is server, Bob is client
//     if (party == BOB) {
//         for(size_t i=0;i<num_cts;i++){
//             Ciphertext ct;
//             linear.he->encryptor->encrypt(pts[i], ct);
//             cts[i] = ct;
//             if(i<num_cts/2){
//                 half_cts[i] = ct;
//             }
//         }
//         cout << "client send_encrypted_vector" << endl;
// 		// simulate x+y*j optimization, reduce input comm by half
//         // send half_cts instead of cts
// 		send_encrypted_vector(io, half_cts);
// 		// send_encrypted_vector(io, cts);
// 		// vector<Ciphertext> share_client(dim);
// 		recv_encrypted_vector(linear.he->context, io, cts);
//         for(int i=0;i<num_cts;i++){
//             linear.he->decryptor->decrypt(cts[i], output_pts[i]);
//         }
// 	}
//     else{
//         // receive half_cts instead of cts
//         recv_encrypted_vector(linear.he->context, io, half_cts);
//         for(int i=0;i<half_cts.size();i++){
//             seal::Ciphertext ct1;
//             seal::Ciphertext tmp;
//             // x + y*j => x - y*j
//             linear.he->evaluator->complex_conjugate(half_cts[i], *linear.he->gal_keys, tmp);
//             // x + y*j - (x - y*j)*
//             // - 2 * y*j
//             linear.he->evaluator->sub(tmp, half_cts[i], ct1);
//             // - 2 * y*j => 2*y
//             MulImageUnitInplace(ct1, *linear.he->context);
//             // x + y*j + (x - y*j) => 2*x
//             linear.he->evaluator->add_inplace(half_cts[i], tmp);
//             cts[i] = half_cts[i];
//             cts[i+half_cts.size()] = ct1;
//         }
//         linear.he->print_parameters(linear.he->context->get_context_data(cts[0].parms_id())->parms());
//         for(int i=0;i<cts.size();i++){
//             linear.he->evaluator->add_plain_inplace(cts[i], pts[i]);
//             linear.he->evaluator->square_inplace(cts[i]);
//             RelinKeys* relin_keys = linear.he->relin_keys;
//             linear.he->evaluator->relinearize_inplace(cts[i], *relin_keys);
//             linear.he->evaluator->rescale_to_next_inplace(cts[i]);
//         }
//         PRG128 prg;
//         /*
//             生成r并计算x-r，这一过程极为复杂，暂时令r=0
//         */
//         vector<uint64_t> r(dim,0);
//         // prg.random_mod_p<uint64_t>(r.data(), dim, linear.he->get_Q());
//         cout << "server send_encrypted_vector" << endl;
//         send_encrypted_vector(io, cts);
//         for(int i=0;i<num_cts;i++){
//             vector<double> pod_matrix(poly_modulus_degree/2, 0.0);
//             Plaintext tmp;
//             linear.he->encoder->encode(pod_matrix, cts[i].parms_id(), cts[i].scale(), tmp);
//             output_pts[i] = tmp;
//         }
//     }
//     linear.he->print_parameters(linear.he->context->get_context_data(cts[0].parms_id())->parms());
//     vector<vector<HE_CKKS::scalar_t>> output_vector;
//     cout << "send and receive comm:" << this->get_comm() << endl;
//     cout << "send and receive round:" << this->get_round() << endl;
//     output_vector = ckks_to_mpc(output_pts, poly_modulus_degree, linear.he->get_Q(cts[0]), bitwidth, s);
//     cout << "ckks_to_mpc comm:" << this->get_comm() << endl;
//     cout << "ckks_to_mpc round:" << this->get_round() << endl;
//     FixArray output(party, dim, input.signed_, bitwidth, s);
//     for(int i = 0; i < num_cts; i++){
//         for(int j = 0; j < slot_count; ++j){
//             output.data[i*slot_count + j] = output_vector[i][j];
//         }
//     }
//     cout << "element_wise_square done" << endl;
//     return output;
// }

// // evaluate input^2, 输入输出均是mpc share, use BFV in default
// FixArray Bert_ckks::element_wise_square_bfv(int32_t dim, FixArray input){
//     int bitwidth = input.ell; //bw
//     int s = input.s;
//     cout << "in element_wise_square, bitwidth, s:" << bitwidth << "," << s << endl;
//     cout << "should be 0! comm:" << this->get_comm() << ", round:" << this->get_round() << endl;
//     linear.set_HE_bfv(poly_modulus_degree,vector<int>{bitwidth, s, bitwidth}, bitwidth+s);
//     c_offline += this->get_comm();
//     r_offline += this->get_round();
//     cout << "c_offline:" << c_offline << ", r_offline:" << r_offline << endl;
//     if (dim!=input.size){
//         cout << "Error: input size does not match" << endl;
//         return input;
//     }
//     uint64_t slot_count = poly_modulus_degree / 2;
//     uint64_t num_cts = dim / slot_count;
//     cout << "slot_count:" << slot_count << endl;
//     cout << "num_cts:" << num_cts << endl;
//     vector<vector<HE_CKKS::scalar_t>> vec(num_cts);
//     vector<Plaintext> pts;
//     vector<Ciphertext> cts(num_cts);
//     vector<Plaintext> output_pts(num_cts);
//     for(int i = 0; i < num_cts; i++){
//         Plaintext pt;
//         Ciphertext ct;
//         for(int j = 0; j < slot_count; ++j){
//             vec[i].push_back(input.data[i*slot_count + j]);
//         }
//     }
//     cout << "Q:" << linear.he->get_Q() << endl;
//     this->get_comm();
//     this->get_round();
//     pts = mpc_to_ckks(vec, slot_count, linear.he->get_Q(), bitwidth, s, s);
//     cout << "mpc_to_ckks comm:" << this->get_comm() << endl;
//     cout << "mpc_to_ckks round:" << this->get_round() << endl;
//     cout << "pts.size():" << pts.size() << endl;
//     vector<Ciphertext> half_cts(num_cts/2);
//     // Alice is server, Bob is client
//     if (party == BOB) {
//         for(size_t i=0;i<num_cts;i++){
//             Ciphertext ct;
//             linear.he->encryptor->encrypt(pts[i], ct);
//             cts[i] = ct;
//             if(i<num_cts/2){
//                 half_cts[i] = ct;
//             }
//         }
//         cout << "client send_encrypted_vector" << endl;
// 		// simulate x+y*j optimization, reduce input comm by half
//         // send half_cts instead of cts
// 		send_encrypted_vector(io, half_cts);
// 		// send_encrypted_vector(io, cts);
// 		// vector<Ciphertext> share_client(dim);
// 		recv_encrypted_vector(linear.he->context, io, cts);
//         for(int i=0;i<num_cts;i++){
//             linear.he->decryptor->decrypt(cts[i], output_pts[i]);
//         }
// 	}
//     else{
//         // receive half_cts instead of cts
//         recv_encrypted_vector(linear.he->context, io, half_cts);
//         for(int i=0;i<half_cts.size();i++){
//             seal::Ciphertext ct1;
//             seal::Ciphertext tmp;
//             // x + y*j => x - y*j
//             linear.he->evaluator->complex_conjugate(half_cts[i], *linear.he->gal_keys, tmp);
//             // x + y*j - (x - y*j)*
//             // - 2 * y*j
//             linear.he->evaluator->sub(tmp, half_cts[i], ct1);
//             // - 2 * y*j => 2*y
//             MulImageUnitInplace(ct1, *linear.he->context);
//             // x + y*j + (x - y*j) => 2*x
//             linear.he->evaluator->add_inplace(half_cts[i], tmp);
//             cts[i] = half_cts[i];
//             cts[i+half_cts.size()] = ct1;
//         }
//         linear.he->print_parameters(linear.he->context->get_context_data(cts[0].parms_id())->parms());
//         for(int i=0;i<cts.size();i++){
//             linear.he->evaluator->add_plain_inplace(cts[i], pts[i]);
//             linear.he->evaluator->square_inplace(cts[i]);
//             RelinKeys* relin_keys = linear.he->relin_keys;
//             linear.he->evaluator->relinearize_inplace(cts[i], *relin_keys);
//             linear.he->evaluator->rescale_to_next_inplace(cts[i]);
//         }
//         PRG128 prg;
//         /*
//             生成r并计算x-r，这一过程极为复杂，暂时令r=0
//         */
//         vector<uint64_t> r(dim,0);
//         // prg.random_mod_p<uint64_t>(r.data(), dim, linear.he->get_Q());
//         cout << "server send_encrypted_vector" << endl;
//         send_encrypted_vector(io, cts);
//         for(int i=0;i<num_cts;i++){
//             vector<double> pod_matrix(poly_modulus_degree/2, 0.0);
//             Plaintext tmp;
//             linear.he->encoder->encode(pod_matrix, cts[i].parms_id(), cts[i].scale(), tmp);
//             output_pts[i] = tmp;
//         }
//     }
//     linear.he->print_parameters(linear.he->context->get_context_data(cts[0].parms_id())->parms());
//     vector<vector<HE_CKKS::scalar_t>> output_vector;
//     cout << "send and receive comm:" << this->get_comm() << endl;
//     cout << "send and receive round:" << this->get_round() << endl;
//     output_vector = ckks_to_mpc(output_pts, poly_modulus_degree, linear.he->get_Q(cts[0]), bitwidth, s);
//     cout << "ckks_to_mpc comm:" << this->get_comm() << endl;
//     cout << "ckks_to_mpc round:" << this->get_round() << endl;
//     FixArray output(party, dim, input.signed_, bitwidth, s);
//     for(int i = 0; i < num_cts; i++){
//         for(int j = 0; j < slot_count; ++j){
//             output.data[i*slot_count + j] = output_vector[i][j];
//         }
//     }
//     cout << "element_wise_square done" << endl;
//     return output;
// }


/*
    evaluate gelu poly, 输入输出均是mpc share
    ax^4+bx^3+cx^2+dx+e+0.5x
    a = 0.020848611754127593
    b = -0.18352506127082727
    c = 0.5410550166368381
    d = -0.03798164612714154
    e = 0.001620808531841547
*/ 
vector<Ciphertext> Bert_ckks::eval_gelu_poly(vector<Ciphertext> poly1, HE_CKKS* he){
    double cons_list[] = {0.020848611754127593,-0.18352506127082727,0.5410550166368381,-0.03798164612714154,0.001620808531841547};
    int num_cts = poly1.size();
    int poly_modulus_degree = he->poly_modulus_degree;
    int slot_num = poly_modulus_degree / 2;
    vector<Ciphertext> poly4(num_cts),poly3(num_cts),poly2(num_cts), output_cts(num_cts);
    // 先计算x,x^2,x^3,x^4，再对齐它们的Q和scale
    he->print_parameters(he->context->get_context_data(poly1[0].parms_id())->parms());
    vector<double> scale_up(slot_num, 1);
    bool scale_up_encode=false;
    Plaintext scale_up1, scale_up2, scale_up3;
    for(int i=0;i<num_cts;i++){
        he->evaluator->square(poly1[i], poly2[i]);
        he->evaluator->relinearize_inplace(poly2[i], *he->relin_keys);
        he->evaluator->rescale_to_next_inplace(poly2[i]);
        he->evaluator->square(poly2[i], poly4[i]);
        he->evaluator->relinearize_inplace(poly4[i], *he->relin_keys);
        he->evaluator->rescale_to_next_inplace(poly4[i]);
        // cout << "part1 ok" << endl;
        he->evaluator->mod_switch_to_next_inplace(poly1[i]);
        he->evaluator->multiply(poly2[i], poly1[i], poly3[i]);
        he->evaluator->relinearize_inplace(poly3[i], *he->relin_keys);
        he->evaluator->rescale_to_next_inplace(poly3[i]);
        
        he->evaluator->mod_switch_to_inplace(poly1[i], poly4[i].parms_id());
        he->evaluator->mod_switch_to_inplace(poly2[i], poly4[i].parms_id());
        if (!scale_up_encode){
            he->encoder->encode(scale_up, poly4[i].parms_id(), pow(2,12), scale_up1);
            he->encoder->encode(scale_up, poly4[i].parms_id(), pow(2,8), scale_up2);
            he->encoder->encode(scale_up, poly4[i].parms_id(), pow(2,4), scale_up3);
            scale_up_encode = true;
        }
        he->evaluator->multiply_plain_inplace(poly1[i], scale_up1);
        he->evaluator->multiply_plain_inplace(poly2[i], scale_up2);
        he->evaluator->multiply_plain_inplace(poly3[i], scale_up3);
        // poly1[i].scale() = poly3[i].scale();
        // poly2[i].scale() = poly3[i].scale();
        // poly4[i].scale() = poly3[i].scale();
    }
    he->print_parameters(he->context->get_context_data(poly2[0].parms_id())->parms());
    he->print_parameters(he->context->get_context_data(poly3[0].parms_id())->parms());
    he->print_parameters(he->context->get_context_data(poly4[0].parms_id())->parms());
    cout << "poly1~4 scale:" << endl;
    cout << log2(poly1[0].scale()) << "," << poly1[0].scale() << endl;
    cout << log2(poly2[0].scale()) << "," << poly2[0].scale() << endl;
    cout << log2(poly3[0].scale()) << "," << poly3[0].scale() << endl;
    cout << log2(poly4[0].scale()) << "," << poly4[0].scale() << endl;
    cout << "part2 ok" << endl;
    // 现场encode cons_list
    vector<Plaintext> cons_pts(5);
    vector<double> cons_matrix4(slot_num, cons_list[0]);
    vector<double> cons_matrix3(slot_num, cons_list[1]);
    vector<double> cons_matrix2(slot_num, cons_list[2]);
    vector<double> cons_matrix1(slot_num, cons_list[3]);
    vector<double> cons_matrix0(slot_num, cons_list[4]);
    vector<double> cons_matrix(slot_num, 0.5);
    he->encoder->encode(cons_matrix4, poly4[0].parms_id(), pow(2,he->he_scale), cons_pts[0]);
    he->encoder->encode(cons_matrix3, poly3[0].parms_id(), pow(2,he->he_scale), cons_pts[1]);
    he->encoder->encode(cons_matrix2, poly2[0].parms_id(), pow(2,he->he_scale), cons_pts[2]);
    he->encoder->encode(cons_matrix1, poly1[0].parms_id(), pow(2,he->he_scale), cons_pts[3]);
    
    // 注意规则：相同level才可乘，相同level且相同scale才可加，我们忽略了0.5x，显然这一项可以和|x|合并
    for(int i=0;i<num_cts;i++){
        // cout << "poly4 (Q,scale):" <<log2(he->get_Q(poly4[i]))<<","<< log2(poly4[i].scale()) << endl;
        // cout << "cons_pts[0] (Q,scale):" <<log2(he->get_Q(cons_pts[0]))<<","<< log2(cons_pts[0].scale()) << endl;
        he->evaluator->multiply_plain_inplace(poly4[i], cons_pts[0]);
        he->evaluator->multiply_plain_inplace(poly3[i], cons_pts[1]);
        he->evaluator->multiply_plain_inplace(poly2[i], cons_pts[2]);
        he->evaluator->multiply_plain_inplace(poly1[i], cons_pts[3]);
        // Ciphertext ct_half_x;
        // Plaintext pt_half_x;
        // he->encoder->encode(cons_matrix, poly1[i].parms_id(), poly1[i].scale(), pt_half_x);
        // he->evaluator->multiply_plain(poly1[i], pt_half_x, ct_half_x);
        // cout << "part3 ok" << endl;
        
        // cout << poly1[0].scale() << endl;
        // cout << poly2[0].scale() << endl;
        // cout << poly3[0].scale() << endl;
        // cout << poly4[0].scale() << endl;
        he->evaluator->rescale_to_next_inplace(poly4[i]);
        he->evaluator->rescale_to_next_inplace(poly3[i]);
        he->evaluator->rescale_to_next_inplace(poly2[i]);
        he->evaluator->rescale_to_next_inplace(poly1[i]);
        he->evaluator->rescale_to_next_inplace(poly4[i]);
        he->evaluator->rescale_to_next_inplace(poly3[i]);
        he->evaluator->rescale_to_next_inplace(poly2[i]);
        he->evaluator->rescale_to_next_inplace(poly1[i]);
        // now, scale=30, Q=60
        // cout << "scale" << endl;
        // cout << log2(poly1[0].scale()) << endl;
        // cout << log2(poly2[0].scale()) << endl;
        // cout << log2(poly3[0].scale()) << endl;
        // cout << log2(poly4[0].scale()) << endl;
        poly2[i].scale() = poly1[i].scale();
        poly3[i].scale() = poly1[i].scale();
        poly4[i].scale() = poly1[i].scale();
        he->evaluator->add(poly1[i], poly2[i], output_cts[i]);
        // he->evaluator->mod_switch_to_next_inplace(output_cts[i]);
        // cout << "part4 ok" << endl;
        // cout << output_cts[i].scale() << endl;
        // cout << poly3[i].scale() << endl;
        // cout << poly4[i].scale() << endl;
        // output_cts[i].scale() = poly3[i].scale();
        he->evaluator->add_inplace(output_cts[i], poly3[i]);
        // output_cts[i].scale() = poly4[i].scale();
        he->evaluator->add_inplace(output_cts[i], poly4[i]);
        he->encoder->encode(cons_matrix0, output_cts[i].parms_id(), output_cts[i].scale(), cons_pts[4]);
        // cout << "part5 ok" << endl;
        // cout << output_cts[i].scale() << endl;
        // cout << cons_pts[i].scale() << endl;
        he->evaluator->add_plain_inplace(output_cts[i], cons_pts[4]);
        // cout << "part6 ok" << endl;
    }
    cout << "eval_gelu_poly done" << endl;
    return output_cts;
}

// void Bert_ckks::mp2ml_test(){
//     #ifdef BERT_PERF
//     auto t_conversion = high_resolution_clock::now();
//     #endif 
//     if (party == BOB) {
//         vector<double> x = {1,2,3,4,5,6,7,8};
//     }
//     // step 1
//     vector<vector<std::complex<HE_CKKS::scalar_t>>> x_intt(pts.size());
//     cout << "pts.size():" << pts.size() << endl;
//     for(int i=0;i<pts.size();i++){
//         // cout << "i:" << i << endl;
//         x_intt[i] = linear.he->ckks_intt(pts[i]);
//     }
//     vector<HE_CKKS::scalar_t> x_intt_flatten_field,x_intt_flatten_ring(x_intt.size()*x_intt[0].size());
//     for(int i=0;i<x_intt.size();i++){
//         for(int j=0;j<x_intt[i].size();j++){
//             x_intt_flatten_field.push_back(x_intt[i][j].real());
//         }
//     }
//     cout << "step 1 done" << endl;
// 	// step 2
//     int s_in = ceil(log2(pts[0].scale()));
//     s_in = s_out;
//     nonlinear.field_to_ring(
//         1,
//         x_intt_flatten_field.data(),
//         Q,
//         x_intt_flatten_ring.data(),
//         x_intt_flatten_field.size(),
//         bit_out,
//         s_in,
//         s_out
//     );
//     cout << "step 2 done" << endl;
// 	// step 3
//     vector<vector<HE_CKKS::complex_t>> x_decode(pts.size());
//     for(int i=0;i<pts.size();i++){
//         for(int j=0;j<size;j++){
//             x_decode[i].push_back(HE_CKKS::complex_t(x_intt_flatten_ring[i*size+j],0));
//         }
//     }
//     cout << "x_decode[0][0]:" << x_decode[0][0].real() << endl;
//     vector<vector<HE_CKKS::scalar_t>> y(pts.size());
//     for(int i=0;i<pts.size();i++){
//         y[i] = linear.he->ckks_decode(x_decode[i],size);
//     }
//     cout << "ckks_to_mpc done" << endl;
// 	return y;
// }


std::map<char*,FixArray> Bert_ckks::layer_norm_nonlinear(map<char*,FixArray> input) {
    FPMath *fpmath = nonlinear.fpmath[0];
    FixArray x_flat_avg = input["x_flat_avg"];
    FixArray sigma = input["sigma_flat"];
    int bitwidth = x_flat_avg.ell;
    int s = x_flat_avg.s;
    int num_ops = sigma.size;
    int dim = x_flat_avg.size / num_ops;
    cout << "in layer_norm_nonlinear, bitwidth,s,num_ops,dim:" << bitwidth << "," << s << "," << num_ops << "," << dim << endl;
    cout << "sigma.size, should be L:" << sigma.size << endl;
    // 计算平方根倒数
    FixArray sigma_sqrt = fpmath->sqrt(sigma, true);
    // cout << "sqrt comm:" << this->get_comm() << endl;
    // cout << "sqrt round:" << this->get_round() << endl;
    //point8
    FixArray sigma_flat(party, x_flat_avg.size, sigma_sqrt.signed_, bitwidth, s);
    for(int i = 0; i < num_ops; i++){
        for(int j = 0; j < dim; j++){
            sigma_flat.data[i*dim + j] = sigma_sqrt.data[i];
        }
    }
    // 广播乘法
    cout << "x_flat_avg bitwidth,s:" << x_flat_avg.ell << "," << x_flat_avg.s << endl;
    map<char*,FixArray> result;
    result["x_flat_avg"] = x_flat_avg;
    result["sigma_flat"] = sigma_flat;
    FixArray w = fpmath->fix->input(party, x_flat_avg.size, 1, true, bitwidth, s);
    FixArray b = fpmath->fix->input(party, x_flat_avg.size, 1, true, bitwidth, s);
    result["w"] = w;
    result["b"] = b;
    cout << "layer_norm_nonlinear done" << endl;
    return result;
}


FixArray Bert_ckks::gelu_nonlinear(std::map<char*,FixArray> input){
    FixArray x_array = input["x_array"];
    FixArray gelu_poly_flat = input["gelu_poly_flat"];
    FPMath *fpmath = nonlinear.fpmath[0];
    int this_party = party;
    int N = x_array.size;
    int bitwidth = x_array.ell;
    int s = x_array.s;

    BoolArray msb_x = fpmath->fix->MSB(x_array);
    FixArray neg_x = fpmath->fix->mul(x_array, -1);
    // point 2
    
    FixArray y = fpmath->fix->if_else(msb_x, neg_x, x_array);

    BoolArray lt27 = fpmath->fix->LT(y, 2.7*(1 << s));
    // point 8
    BoolArray all_0 = fpmath->bool_op->input(ALICE, N, uint8_t(0));
    FixArray x_plus_y = fpmath->fix->add(x_array, y);   
    FixArray half_x_plus_y = fpmath->fix->right_shift(x_plus_y, 1, all_0.data);
    // point 9
    cout << "gelu_poly_flat(bitwidth,s):" << gelu_poly_flat.ell << "," << gelu_poly_flat.s << endl;
    cout << "x array bitwidth,s:" << x_array.ell << "," << x_array.s << endl;
    cout << "half_x_plus_y(bitwidth,s):" << half_x_plus_y.ell << "," << half_x_plus_y.s << endl;
    half_x_plus_y.s = s;
    //half_x_plus_y = (x+|x|)/2 = max(x,0), if |x|<2.7, 则 gelu_y, 否则half_x_plus_y
    FixArray ret = fpmath->fix->if_else(lt27, gelu_poly_flat, half_x_plus_y);

    // BoolArray msb_ret = fpmath->bool_op->AND(msb_x, lt27);
    // // point 10
    
    // ret = fpmath->fix->extend(ret, 37, msb_ret.data);
    // ret = fpmath->fix->right_shift(ret, 7, msb_ret.data);

    return ret;
}


FixArray Bert_ckks::softmax(FixArray input) {
    FPMath *fpmath = nonlinear.fpmath[0];
    FCMetadata data = linear.data_lin1_0;
    int softmax_dim = data.image_size;
    int qk_size = PACKING_NUM*softmax_dim*softmax_dim;
    int softmax_output_size = PACKING_NUM*softmax_dim*softmax_dim;
    uint64_t* softmax_input_row = new uint64_t[qk_size];
    cout << "OK" << endl;
    linear.plain_cross_packing_postprocess(
        input.data, 
        softmax_input_row,
        // we need row packing
        false,
        data);
    cout << "OK2" << endl;
    // may be need a truncation
    uint64_t* softmax_output_row = new uint64_t[softmax_output_size];
    uint64_t* softmax_l_row = new uint64_t[softmax_output_size];
    // Softmax
    nonlinear.softmax(
        1,
        softmax_input_row,
        softmax_output_row,
        softmax_l_row,
        12*softmax_dim,
        softmax_dim,
        NL_ELL,
        NL_SCALE);
    cout << "OK3" << endl;
    FixArray result = fpmath->fix->input(party, softmax_output_size, softmax_output_row, true, 40, 20);
    return result;
}

