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
#include "ckks_bert_basic.cpp"

using namespace sci;
using namespace std;
using namespace seal;

#define MAX_THREADS 1
// #define SCI_HE

/*
    do not support multiple threads
    dim is hidden dimension, num_ops is sequence length
*/ 
// std::map<char*,FixArray> Bert_ckks::layer_norm_part1(uint64_t *x, int num_ops, int dim, int bitwidth, int s) {
//     cout << "bitwidth, s:" << bitwidth << "," << s << endl;
//     map<char*,FixArray> result;
//     FPMath *fpmath = nonlinear.fpmath[0];
//     // ArgMaxProtocol<uint64_t> *argmax_oracle;
//     int this_party = party;
//     vector<FixArray> input_array;
//     for(int i = 0; i < num_ops; i++){
//         input_array.push_back(fpmath->fix->input(this_party, dim, &x[i*dim], true, bitwidth, s));
//     }
//     FixArray w = fpmath->fix->input(this_party, num_ops*dim, 1, true, bitwidth, s);
//     FixArray b = fpmath->fix->input(this_party, num_ops*dim, 1, true, bitwidth, s);
//     // layer_norm part1 from bolt, change OT to HE
//     uint64_t bw_x = bitwidth+s;
//     uint64_t ell_mask = (bitwidth == 64 ? -1 : ((1ULL << bitwidth) - 1));
//     uint64_t bw_x_mask = (bw_x == 64 ? -1 : ((1ULL << bw_x) - 1));
//     bool signed_ = input_array[0].signed_;
//     vector<FixArray> ret(num_ops);
//     BoolArray all_0 = fpmath->bool_op->input(ALICE, num_ops*dim, uint8_t(0));
//     BoolArray all_1 = fpmath->bool_op->input(ALICE, num_ops*dim, 1);
//     // sum是(ell,s)，将其扩展至(ell+s,s)
//     FixArray sum = fpmath->fix->tree_sum(input_array);
//     cout << "comm:" << this->get_comm() << endl;
//     cout << "round:" << this->get_round() << endl;
//     sum = fpmath->fix->extend(sum,bitwidth+s);
//     cout << "extend comm:" << this->get_comm() << endl;
//     cout << "extend round:" << this->get_round() << endl;
//     uint64_t dn_scalar = uint64_t(((1.0 / dim) * pow(2, s))) & bw_x_mask;
//     //第一个mul是scaler乘法，维度是N, 乘数还是明文1/n，因此可以将sum位宽扩展为ell+s，这样无需任何协议，本地乘法即可。
//     //avg位宽，scale保持为(ell, s)
//     for(int i = 0; i < num_ops; i++){
//         sum.data[i] = (sum.data[i] * dn_scalar) & bw_x_mask;
//     }
//     sum.s = s+s;
//     // sum是(ell+s,2s)，将其tr至(ell,s)
//     FixArray avg = fpmath->fix->truncate_reduce(sum, s);
//     cout << "tr comm:" << this->get_comm() << endl;
//     cout << "tr round:" << this->get_round() << endl;
//     FixArray avg_flat(party, num_ops*dim, sum.signed_, bitwidth, s);
//     for(int i = 0; i < num_ops; i++){
//         for(int j = 0; j < dim; j++){
//             avg_flat.data[i*dim + j] = avg.data[i];
//         }
//     }
//     // 计算(x-x_avg)^2, x_flat_avg是(ell,s)
//     FixArray x_flat = concat(input_array);
//     FixArray x_flat_avg = fpmath->fix->sub(x_flat, avg_flat);
//     // x_flat_avg = fpmath->fix->extend(x_flat_avg, bitwidth+s);
//     cout << "extend2 comm:" << this->get_comm() << endl;
//     cout << "extend2 round:" << this->get_round() << endl;
//     //第二个mul是element-wise乘法，维度是N*n，输入(ell,s)，输出是(ell,s),采用CKKS HE。
//     FixArray x_flat_avg_square = element_wise_square(x_flat_avg.size,x_flat_avg);
//     // Bolt's implementation
//     // BoolArray msb_x_avg = fpmath->fix->MSB(x_flat_avg);
//     // FixArray x_flat_avg_square = fpmath->fix->mul(x_flat_avg, x_flat_avg, bitwidth + s, msb_x_avg.data, msb_x_avg.data);
//     // x_flat_avg_square = fpmath->fix->truncate_reduce(x_flat_avg_square, s);
//     cout << "element-wise mult. comm:" << this->get_comm() << endl;
//     cout << "element-wise mult. round:" << this->get_round() << endl;
//     //point6
//     vector<FixArray> square_group(num_ops);
//     for(int i = 0; i < num_ops; i++){
//         square_group[i] = FixArray(party, dim, signed_, bitwidth, s);
//         memcpy(square_group[i].data, &x_flat_avg_square.data[i*dim], dim*sizeof(uint64_t));
//     }
//     // square_sum是(ell,s)，将其扩展至(ell+s,s)
//     FixArray square_sum = fpmath->fix->tree_sum(square_group);
//     uint8_t* msb_0 = new uint8_t[square_sum.size];
//     memset(msb_0, 0, square_sum.size);
//     square_sum = fpmath->fix->extend(square_sum,bitwidth+s,msb_0);
//     cout << "ext2 comm:" << this->get_comm() << endl;
//     cout << "ext2 round:" << this->get_round() << endl;
//     // square_sum*(1/n)，本地乘法
//     for(int i = 0; i < square_sum.size; i++){
//         square_sum.data[i] = (square_sum.data[i] * dn_scalar) & bw_x_mask;
//     }
//     square_sum.s = 2*s;
//     // square_sum是(ell+s,2s)，将其tr至(ell,s)
//     // cout << "square_sum: (bitwidth,s):" << square_sum.ell << "," << square_sum.s << endl;
//     square_sum = fpmath->fix->truncate_reduce(square_sum, s, msb_0);
//     cout << "tr2 comm:" << this->get_comm() << endl;
//     cout << "tr2 round:" << this->get_round() << endl;
//     // cout << "after TR, square_sum: (bitwidth,s):" << square_sum.ell << "," << square_sum.s << endl;
//     //point7
//     // 计算平方根倒数
//     FixArray sigma = fpmath->sqrt(square_sum, true);
//     cout << "sqrt comm:" << this->get_comm() << endl;
//     cout << "sqrt round:" << this->get_round() << endl;
//     //point8
//     FixArray sigma_flat(party, num_ops*dim, sum.signed_, bitwidth, s);
//     for(int i = 0; i < num_ops; i++){
//         for(int j = 0; j < dim; j++){
//             sigma_flat.data[i*dim + j] = sigma.data[i];
//         }
//     }
//     // 广播乘法
//     cout << "x_flat_avg bitwidth,s:" << x_flat_avg.ell << "," << x_flat_avg.s << endl;
//     result["x_flat_avg"] = x_flat_avg;
//     result["sigma_flat"] = sigma_flat;
//     result["w"] = w;
//     result["b"] = b;
//     // FixArray x_avg_sigma = fix->mul(x_flat_avg, sigma_flat, ell+s, msb_x_avg.data, all_0.data);
//     // x_avg_sigma = fix->truncate_reduce(x_avg_sigma, s);
//     // //point9
//     // // Weight and Bias
//     // x_avg_sigma = fix->mul(x_avg_sigma, w, ell+s);
//     // x_avg_sigma = fix->truncate_reduce(x_avg_sigma, s);
//     // x_avg_sigma = fix->add(x_avg_sigma, b);
//     cout << "layer_norm_part1 done" << endl;
//     return result;
// }

/*
    step1. 计算layernorm结尾的两个element-wise乘法
    step2. 计算attention
    input 的 bitwidth 与 scale 需要是正确的MPC下的值
*/
vector<vector<HE_CKKS::scalar_t>> Bert_ckks::layer_norm_part2_attention(std::map<char*,FixArray> input){
    // mult. depth = 4
    FixArray x_flat_avg = input["x_flat_avg"];
    FixArray sigma_flat = input["sigma_flat"];
    FixArray w = input["w"];
    FixArray b = input["b"];

    int bitwidth = x_flat_avg.ell;
    int s = x_flat_avg.s;
    uint64_t dim = x_flat_avg.size;
    cout << "should be 0! comm:" << this->get_comm() << ", round:" << this->get_round() << endl;
    HE_CKKS* he = linear.he1;
    uint64_t slot_count = he->num_slots;
    uint64_t degree = he->poly_modulus_degree;
    uint64_t num_cts = dim / slot_count;
    vector<vector<HE_CKKS::scalar_t>> vec_x_flat_avg(num_cts), vec_sigma_flat(num_cts);
    for(int i = 0; i < num_cts; i++){
        for(int j = 0; j < slot_count; j++){
            vec_x_flat_avg[i].push_back(x_flat_avg.data[i*slot_count + j]);
            vec_sigma_flat[i].push_back(sigma_flat.data[i*slot_count + j]);
        }
    }
    
    vector<Plaintext> pts_x_flat_avg, pts_sigma_flat;
    
    FCMetadata data = linear.data_lin1_0;
    vector<Ciphertext> output_cts(
        data.image_size * data.image_size * 12 / data.slot_count 
        + data.image_size * data.filter_w * 12 / data.slot_count
    );
    vector<Plaintext> output_pts(
        data.image_size * data.image_size * 12 / data.slot_count 
        + data.image_size * data.filter_w * 12 / data.slot_count
    );
    cout << this->get_comm() << endl;
    cout << this->get_round() << endl;
    
    pts_x_flat_avg = mpc_to_ckks(vec_x_flat_avg, slot_count, he->get_Q(), bitwidth, s, he->he_scale, he);
    // MARK: Can be parallel
    // cout << "mpc_to_ckks1 comm:" << this->get_comm() << endl;
    // cout << "mpc_to_ckks1 round:" << this->get_round() << endl;
    pts_sigma_flat = mpc_to_ckks(vec_sigma_flat, slot_count, he->get_Q(), bitwidth, s, he->he_scale, he);
    // cout << "mpc_to_ckks2 comm:" << this->get_comm() << endl;
    // cout << "mpc_to_ckks2 round:" << this->get_round() << endl;
    this->conversion_comm += this->get_comm();
    this->conversion_round += this->get_round();
    vector<Ciphertext> cts_x_flat(pts_x_flat_avg.size()), cts_sigma_flat(pts_sigma_flat.size());
    vector<Ciphertext> cts_x_flat_half(cts_x_flat.size()/2);
    vector<Ciphertext> cts_sigma_flat_half(cts_sigma_flat.size()/2);
    if (party == BOB) {
        for(size_t i=0;i<num_cts;i++){
            Ciphertext ct1,ct2;
            he->encryptor->encrypt(pts_x_flat_avg[i], ct1);
            he->encryptor->encrypt(pts_sigma_flat[i], ct2);
            cts_x_flat[i] = ct1;
            cts_sigma_flat[i] = ct2;
            if(i<num_cts/2){
                cts_x_flat_half[i] = cts_x_flat[i];
                cts_sigma_flat_half[i] = cts_sigma_flat[i];
            }
        }
        cout << "send_encrypted_vector" << endl;
		// test_util();
		send_encrypted_vector(io, cts_x_flat_half);
        send_encrypted_vector(io, cts_sigma_flat_half);
        cout << "send_encrypted_vector done" << endl;
		// send_encrypted_vector(io, cts);
		// vector<Ciphertext> share_client(dim);
		recv_encrypted_vector(he->context, io, output_cts);
        for(int i=0;i<output_cts.size();i++){
            he->decryptor->decrypt(output_cts[i], output_pts[i]);
        }
	}
    else{
        recv_encrypted_vector(he->context, io, cts_x_flat_half);
        recv_encrypted_vector(he->context, io, cts_sigma_flat_half);
        // simulate complex encoding
        for(int i=0;i<cts_x_flat_half.size();i++){
            seal::Ciphertext ct1,ct2;
            seal::Ciphertext tmp,tmp2;
            // x + y*j => x - y*j
            he->evaluator->complex_conjugate(cts_x_flat_half[i], *he->gal_keys, tmp);
            // x + y*j - (x - y*j)*
            // - 2 * y*j
            he->evaluator->sub(tmp, cts_x_flat_half[i], ct1);
            // - 2 * y*j => 2*y
            MulImageUnitInplace(ct1, *he->context);
            // x + y*j + (x - y*j) => 2*x
            he->evaluator->add_inplace(cts_x_flat_half[i], tmp);
            cts_x_flat[i] = cts_x_flat_half[i];
            cts_x_flat[i+cts_x_flat_half.size()] = ct1;

            he->evaluator->complex_conjugate(cts_sigma_flat_half[i], *he->gal_keys, tmp2);
            he->evaluator->sub(tmp2, cts_sigma_flat_half[i], ct2);
            MulImageUnitInplace(ct2, *he->context);
            he->evaluator->add_inplace(cts_sigma_flat_half[i], tmp2);
            cts_sigma_flat[i] = cts_sigma_flat_half[i];
            cts_sigma_flat[i+cts_sigma_flat_half.size()] = ct2;
        }
        cout << "num of input cts:" << cts_x_flat.size()+cts_sigma_flat.size() << endl;
        cout << "bitwidth of input ct:" << cts_x_flat[0].coeff_modulus_size() << endl;
        cout << "recv_encrypted_vector" << endl;
        // cout << "cts_x_flat[0].scale()" << cts_x_flat[0].scale() << endl;
        for(int i=0;i<cts_x_flat.size();i++){
            /*
                get x_avg and sigma, compute (x_avg * w) * sigma + b
                end with a single rescale + relinearize
            */ 
            he->evaluator->add_plain_inplace(cts_x_flat[i], pts_x_flat_avg[i]);
            he->evaluator->add_plain_inplace(cts_sigma_flat[i], pts_sigma_flat[i]);
            
            // encode weight and bias
            vector<double> w_matrix(slot_count),b_matrix(slot_count);
            Plaintext tmp_w,tmp_b;
            for(int j=0;j<slot_count;j++){
                w_matrix[j] = (int64_t)w.data[i*slot_count + j];
                b_matrix[j] = (int64_t)b.data[i*slot_count + j];
            }
            // cout << "cts_x_flat.scale(): " << cts_x_flat[i].scale() << endl;
            he->encoder->encode(w_matrix, cts_x_flat[i].parms_id(), pow(2, he->he_scale), tmp_w);
            he->evaluator->multiply_plain_inplace(cts_x_flat[i], tmp_w);
            he->evaluator->multiply_inplace(cts_x_flat[i], cts_sigma_flat[i]);
            // cout << "cts_x_flat.scale(): " << cts_x_flat[i].scale() << endl;
            he->encoder->encode(b_matrix, cts_x_flat[i].parms_id(), cts_x_flat[i].scale(), tmp_b);
            he->evaluator->add_plain_inplace(cts_x_flat[i], tmp_b);
            he->evaluator->rescale_to_next_inplace(cts_x_flat[i]);
            he->evaluator->relinearize_inplace(cts_x_flat[i], *he->relin_keys);
            
            // cout << "cts_x_flat.scale(): " << cts_x_flat[i].scale() << endl;
        }
        cout << "scale:" << log2(cts_x_flat[0].scale()) << endl;
        cout << "(x_avg * sigma) * w + b done" << endl;  
        vector<Ciphertext> q_k_v = linear.linear_1(
            he,
            cts_x_flat,
            linear.pp_1[0],
            data
        );

        #pragma omp parallel for
        for(int i=0;i<q_k_v.size();i++){
            he->evaluator->rescale_to_next_inplace(q_k_v[i]);
        }
        /*
            before rescale, ct scale=145, Q=(60,55,60)
            after rescale, ct scale=30, Q=(60,)
        */
        cout << "Q:" << log2(he->get_Q(q_k_v[0])) << endl;
        cout << "scale:" << log2(q_k_v[0].scale()) << endl;
        PRG128 prg;
        /*
            生成r并计算x-r，这一过程极为复杂，暂时令r=0
        */
        vector<uint64_t> r(q_k_v.size()*he->poly_modulus_degree,0);
        // prg.random_mod_p<uint64_t>(r.data(), dim, linear.he->get_Q());
        cout << "num of output cts:" << q_k_v.size() << endl;
        cout << "bitwidth of output ct:" << q_k_v[0].coeff_modulus_size() << endl;
        send_encrypted_vector(io, q_k_v);
        for(int i=0;i<q_k_v.size();i++){
            vector<double> pod_matrix(degree/2, 0.0);
            Plaintext tmp;
            he->encoder->encode(pod_matrix, q_k_v[i].parms_id(), q_k_v[i].scale(), tmp);
            output_pts[i] = tmp;
        }
    }
    this->ct_transfer_comm += this->get_comm();
    this->ct_transfer_round += this->get_round();
    // cout << "send and receive comm:" << this->get_comm() << endl;
    // cout << "send and receive round:" << this->get_round() << endl;
    // he->print_parameters(he->context->get_context_data(output_pts[0].parms_id())->parms());
    vector<vector<HE_CKKS::scalar_t>> output_vector;
    output_vector = ckks_to_mpc(output_pts, degree, he->get_Q(output_pts[0]), bitwidth, s, he);
    // cout << "ckks_to_mpc comm:" << this->get_comm() << endl;
    // cout << "ckks_to_mpc round:" << this->get_round() << endl;
    this->conversion_comm += this->get_comm();
    this->conversion_round += this->get_round();
    return output_vector;
}

/*
    scale已经搞定了，差一个matmul_cc到matmul_cp的packing
*/
std::map<char*,FixArray> Bert_ckks::Softmax_v_linear_layer_norm_part1(FixArray input, int layer_id){
    FCMetadata data = linear.data_lin1_0;
    int bitwidth,s;
    bitwidth = input.ell;
    s = input.s;
    int softmax_dim = data.image_size;
    int softmax_output_size = PACKING_NUM*softmax_dim*softmax_dim;
    uint64_t* softmax_output_pack = new uint64_t[softmax_output_size];
    linear.preprocess_softmax(
        input.data,
        softmax_output_pack,
        data
    );
    cout << "OK2" << endl;
    HE_CKKS* he = linear.he2;
    int poly_modulus_degree = he->poly_modulus_degree;
    int slot = poly_modulus_degree / 2;
    int num_cts_softmax = softmax_output_size / slot;
    int num_cts_ln = data.image_size * COMMON_DIM / slot;
    vector<vector<HE_CKKS::scalar_t>> softmax_output_pack_vector(num_cts_softmax);
    vector<Plaintext> softmax_output_pack_pts;
    for(int i=0;i<num_cts_softmax;i++){
        for(int j=0;j<slot;j++){
            softmax_output_pack_vector[i].push_back(softmax_output_pack[i*slot+j]);
        }
    }
    softmax_output_pack_pts = mpc_to_ckks(softmax_output_pack_vector, slot, he->get_Q(), bitwidth, s, he->he_scale, he);
    this->conversion_comm += this->get_comm();
    this->conversion_round += this->get_round();
    vector<Plaintext> x_avg_pts(num_cts_ln);
    vector<Plaintext> sigma_pts(1);
    vector<Ciphertext> enc_x_avg(num_cts_ln);
    vector<Ciphertext> enc_sigma(1);
    vector<Ciphertext> enc_softmax(num_cts_softmax);
    vector<Ciphertext> enc_v(num_cts_ln);
    vector<Ciphertext> enc_softmax_half(num_cts_softmax/2);
    vector<Ciphertext> enc_v_half(num_cts_ln/2);
    // simulate x+y*j optimization, reduce input comm by half, send half_cts instead of cts
    if (party==BOB){
        for(int i=0;i<num_cts_softmax;i++){
            Ciphertext ct;
            he->encryptor->encrypt(softmax_output_pack_pts[i], ct);
            enc_softmax[i] = ct;
            if(i<num_cts_softmax/2){
                enc_softmax_half[i] = ct;
            }
        }
        for(int i=0;i<num_cts_ln;i++){
            vector<double> fake_vec(slot, 1.0);
            Plaintext pt;
            Ciphertext ct;
            he->encoder->encode(fake_vec, pow(2,he->he_scale), pt);
            he->encryptor->encrypt(pt, ct);
            enc_v[i] = ct;
            if(i<num_cts_ln/2){
                enc_v_half[i] = ct;
            }
        }
        send_encrypted_vector(io, enc_softmax_half);
        send_encrypted_vector(io, enc_v_half);
        cout << "send_encrypted_vector done" << endl;
        //TODO: recv_encrypted_vector
        recv_encrypted_vector(he->context, io, enc_x_avg);
        recv_encrypted_vector(he->context, io, enc_sigma);
        for(int i=0;i<enc_x_avg.size();i++){
            he->decryptor->decrypt(enc_x_avg[i], x_avg_pts[i]);
        }
        he->decryptor->decrypt(enc_sigma[0], sigma_pts[0]);
    }
    else{
        recv_encrypted_vector(he->context, io, enc_softmax_half);
        recv_encrypted_vector(he->context, io, enc_v_half);
        for(int i=0;i<enc_softmax_half.size();i++){
            seal::Ciphertext ct1;
            seal::Ciphertext tmp;
            // x + y*j => x - y*j
            he->evaluator->complex_conjugate(enc_softmax_half[i], *he->gal_keys, tmp);
            // x + y*j - (x - y*j)*
            // - 2 * y*j
            he->evaluator->sub(tmp, enc_softmax_half[i], ct1);
            // - 2 * y*j => 2*y
            MulImageUnitInplace(ct1, *he->context);
            // x + y*j + (x - y*j) => 2*x
            he->evaluator->add_inplace(enc_softmax_half[i], tmp);
            enc_softmax[i] = enc_softmax_half[i];
            enc_softmax[i+enc_softmax_half.size()] = ct1;
        }
        for(int i=0;i<enc_v_half.size();i++){
            seal::Ciphertext ct1;
            seal::Ciphertext tmp;
            // x + y*j => x - y*j
            he->evaluator->complex_conjugate(enc_v_half[i], *he->gal_keys, tmp);
            // x + y*j - (x - y*j)*
            // - 2 * y*j
            he->evaluator->sub(tmp, enc_v_half[i], ct1);
            // - 2 * y*j => 2*y
            MulImageUnitInplace(ct1, *he->context);
            // x + y*j + (x - y*j) => 2*x
            he->evaluator->add_inplace(enc_v_half[i], tmp);
            enc_v[i] = enc_v_half[i];
            enc_v[i+enc_v_half.size()] = ct1;
        }
        cout << "num of input cts:" << enc_softmax.size() << endl;
        cout << "recv_encrypted_vector" << endl;
        #pragma omp parallel for num_threads(4)
        for(int i=0;i<enc_softmax.size();i++){
            he->evaluator->add_plain_inplace(enc_softmax[i], softmax_output_pack_pts[i]);
        }
        cout << "OK3" << endl;
        auto soft_mask_ct = linear.softmax_mask_ct_ct(he, data);
        cout << "OK4" << endl;
        auto pack_softmax_ct = linear.preprocess_softmax_s1_ct_ct(he, enc_softmax, data, soft_mask_ct);
        cout << "OK5" << endl;
        vector<Ciphertext> softmax_V_result(12 * data.image_size * data.filter_w / data.slot_count);
        linear.softmax_v(he, pack_softmax_ct, enc_v, data, softmax_V_result);
        // softmax_V_result's pack may be changed
        data = linear.data_lin2;
        cout << "OK6" << endl;
        vector<Ciphertext> h3 = linear.linear_2(
            he,
            softmax_V_result, 
            linear.pp_2[layer_id],
            data
        );
        cout << "OK7" << endl;
        // 模拟residual, h3 scale = 3s-60, Q = [60, 40, 60, 60, 60], h3 is the input for layernorm
        for(int i=0;i<h3.size();i++){
            Ciphertext tmp = Ciphertext(h3[i]);
            he->evaluator->add_inplace(h3[i], tmp);
            he->evaluator->rescale_to_next_inplace(h3[i]);
            he->evaluator->relinearize_inplace(h3[i], *he->relin_keys);
        }
        cout << "OK1" << endl;
        // 开始计算layernorm
        // 1. X_sum
        int L = data.image_size;
        vector<Ciphertext> X_avg;
        vector<Ciphertext> sigma;
        for(int i=0;i<h3.size();i++){
            Ciphertext tmp = Ciphertext(h3[i]);
            X_avg.push_back(tmp);
        }
        for(int i=1;i<=std::log2(slot/L);i++){
            #pragma omp parallel for num_threads(4)
            for(int j=0;j<h3.size();j++){
                Ciphertext tmp;
                he->evaluator->rotate_vector(X_avg[j],pow(2,i-1)*L,*he->gal_keys,tmp);
                he->evaluator->add_inplace(X_avg[j],tmp);
            }
        }
        cout << "OK1" << endl;
        Ciphertext X_sum = Ciphertext(X_avg[0]);
        for(int i=1;i<X_avg.size();i++){
            he->evaluator->add_inplace(X_sum,X_avg[i]);
        }
        #pragma omp parallel for num_threads(4)
        for(int i=0;i<X_avg.size();i++){
            X_avg[i] = Ciphertext(X_sum);
        }
        cout << "OK2" << endl;
        // 2. X_avg, scale=4s-60
        vector<double> inv_D(slot, 1.0/768);
        Plaintext inv_D_p;
        he->encoder->encode(inv_D, X_avg[0].parms_id(), pow(2,he->he_scale), inv_D_p);
        for(int i=0;i<X_avg.size();i++){
            he->evaluator->multiply_plain_inplace(X_avg[i],inv_D_p);
        }
        cout << "OK3" << endl;
        // 3. scale up h3 (that is, x) from 3s-60 to 4s-60, thus h3 can sub X_avg
        Plaintext scale_up;
        vector<double> all_1(slot, 1);
        he->encoder->encode(all_1, h3[0].parms_id(), pow(2,he->he_scale), scale_up);
        /*
           4. sigma = h3 = h3-X_avg
           do 1*rs; then scale = 4s-60*2=52, Q = [60, 40, 60, 60]
        */ 
       cout << "h3.scale:" << log2(h3[0].scale()) << endl;
       cout << "X_avg.scale:" << log2(X_avg[0].scale()) << endl;
        for(int i=0;i<h3.size();i++){
            he->evaluator->multiply_plain_inplace(h3[i],scale_up);
            he->evaluator->sub_inplace(h3[i],X_avg[i]);
            he->evaluator->rescale_to_next_inplace(h3[i]);
        }
        cout << "OK4" << endl;
        /*
            compute X_sum2=sigma^2, 注意sigma^2维度需要mask成(L,1),为了降低后面sqrt维度
            同时把h3转化为 scale: 12, Q:60 by 2*ms, 1*rs
            sigma scale = 52^2-60, Q = [60, 40, 60]
        */ 
        cout << "h3.scale:" << log2(h3[0].scale()) << endl;
        for(int i=0;i<h3.size();i++){
            Ciphertext tmp = Ciphertext(h3[i]);
            sigma.push_back(tmp);
            he->evaluator->square_inplace(sigma[i]);
            he->evaluator->relinearize_inplace(sigma[i], *he->relin_keys);
            he->evaluator->rescale_to_next_inplace(sigma[i]);
            he->evaluator->mod_switch_to_next_inplace(h3[i]);
            he->evaluator->mod_switch_to_next_inplace(h3[i]);
            he->evaluator->rescale_to_next_inplace(h3[i]);
        }
        cout << "OK5" << endl;
        for(int i=1;i<=std::log2(slot/L);i++){
            #pragma omp parallel for num_threads(4)
            for(int j=0;j<sigma.size();j++){
                Ciphertext tmp;
                he->evaluator->rotate_vector(sigma[j],pow(2,i-1)*L,*he->gal_keys,tmp);
                he->evaluator->add_inplace(sigma[j],tmp);
            }
        }
        cout << "OK5" << endl;
        Ciphertext X_sum2 = Ciphertext(sigma[0]);
        for(int i=1;i<sigma.size();i++){
            he->evaluator->add_inplace(X_sum2,sigma[i]);
        }
        vector<double> mask(slot,0);
        for(int i=0;i<L;i++){
            mask[i] = 1;
        }
        cout << "OK5" << endl;
        Plaintext mask_p;
        he->encoder->encode(mask, X_sum2.parms_id(), pow(2,he->he_scale), mask_p);
        he->evaluator->multiply_plain_inplace(X_sum2,mask_p);
        vector<double> inv_D2(slot, 1.0/768);
        Plaintext inv_D_p2;
        he->encoder->encode(inv_D, X_sum2.parms_id(), pow(2,he->he_scale), inv_D_p2);
        he->evaluator->multiply_plain_inplace(X_sum2,inv_D_p2);
        //before scale=130, Q=[60, 40, 60]; after 2*rs scale = 30, Q = 60
        he->evaluator->rescale_to_next_inplace(X_sum2);
        he->evaluator->rescale_to_next_inplace(X_sum2);
        PRG128 prg;
        /*
            生成r并计算x-r，这一过程极为复杂，暂时令r=0
        */
        cout << "num of output cts:" << sigma.size()+1 << endl;
        cout << "bitwidth of h3 ct:" << h3[0].coeff_modulus_size() << endl;
        cout << "bitwidth of X_sum2 ct:" << X_sum2.coeff_modulus_size() << endl;
        vector<Ciphertext> enc_sigma;
        enc_sigma.push_back(X_sum2);
        cout << "h3.scale:" << log2(h3[0].scale()) << endl;
        cout << "enc_sigma.scale:" << log2(enc_sigma[0].scale()) << endl;
        send_encrypted_vector(io, h3);
        send_encrypted_vector(io, enc_sigma);
        Plaintext tmp;
        vector<double> pod_matrix(poly_modulus_degree/2, 0.0);
        for(int i=0;i<h3.size();i++){
            he->encoder->encode(pod_matrix, h3[i].parms_id(), h3[i].scale(), tmp);
            x_avg_pts[i] = tmp;
        }
        he->encoder->encode(pod_matrix, X_sum2.parms_id(), X_sum2.scale(), tmp);
        sigma_pts[0] = tmp;
    }
    this->ct_transfer_comm += this->get_comm();
    this->ct_transfer_round += this->get_round();
    // cout << "send and receive cts, comm:" << this->get_comm() << endl;
    // cout << "send and receive cts, round:" << this->get_round() << endl;
    cout << "send and receive done in linear4_add_layernorm_part1" << endl;
    vector<vector<HE_CKKS::scalar_t>> vec_x_avg(x_avg_pts.size());
    vector<vector<HE_CKKS::scalar_t>> vec_sigma(sigma_pts.size());
    vec_x_avg = ckks_to_mpc(x_avg_pts, poly_modulus_degree, he->get_Q(x_avg_pts[0]), bitwidth, s, he);
    vec_sigma = ckks_to_mpc(sigma_pts, poly_modulus_degree, he->get_Q(sigma_pts[0]), bitwidth, s, he);
    cout << "vec_sigma.size():" << vec_sigma.size() << ", should be 1" << endl;
    this->conversion_comm += this->get_comm();
    this->conversion_round += this->get_round();
    // cout << "mpc_to_ckks comm:" << this->get_comm() << endl;
    // cout << "mpc_to_ckks round:" << this->get_round() << endl;
    cout << "bitwidth,s:" << bitwidth << "," << s << endl;
    FixArray x_flat_avg, sigma_flat;
    FPMath *fpmath = nonlinear.fpmath[0];
    x_flat_avg = fpmath->fix->input(party, num_cts_ln*slot,1, true, bitwidth, s);
    sigma_flat = fpmath->fix->input(party, data.image_size,1, true, bitwidth, s);
    for(int i=0;i<vec_x_avg.size();i++){
        for(int j=0;j<vec_x_avg[i].size();j++){
            x_flat_avg.data[i*vec_x_avg[i].size()+j] = vec_x_avg[i][j];
        }
    }
    for(int j=0;j<data.image_size;j++){
        sigma_flat.data[j] = vec_sigma[0][j];
    }
    map<char*,FixArray> result;
    result["x_flat_avg"] = x_flat_avg;
    result["sigma_flat"] = sigma_flat;
    return result;
}

/*
    do not support multiple threads
    num_ops is sequence*dim
*/ 
// std::map<char*,FixArray> Bert_ckks::gelu_part1(uint64_t *x, int num_ops, int bitwidth, int s){
//     FPMath *fpmath = nonlinear.fpmath[0];
//     map<char*,FixArray> result;
//     int this_party = party;
//     int N = num_ops;
//     FixArray x_array;
//     x_array = fpmath->fix->input(this_party, num_ops, x, true, bitwidth, s);
//     vector<Plaintext> gelu_poly_pts = eval_gelu_poly_old(x_array);
//     vector<vector<HE_CKKS::scalar_t>> gelu_poly_mpc = ckks_to_mpc(gelu_poly_pts, poly_modulus_degree, linear.he->get_Q(gelu_poly_pts[0]), bitwidth, s);
//     FixArray gelu_poly_flat(this_party, num_ops, true, bitwidth, s);
//     for(int i=0;i<gelu_poly_mpc.size();i++){
//         for(int j=0;j<gelu_poly_mpc[i].size();j++){
//             gelu_poly_flat.data[i*gelu_poly_mpc[i].size()+j] = gelu_poly_mpc[i][j];
//         }
//     }
//     result["gelu_poly_flat"] = gelu_poly_flat;
//     result["x_array"] = x_array;
//     return result;
// }

/*
    input 的 bitwidth 与 scale 需要是正确的MPC下的值
*/
std::map<char*,FixArray> Bert_ckks::ln_part2_linear3_gelu_part1(std::map<char*,FixArray> input){
    // multi depth = 6
    
    FixArray x_flat_avg = input["x_flat_avg"];
    FixArray sigma_flat = input["sigma_flat"];
    FixArray w = input["w"];
    FixArray b = input["b"];
    FPMath *fpmath = nonlinear.fpmath[0];
    map<char*,FixArray> result;
    int bitwidth = x_flat_avg.ell;
    int s = x_flat_avg.s;
    cout << "in ln_part2_linear3_gelu_part1, bitwidth,s:" << bitwidth << "," << s << endl;
    cout << "should be 0! comm:" << this->get_comm() << ", round:" << this->get_round() << endl;
    HE_CKKS* he = linear.he3;
    uint64_t dim = x_flat_avg.size;
    uint64_t poly_modulus_degree = he->poly_modulus_degree;
    uint64_t slot_count = poly_modulus_degree / 2;
    uint64_t num_cts = dim / slot_count;
    FixArray gelu_poly_flat, x_array;

    vector<vector<HE_CKKS::scalar_t>> vec_x_flat_avg(num_cts), vec_sigma_flat(num_cts);
    for(int i = 0; i < num_cts; i++){
        for(int j = 0; j < slot_count; j++){
            vec_x_flat_avg[i].push_back(x_flat_avg.data[i*slot_count + j]);
            vec_sigma_flat[i].push_back(sigma_flat.data[i*slot_count + j]);
        }
    }
    vector<Plaintext> pts_x_flat_avg, pts_sigma_flat;
    
    FCMetadata data = linear.data_lin3;
    int gelu_input_size = data.image_size*3072;
    int gelu_cts_size = gelu_input_size / slot_count;
    vector<Ciphertext> output_cts(
        gelu_cts_size
    );
    vector<Plaintext> output_pts(
        gelu_cts_size
    );
    cout << "gelu_cts_size:" << gelu_cts_size << endl;
    pts_x_flat_avg = mpc_to_ckks(vec_x_flat_avg, slot_count, linear.he3->get_Q(), bitwidth, s, linear.he3->he_scale, linear.he3);
    pts_sigma_flat = mpc_to_ckks(vec_sigma_flat, slot_count, linear.he3->get_Q(), bitwidth, s, linear.he3->he_scale, linear.he3);
    // cout << "mpc_to_ckks comm:" << this->get_comm() << endl;
    // cout << "mpc_to_ckks round:" << this->get_round() << endl;
    this->conversion_comm += this->get_comm();
    this->conversion_round += this->get_round();
    vector<Ciphertext> cts_x_flat(pts_x_flat_avg.size()), cts_sigma_flat(pts_sigma_flat.size());
    
    // simulate x+y*j optimization, reduce input comm by half
    // send half_cts instead of cts
    vector<Ciphertext> cts_x_flat_half(cts_x_flat.size()/2);
    vector<Ciphertext> cts_sigma_flat_half(cts_sigma_flat.size()/2);
    if (party == BOB) {
        for(size_t i=0;i<num_cts;i++){
            Ciphertext ct1,ct2;
            he->encryptor->encrypt(pts_x_flat_avg[i], ct1);
            he->encryptor->encrypt(pts_sigma_flat[i], ct2);
            cts_x_flat[i] = ct1;
            cts_sigma_flat[i] = ct2;
            if(i<num_cts/2){
                cts_x_flat_half[i] = cts_x_flat[i];
                cts_sigma_flat_half[i] = cts_sigma_flat[i];
            }
        }
        cout << "send_encrypted_vector" << endl;
		send_encrypted_vector(io, cts_x_flat_half);
        send_encrypted_vector(io, cts_sigma_flat_half);
        cout << "send_encrypted_vector done" << endl;
		// send_encrypted_vector(io, cts);
		// vector<Ciphertext> share_client(dim);
		recv_encrypted_vector(he->context, io, output_cts);
        for(int i=0;i<output_cts.size();i++){
            he->decryptor->decrypt(output_cts[i], output_pts[i]);
        }
	}
    else{
        recv_encrypted_vector(he->context, io, cts_x_flat_half);
        recv_encrypted_vector(he->context, io, cts_sigma_flat_half);
        for(int i=0;i<cts_x_flat_half.size();i++){
            seal::Ciphertext ct1,ct2;
            seal::Ciphertext tmp,tmp2;
            // x + y*j => x - y*j
            he->evaluator->complex_conjugate(cts_x_flat_half[i], *he->gal_keys, tmp);
            // x + y*j - (x - y*j)*
            // - 2 * y*j
            he->evaluator->sub(tmp, cts_x_flat_half[i], ct1);
            // - 2 * y*j => 2*y
            MulImageUnitInplace(ct1, *he->context);
            // x + y*j + (x - y*j) => 2*x
            he->evaluator->add_inplace(cts_x_flat_half[i], tmp);
            cts_x_flat[i] = cts_x_flat_half[i];
            cts_x_flat[i+cts_x_flat_half.size()] = ct1;

            he->evaluator->complex_conjugate(cts_sigma_flat_half[i], *he->gal_keys, tmp2);
            he->evaluator->sub(tmp2, cts_sigma_flat_half[i], ct2);
            MulImageUnitInplace(ct2, *he->context);
            he->evaluator->add_inplace(cts_sigma_flat_half[i], tmp2);
            cts_sigma_flat[i] = cts_sigma_flat_half[i];
            cts_sigma_flat[i+cts_sigma_flat_half.size()] = ct2;
        }
        cout << "num of input cts:" << cts_x_flat.size()+cts_sigma_flat.size() << endl;
        cout << "bitwidth of input ct:" << cts_x_flat[0].coeff_modulus_size() << endl;
        cout << "recv_encrypted_vector" << endl;
        // cout << "cts_x_flat[0].scale()" << cts_x_flat[0].scale() << endl;
        // get x_avg and sigma, compute (x_avg * sigma) * w + b
        for(int i=0;i<cts_x_flat.size();i++){
            he->evaluator->add_plain_inplace(cts_x_flat[i], pts_x_flat_avg[i]);
            he->evaluator->add_plain_inplace(cts_sigma_flat[i], pts_sigma_flat[i]);
            
            // encode weight and bias
            vector<double> w_matrix(slot_count),b_matrix(slot_count);
            Plaintext tmp_w,tmp_b;
            for(int j=0;j<slot_count;j++){
                w_matrix[j] = (int64_t)w.data[i*slot_count + j];
                b_matrix[j] = (int64_t)b.data[i*slot_count + j];
            }
            // cout << "cts_x_flat.scale(): " << cts_x_flat[i].scale() << endl;
            he->encoder->encode(w_matrix, cts_x_flat[i].parms_id(), cts_x_flat[i].scale(), tmp_w);
            he->evaluator->multiply_plain_inplace(cts_x_flat[i], tmp_w);
            he->evaluator->multiply_inplace(cts_x_flat[i], cts_sigma_flat[i]);
            // cout << "cts_x_flat.scale(): " << cts_x_flat[i].scale() << endl;
            he->encoder->encode(b_matrix, cts_x_flat[i].parms_id(), cts_x_flat[i].scale(), tmp_b);
            he->evaluator->add_plain_inplace(cts_x_flat[i], tmp_b);
            he->evaluator->rescale_to_next_inplace(cts_x_flat[i]);
            he->evaluator->relinearize_inplace(cts_x_flat[i], *he->relin_keys);
        }
        
        cout << "(x_avg * sigma) * w + b done" << endl;  
        cout << "Q, scale:" << log2(he->get_Q(cts_x_flat[0])) << "," << log2(cts_x_flat[0].scale()) << endl;
        /*
            计算linear3
        */
        FCMetadata data = linear.data_lin3;
        vector<Ciphertext> linear3_output = linear.linear_2(
            he,
            cts_x_flat, 
            linear.pp_3[0],
            data
        );
        cout << "linear2 done" << endl;
        cout << "scale:" << log2(linear3_output[0].scale()) << endl;
        for(int i=0;i<linear3_output.size();i++){
            he->evaluator->rescale_to_next_inplace(linear3_output[i]);
            he->evaluator->relinearize_inplace(linear3_output[i], *he->relin_keys);
        }
        cout << "scale:" << log2(linear3_output[0].scale()) << endl;
        /*
            before: scale 64, Q= (60, 32, 60, 60, 60)
        */
        vector<Ciphertext> gelu_output = eval_gelu_poly(linear3_output, he);
        PRG128 prg;
        /*
            生成r并计算x-r，这一过程极为复杂，暂时令r=0
        */
        // prg.random_mod_p<uint64_t>(r.data(), dim, linear.he->get_Q());
        cout << "num of output cts:" << gelu_output.size() << endl;
        cout << "bitwidth of output ct:" << gelu_output[0].coeff_modulus_size() << endl;
        send_encrypted_vector(io, gelu_output);
        cout << "linear3_output.size():" << linear3_output.size() << endl;
        cout << "output_pts.size():" << output_pts.size() << endl;
        for(int i=0;i<gelu_output.size();i++){
            vector<double> pod_matrix(poly_modulus_degree/2, 0.0);
            Plaintext tmp;
            he->encoder->encode(pod_matrix, gelu_output[i].parms_id(), gelu_output[i].scale(), tmp);
            output_pts[i] = tmp;
        }
    }

    this->ct_transfer_comm += this->get_comm();
    this->ct_transfer_round += this->get_round();
    cout << "send and receive done in ln_part2_linear3_gelu_part1" << endl;
    vector<vector<HE_CKKS::scalar_t>> vec_output(output_pts.size());
    vec_output = ckks_to_mpc(output_pts, poly_modulus_degree, he->get_Q(output_pts[0]), bitwidth, s, he);
    this->conversion_comm += this->get_comm();
    this->conversion_round += this->get_round();
    // cout << "mpc_to_ckks comm:" << this->get_comm() << endl;
    // cout << "mpc_to_ckks round:" << this->get_round() << endl;
    cout << "bitwidth,s:" << bitwidth << "," << s << endl;
    gelu_poly_flat = fpmath->fix->input(party, gelu_input_size, true, bitwidth, s);
    x_array = fpmath->fix->input(party, gelu_input_size,1, true, bitwidth, s);
    cout << "gelu_input_size:" << gelu_input_size << endl;
    cout << "vec_output.size()*vec_output[0].size():" << vec_output.size()*vec_output[0].size() << endl;
    for(int i=0;i<vec_output.size();i++){
        for(int j=0;j<vec_output[i].size();j++){
            gelu_poly_flat.data[i*vec_output[i].size()+j] = vec_output[i][j];
        }
    }
    gelu_poly_flat.ell = bitwidth;
    gelu_poly_flat.s = s;
    cout << "gelu_poly_flat, bitwidth,s:" << gelu_poly_flat.ell << "," << gelu_poly_flat.s << endl;
    result["gelu_poly_flat"] = gelu_poly_flat;
    result["x_array"] = x_array;
    cout << "ln_part2_linear3_gelu_part1" << endl;
    return result;
}


/*
    input 维度是 (L,3072)，linear4后降维至 (L,768)
*/
std::map<char*,FixArray> Bert_ckks::linear4_add_layernorm_part1(FixArray input, int layer_id){
    cout << "in linear4_add_layernorm_part1" << endl;
    FCMetadata data = linear.data_lin4;
    HE_CKKS* he = linear.he4;
    int bitwidth,s;
    bitwidth = input.ell;
    s = input.s;
    int gelu_input_size = data.image_size*COMMON_DIM*4;
    int poly_modulus_degree = he->poly_modulus_degree;
    int slot = poly_modulus_degree / 2;
    int num_cts_gelu = gelu_input_size / slot;
    int num_cts_ln = data.image_size*COMMON_DIM / slot;
    vector<vector<HE_CKKS::scalar_t>> h6(num_cts_gelu);
    vector<Plaintext> h6_pts;
    cout << num_cts_gelu*slot << endl;
    for(int i=0;i<num_cts_gelu;i++){
        for(int j=0;j<slot;j++){
            h6[i].push_back(input.data[i*slot+j]);
        }
    }
    h6_pts = mpc_to_ckks(h6, slot, he->get_Q(), bitwidth, s, he->he_scale, he);
    this->conversion_comm += this->get_comm();
    this->conversion_round += this->get_round();
    vector<Ciphertext> enc_h6(num_cts_gelu);
    vector<Plaintext> x_avg_pts(num_cts_ln);
    vector<Plaintext> sigma_pts(1);
    vector<Ciphertext> enc_x_avg(num_cts_ln);
    vector<Ciphertext> enc_sigma(1);
    vector<Ciphertext> enc_h6_half(num_cts_gelu/2);
    if (party==BOB){
        for(int i=0;i<num_cts_gelu;i++){
            Ciphertext ct;
            he->encryptor->encrypt(h6_pts[i], ct);
            enc_h6[i] = ct;
            if(i<num_cts_gelu/2){
                enc_h6_half[i] = ct;
            }
        }
        send_encrypted_vector(io, enc_h6_half);
        cout << "send_encrypted_vector done" << endl;
        //TODO: recv_encrypted_vector
        recv_encrypted_vector(he->context, io, enc_x_avg);
        recv_encrypted_vector(he->context, io, enc_sigma);
        for(int i=0;i<enc_x_avg.size();i++){
            he->decryptor->decrypt(enc_x_avg[i], x_avg_pts[i]);
        }
        he->decryptor->decrypt(enc_sigma[0], sigma_pts[0]);
    }
    else{
        recv_encrypted_vector(he->context, io, enc_h6_half);
        for(int i=0;i<enc_h6_half.size();i++){
            seal::Ciphertext ct1;
            seal::Ciphertext tmp;
            // x + y*j => x - y*j
            he->evaluator->complex_conjugate(enc_h6_half[i], *he->gal_keys, tmp);
            // x + y*j - (x - y*j)*
            // - 2 * y*j
            he->evaluator->sub(tmp, enc_h6_half[i], ct1);
            // - 2 * y*j => 2*y
            MulImageUnitInplace(ct1, *he->context);
            // x + y*j + (x - y*j) => 2*x
            he->evaluator->add_inplace(enc_h6_half[i], tmp);
            enc_h6[i] = enc_h6_half[i];
            enc_h6[i+enc_h6_half.size()] = ct1;
        }
        cout << "num of input cts:" << enc_h6.size() << endl;
        cout << "recv_encrypted_vector" << endl;
        #pragma omp parallel for num_threads(4)
        for(int i=0;i<enc_h6.size();i++){
            he->evaluator->add_plain_inplace(enc_h6[i], h6_pts[i]);
        }
        vector<Ciphertext> h7 = linear.linear_2(
            he,
            enc_h6, 
            linear.pp_4[layer_id],
            data
        );
        cout << "h7.size():" << h7.size() << ", should be:" << num_cts_ln << endl;
        cout << "scale:" << log2(h7[0].scale()) << endl;
        he->print_parameters(he->context->get_context_data(h7[0].parms_id())->parms());
        //模拟residual
        vector<Ciphertext> h4_cache_12;
        for(int i=0;i<h7.size();i++){
            Ciphertext tmp = Ciphertext(h7[i]);
            h4_cache_12.push_back(tmp);
            he->evaluator->add_inplace(h7[i], tmp);
        }
        cout << "OK1" << endl;
        // 开始计算layernorm
        // 1. X_sum
        int L = data.image_size;
        vector<Ciphertext> X_avg;
        vector<Ciphertext> sigma;
        for(int i=0;i<h7.size();i++){
            Ciphertext tmp = Ciphertext(h7[i]);
            X_avg.push_back(tmp);
        }
        for(int i=1;i<=std::log2(slot/L);i++){
            #pragma omp parallel for num_threads(4)
            for(int j=0;j<h7.size();j++){
                Ciphertext tmp;
                he->evaluator->rotate_vector(X_avg[j],pow(2,i-1)*L,*he->gal_keys,tmp);
                he->evaluator->add_inplace(X_avg[j],tmp);
            }
        }
        cout << "OK1" << endl;
        Ciphertext X_sum = Ciphertext(X_avg[0]);
        for(int i=1;i<num_cts_ln;i++){
            he->evaluator->add_inplace(X_sum,X_avg[i]);
        }
        #pragma omp parallel for num_threads(4)
        for(int i=0;i<num_cts_ln;i++){
            X_avg[i] = Ciphertext(X_sum);
        }
        cout << "OK2" << endl;
        // 2. X_avg, scale=s*3
        vector<double> inv_D(slot, 1.0/768);
        Plaintext inv_D_p;
        he->encoder->encode(inv_D, X_avg[0].parms_id(), pow(2,he->he_scale), inv_D_p);
        for(int i=0;i<num_cts_ln;i++){
            he->evaluator->multiply_plain_inplace(X_avg[i],inv_D_p);
        }
        cout << "OK3" << endl;
        // 3. scale up h7 from s*2 to s*3
        Plaintext scale_up;
        vector<double> all_1(slot, 1);
        he->encoder->encode(all_1, h7[0].parms_id(), pow(2,he->he_scale), scale_up);
        /*
           4. h7 = h7-X_avg
           scale = 3*s-60, Q=[60, 34, 60, 60]
        */ 
       cout << "h7.scale:" << log2(h7[0].scale()) << endl;
       cout << "X_avg.scale:" << log2(X_avg[0].scale()) << endl;
        for(int i=0;i<num_cts_ln;i++){
            he->evaluator->multiply_plain_inplace(h7[i],scale_up);
            he->evaluator->sub_inplace(h7[i],X_avg[i]);
            he->evaluator->rescale_to_next_inplace(h7[i]);
        }
        cout << "OK4" << endl;
        /*
            compute X_sum2=sigma^2, 注意sigma^2维度需要mask成(L,1)，为了降低后面sqrt维度
            同时将h7 转化为 scale: 20, Q: 60 by 2*ms, 1*rs
        */ 
        cout << "h7.scale:" << log2(h7[0].scale()) << endl;
        for(int i=0;i<num_cts_ln;i++){
            Ciphertext tmp = Ciphertext(h7[i]);
            sigma.push_back(tmp);
            he->evaluator->square_inplace(sigma[i]);
            he->evaluator->relinearize_inplace(sigma[i], *he->relin_keys);
            he->evaluator->mod_switch_to_next_inplace(h7[i]);
            he->evaluator->mod_switch_to_next_inplace(h7[i]);
            he->evaluator->rescale_to_next_inplace(h7[i]);
        }
        cout << "OK5" << endl;
        for(int i=1;i<=std::log2(slot/L);i++){
            #pragma omp parallel for num_threads(4)
            for(int j=0;j<num_cts_ln;j++){
                Ciphertext tmp;
                he->evaluator->rotate_vector(sigma[j],pow(2,i-1)*L,*he->gal_keys,tmp);
                he->evaluator->add_inplace(sigma[j],tmp);
            }
        }
        cout << "OK5" << endl;
        Ciphertext X_sum2 = Ciphertext(sigma[0]);
        for(int i=1;i<num_cts_ln;i++){
            he->evaluator->add_inplace(X_sum2,sigma[i]);
        }
        vector<double> mask(slot,0);
        for(int i=0;i<L;i++){
            mask[i] = 1;
        }
        cout << "OK5" << endl;
        Plaintext mask_p;
        he->encoder->encode(mask, X_sum2.parms_id(), pow(2,he->he_scale), mask_p);
        he->evaluator->multiply_plain_inplace(X_sum2,mask_p);
        vector<double> inv_D2(slot, 1.0/768);
        Plaintext inv_D_p2;
        he->encoder->encode(inv_D, X_sum2.parms_id(), pow(2,he->he_scale), inv_D_p2);
        he->evaluator->multiply_plain_inplace(X_sum2,inv_D_p2);
        //before scale=120, Q=[60, 30, 60]; after 2*rs scale = 30, Q = 60
        he->evaluator->rescale_to_next_inplace(X_sum2);
        he->evaluator->rescale_to_next_inplace(X_sum2);
        // 6. send cts to client, and ckks to mpc
        PRG128 prg;
        /*
            生成r并计算x-r，这一过程极为复杂，暂时令r=0
        */
        cout << "num of output cts:" << sigma.size()+1 << endl;
        cout << "bitwidth of h7 ct:" << h7[0].coeff_modulus_size() << endl;
        cout << "bitwidth of X_sum2 ct:" << X_sum2.coeff_modulus_size() << endl;
        vector<Ciphertext> enc_sigma;
        enc_sigma.push_back(X_sum2);
        cout << "h7.scale:" << log2(h7[0].scale()) << endl;
        cout << "enc_sigma.scale:" << log2(enc_sigma[0].scale()) << endl;
        send_encrypted_vector(io, h7);
        send_encrypted_vector(io, enc_sigma);
        Plaintext tmp;
        vector<double> pod_matrix(poly_modulus_degree/2, 0.0);
        for(int i=0;i<num_cts_ln;i++){
            he->encoder->encode(pod_matrix, h7[i].parms_id(), h7[i].scale(), tmp);
            x_avg_pts[i] = tmp;
        }
        he->encoder->encode(pod_matrix, X_sum2.parms_id(), X_sum2.scale(), tmp);
        sigma_pts[0] = tmp;
    }
    // cout << "send and receive cts, comm:" << this->get_comm() << endl;
    // cout << "send and receive cts, round:" << this->get_round() << endl;
    this->ct_transfer_comm += this->get_comm();
    this->ct_transfer_round += this->get_round();
    cout << "send and receive done in linear4_add_layernorm_part1" << endl;
    vector<vector<HE_CKKS::scalar_t>> vec_x_avg(x_avg_pts.size());
    vector<vector<HE_CKKS::scalar_t>> vec_sigma(sigma_pts.size());
    vec_x_avg = ckks_to_mpc(x_avg_pts, poly_modulus_degree, he->get_Q(x_avg_pts[0]), bitwidth, s, he);
    vec_sigma = ckks_to_mpc(sigma_pts, poly_modulus_degree, he->get_Q(sigma_pts[0]), bitwidth, s, he);
    cout << "vec_sigma.size():" << vec_sigma.size() << ", should be 1" << endl;
    this->conversion_comm += this->get_comm();
    this->conversion_round += this->get_round();
    // cout << "mpc_to_ckks comm:" << this->get_comm() << endl;
    // cout << "mpc_to_ckks round:" << this->get_round() << endl;
    cout << "bitwidth,s:" << bitwidth << "," << s << endl;
    FixArray x_flat_avg, sigma_flat;
    FPMath *fpmath = nonlinear.fpmath[0];
    x_flat_avg = fpmath->fix->input(party, num_cts_ln*slot, true, bitwidth, s);
    sigma_flat = fpmath->fix->input(party, data.image_size, true, bitwidth, s);
    for(int i=0;i<vec_x_avg.size();i++){
        for(int j=0;j<vec_x_avg[i].size();j++){
            x_flat_avg.data[i*vec_x_avg[i].size()+j] = vec_x_avg[i][j];
        }
    }
    for(int j=0;j<data.image_size;j++){
        sigma_flat.data[j] = vec_sigma[0][j];
    }
    x_flat_avg.ell = bitwidth;
    x_flat_avg.s = s;
    sigma_flat.ell = bitwidth;
    sigma_flat.s = s;
    map<char*,FixArray> result;
    result["x_flat_avg"] = x_flat_avg;
    result["sigma_flat"] = sigma_flat;
    return result;
}
