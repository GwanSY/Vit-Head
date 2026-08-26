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
#include <fstream>
#include <iostream>
#include <thread>
#include <cmath>
#include <vector>
#include "ckks_bert.h"

using namespace sci;
using namespace std;
using namespace seal;

#define MAX_THREADS 12
// #define SCI_HE

int party = 0;
int port = 8000;
int num_threads = 1;
string address = "127.0.0.1";
int bitlength = 37;
int32_t dim = 768;
int32_t array_size = 128;
int32_t bw_x = 37;
int32_t bw_y = 37;
int32_t s_x = 12;
int32_t s_y = 12;
int32_t input_size = dim*array_size;
int32_t poly_modulus_degree=8192;

bool signed_ = true;

template<typename T> void write_to_file2(std::string filename, T data)
{
    // std::ofstream file(filename, std::ios::app);  // 使用std::ios::app模式打开文件，代表续写
    std::ofstream file(filename);
    // printf("data size:%ld\n",data.size());
    // fflush(stdout);
    for (size_t i = 0; i < data.size(); ++i)
    {
        file << data[i];
        if (i != data.size() - 1)  // 不在最后一个元素后添加逗号
        {
            file << ",";
        }
    }
    file.close();
}

template<typename T> std::vector<T> read_from_file(std::string filename)
{
    std::vector<T> data;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line, ','))
    {
        std::istringstream iss(line);
        T value;
        iss >> value;
        data.push_back(value);
    }
    file.close();
    return data;
}

void ifft_weight_test(int party, Bert_ckks bert_ckks){
  vector<HE_CKKS::complex_t> x(8192),y(8192);
  vector<HE_CKKS::scalar_t> y_real(8192),y_imag(8192);
  // bert_ckks.linear.set_HE(8192,vector<int>{31,20,31},20);
  for(int i=0;i<8192;i++){
    if(i%100==0){
      cout << "i:" << i << endl;
    }
    for(int j=0;j<8192;j++){
      if(j==i){
        x[j] = HE_CKKS::complex_t(1*(1ULL<<20),0);
      }
      else{
        x[j] = HE_CKKS::complex_t(0,0);
      }
    }
    y = bert_ckks.linear.he1->ifft(x, 8192, true);
    for(int j=0;j<8192;j++){  
      y_real[j] = y[j].real();
      y_imag[j] = y[j].imag();
    }
    write_to_file2("/ezpc_dir/EzPC/SCI/tests/bert_bolt/output/y_real.txt", y_real);
    write_to_file2("/ezpc_dir/EzPC/SCI/tests/bert_bolt/output/y_imag.txt", y_imag);
  }

  
  // write_to_file2("/ezpc_dir/EzPC/SCI/tests/bert_bolt/output/y_int.txt", y_int);
}

void ifft_test(int party, Bert_ckks bert_ckks){
  vector<HE_CKKS::complex_t> x(8192),y(8192);
  for(int j=0;j<8192;j++){
    if(j==1){
      x[j] = HE_CKKS::complex_t(1*(1ULL<<20),0);
    }
    else{
      x[j] = HE_CKKS::complex_t(0,0);
    }
    // x[j] = HE_CKKS::complex_t(rand()%(1ULL<<32),rand()%(1ULL<<32));
  }
  y = bert_ckks.linear.he1->ifft(x, 8192, true);
  write_to_file2("/ezpc_dir/EzPC/SCI/tests/bert_bolt/output/y.txt", y);
  y = bert_ckks.linear.he1->ifft(x, 8192, true);
  write_to_file2("/ezpc_dir/EzPC/SCI/tests/bert_bolt/output/y_matrix.txt", y);
}

void field_to_ring_test(int party, Bert_ckks bert_ckks){
  vector<HE_CKKS::scalar_t> x_field(1),x_ring(1);
  if(party == ALICE){
    x_field[0]=3;
  }
  else{
    x_field[0]=11;
  }
  cout << "init ok" << endl;
  uint64_t Q = 13;
  int size = 1;
  int bitlength = 4;
  int s_in = 2;
  int s_out = 2;
  bert_ckks.nonlinear.field_to_ring(
      1,
      x_field.data(),
      Q,
      x_ring.data(),
      size,
      bitlength,
      s_in,
      s_out
  );
  cout << "x_ring[0]:" << x_ring[0] << endl;
}

void ring_to_field_test(int party, Bert_ckks bert_ckks){
  vector<HE_CKKS::scalar_t> x_field(1),x_ring(1);
  if(party == ALICE){
    x_ring[0]=4;
  }
  else{
    x_ring[0]=6;
  }
  cout << "init ok" << endl;
  uint64_t Q = 1000000000339;
  int size = 1;
  int bitlength = 3;
  int s_in = 1;
  int s_out = 2;
  bert_ckks.nonlinear.ring_to_field(
      1,
      x_ring.data(),
      Q,
      x_field.data(),
      size,
      bitlength,
      s_in,
      s_out
  );
  cout << "x_field[0]:" << x_field[0] << endl;
}

void mpc_to_ckks_test(int party, Bert_ckks bert_ckks){
  vector<vector<HE_CKKS::scalar_t>> x_share(1);
  if(party == ALICE){
    for(int i=0;i<bert_ckks.linear.he1->num_slots;i++){
      x_share[0].push_back(4*(1<<bert_ckks.linear.he1->fft_scale));
    }
    x_share[0][0]=2*(1<<bert_ckks.linear.he1->fft_scale);
  }
  else{
    for(int i=0;i<bert_ckks.linear.he1->num_slots;i++){
      x_share[0].push_back(0);
    }
  }
  vector<Plaintext> y_share;
  uint64_t Q = bert_ckks.linear.he1->get_Q();
  for(int i=0;i<x_share.size();i++){
    y_share = bert_ckks.mpc_to_ckks(x_share, bert_ckks.linear.he1->num_slots, Q, 40, bert_ckks.linear.he1->fft_scale, bert_ckks.linear.he1->he_scale, bert_ckks.linear.he1);
  }
  x_share = bert_ckks.ckks_to_mpc(y_share, bert_ckks.linear.he1->poly_modulus_degree, Q, 40, bert_ckks.linear.he1->he_scale, bert_ckks.linear.he1);
  cout << "x_share[0][0]:" << x_share[0][0] << endl;
  cout << "comm:" << bert_ckks.get_comm() << endl;
  cout << "round:" << bert_ckks.get_round() << endl;
}

// void square_test(int party, Bert_ckks bert_ckks){
//   FixArray x(party, 4096, false, 41, 20);
//   FixArray y;
//   y = bert_ckks.element_wise_square(4096, x);
//   cout << "y.data[0]:" << y.data[0] << endl;
//   cout << "comm:" << bert_ckks.get_comm() << endl;
//   cout << "round:" << bert_ckks.get_round() << endl;
// }

void layernorm_test(int party, Bert_ckks bert_ckks){
  int bitwidth=40, s=20;
  vector<uint64_t> x(dim*array_size);
  map<char*,FixArray> result,input;
  FixArray x_flat_avg, sigma_flat;
  FPMath *fpmath = bert_ckks.nonlinear.fpmath[0];
  x_flat_avg = fpmath->fix->input(party, bert_ckks.linear.data_lin1_0.image_size*bert_ckks.linear.data_lin1_0.filter_h,1, true, bitwidth, s);
  sigma_flat = fpmath->fix->input(party, bert_ckks.linear.data_lin1_0.image_size,1, true, bitwidth, s);
  x_flat_avg.ell = bitwidth;
  x_flat_avg.s = s;
  sigma_flat.ell = bitwidth;
  sigma_flat.s = s;
  input["x_flat_avg"] = x_flat_avg;
  input["sigma_flat"] = sigma_flat;
  result = bert_ckks.layer_norm_nonlinear(input);
  vector<vector<HE_CKKS::scalar_t>> result2;
  cout << "result[x_flat_avg]:" << result["x_flat_avg"].size << endl;
  cout << "result[sigma_flat]:" << result["sigma_flat"].size << endl;
  cout << "result[w]:" << result["w"].size << endl;
  cout << "result[b]:" << result["b"].size << endl;
  cout << "comm:" << bert_ckks.get_comm() << endl;
  cout << "round:" << bert_ckks.get_round() << endl;
  double tmp_comm = bert_ckks.n_comm;
  double tmp_round = bert_ckks.n_rounds;
  result2 = bert_ckks.layer_norm_part2_attention(result);

  cout << "fusion comm:" << bert_ckks.n_comm-tmp_comm << endl;
  cout << "fusion round:" << bert_ckks.n_rounds-tmp_round << endl;
}


void gelu_test(int party, Bert_ckks bert_ckks){
  int bitwidth=40, s=20;
  map<char*,FixArray> result,input;
  FixArray x_flat_avg, sigma_flat;
  FPMath *fpmath = bert_ckks.nonlinear.fpmath[0];
  x_flat_avg = fpmath->fix->input(party, bert_ckks.linear.data_lin1_0.image_size*bert_ckks.linear.data_lin1_0.filter_h,1, true, bitwidth, s);
  sigma_flat = fpmath->fix->input(party, bert_ckks.linear.data_lin1_0.image_size,1, true, bitwidth, s);
  x_flat_avg.ell = bitwidth;
  x_flat_avg.s = s;
  sigma_flat.ell = bitwidth;
  sigma_flat.s = s;
  input["x_flat_avg"] = x_flat_avg;
  input["sigma_flat"] = sigma_flat;
  result = bert_ckks.layer_norm_nonlinear(input);
  cout << "layer_norm_part1 comm:" << bert_ckks.get_comm() << endl;
  cout << "layer_norm_part1 round:" << bert_ckks.get_round() << endl;

  double tmp_comm = bert_ckks.n_comm;
  double tmp_round = bert_ckks.n_rounds;
  map<char*,FixArray> result2;
  result2 = bert_ckks.ln_part2_linear3_gelu_part1(result);
  cout << "ln_part2_linear3_gelu_part1 comm:" << bert_ckks.get_comm() << endl;
  cout << "ln_part2_linear3_gelu_part1 round:" << bert_ckks.get_round() << endl;
  cout << "fusion comm:" << bert_ckks.n_comm-tmp_comm << endl;
  cout << "fusion round:" << bert_ckks.n_rounds-tmp_round << endl;

  // FixArray result3;
  // result3 = bert_ckks.gelu_nonlinear(result2);
  // cout << "gelu_nonlinear comm:" << bert_ckks.get_comm() << endl;
  // cout << "gelu_nonlinear round:" << bert_ckks.get_round() << endl;

  // cout << "total comm:" << bert_ckks.n_comm << endl;
  // cout << "total round:" << bert_ckks.n_rounds << endl;
}

void part4_test(int party, Bert_ckks bert_ckks){
  int bitwidth=40, s=20;
  FPMath *fpmath = bert_ckks.nonlinear.fpmath[0];
  FixArray input = fpmath->fix->input(party, bert_ckks.linear.data_lin4.image_size*3072, 1, true, bitwidth, s);
  // cout << "bert_ckks.linear.data_lin4.image_size*3072:" << bert_ckks.linear.data_lin4.image_size*3072 << endl;
  input.ell = bitwidth;
  input.s = s;
  double tmp_comm = bert_ckks.n_comm;
  double tmp_round = bert_ckks.n_rounds;
  map<char*,FixArray> result2;
  result2 = bert_ckks.linear4_add_layernorm_part1(input, 0);
  cout << "linear4_add_layernorm_part1 comm:" << bert_ckks.n_comm-tmp_comm << endl;
  cout << "linear4_add_layernorm_part1 round:" << bert_ckks.n_rounds-tmp_round << endl;
}

void part2_test(int party, Bert_ckks bert_ckks){
  int bitwidth=40, s=20;
  auto data = bert_ckks.linear.data_lin2;
  FPMath *fpmath = bert_ckks.nonlinear.fpmath[0];
  FixArray input = fpmath->fix->input(party, PACKING_NUM*data.image_size*data.image_size, 1, true, bitwidth, s);
  // cout << "bert_ckks.linear.data_lin4.image_size*3072:" << bert_ckks.linear.data_lin4.image_size*3072 << endl;
  double tmp_comm = bert_ckks.n_comm;
  double tmp_round = bert_ckks.n_rounds;
  map<char*,FixArray> result2;
  result2 = bert_ckks.Softmax_v_linear_layer_norm_part1(input, 0);
  cout << "linear4_add_layernorm_part1 comm:" << bert_ckks.n_comm-tmp_comm << endl;
  cout << "linear4_add_layernorm_part1 round:" << bert_ckks.n_rounds-tmp_round << endl;
}

void softmax_test(int party, Bert_ckks bert_ckks){
  int bitwidth=40, s=20;
  FixArray input,result;
  FCMetadata data = bert_ckks.linear.data_lin1_0;
  int softmax_dim = data.image_size;
  int softmax_output_size = PACKING_NUM*softmax_dim*softmax_dim;
  input = bert_ckks.nonlinear.fpmath[0]->fix->input(party, softmax_output_size, 1, true, bitwidth, s);
  result = bert_ckks.softmax(input);
  cout << "comm:" << bert_ckks.get_comm() << endl;
  cout << "round:" << bert_ckks.get_round() << endl;
}

void transformer_block_test(int party, Bert_ckks bert_ckks){
  int bitwidth=40, s=20;
  map<char*,FixArray> result,input;
  FixArray x_flat_avg, sigma_flat;
  FPMath *fpmath = bert_ckks.nonlinear.fpmath[0];
  // 1. layernorm+attention
  x_flat_avg = fpmath->fix->input(party, bert_ckks.linear.data_lin1_0.image_size*bert_ckks.linear.data_lin1_0.filter_h,1, true, bitwidth, s);
  sigma_flat = fpmath->fix->input(party, bert_ckks.linear.data_lin1_0.image_size, 1, true, bitwidth, s);
  input["x_flat_avg"] = x_flat_avg;
  input["sigma_flat"] = sigma_flat;
  result = bert_ckks.layer_norm_nonlinear(input);
  cout << "layer_norm_nonlinear done" << endl;
  bert_ckks.other_nonlinear_comm += bert_ckks.get_comm();
  bert_ckks.other_nonlinear_round += bert_ckks.get_round();
  vector<vector<HE_CKKS::scalar_t>> result2;
  result2 = bert_ckks.layer_norm_part2_attention(result);
  cout << "linear part 1 done" << endl;
  // 2. softmax
  FixArray qk,result_fix;
  qk = bert_ckks.nonlinear.fpmath[0]->fix->input(party, result2.size()*result2[0].size(), 1, true, bitwidth, s);
  for(int i=0;i<result2.size();i++){
    for(int j=0;j<result2[i].size();j++){
      qk.data[i*result2[i].size()+j] = result2[i][j];
    }
  }
  qk = bert_ckks.softmax(qk);
  bert_ckks.other_nonlinear_comm += bert_ckks.get_comm();
  bert_ckks.other_nonlinear_round += bert_ckks.get_round();
  cout << "softmax done" << endl;
  // 3. linear part2
  result = bert_ckks.Softmax_v_linear_layer_norm_part1(qk, 0);
  cout << "linear part 2 done" << endl;
  // 4. layernorm nonlinear
  result = bert_ckks.layer_norm_nonlinear(result);
  cout << "layer_norm_nonlinear done" << endl;
  bert_ckks.other_nonlinear_comm += bert_ckks.get_comm();
  bert_ckks.other_nonlinear_round += bert_ckks.get_round();
  // 5. linear part3
  result = bert_ckks.ln_part2_linear3_gelu_part1(result);
  cout << "linear part 3 done" << endl;
  // 6. gelu nonlinear
  result_fix = bert_ckks.gelu_nonlinear(result);
  cout << "gelu nonlinear done" << endl;
  bert_ckks.other_nonlinear_comm += bert_ckks.get_comm();
  bert_ckks.other_nonlinear_round += bert_ckks.get_round();
  // 7. linear part4
  result = bert_ckks.linear4_add_layernorm_part1(result_fix, 0);
  cout << "linear part 4 done" << endl;
  cout << "-----------comm------------" << endl;
  cout << "ct_transfer_comm:" << bert_ckks.ct_transfer_comm << endl;
  cout << "conversion_comm:" << bert_ckks.conversion_comm << endl;
  cout << "other_nonlinear_comm:" << bert_ckks.other_nonlinear_comm << endl;
  cout << "-----------round------------" << endl;
  cout << "ct_transfer_round:" << bert_ckks.ct_transfer_round << endl;
  cout << "conversion_round:" << bert_ckks.conversion_round << endl;
  cout << "other_nonlinear_round:" << bert_ckks.other_nonlinear_round << endl;
}

int main(int argc, char **argv) {
  /************* Argument Parsing  ************/
  /********************************************/
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("N", dim, "Number of operation operations");
  amap.arg("nt", num_threads, "Number of threads");
  amap.arg("ip", address, "IP Address of server (ALICE)");

  amap.parse(argc, argv);
  printf("number of points: %d\n", input_size);
  /********** Setup HE_CKKS ***********/
  /********************************************/
  /********** Setup IO and Base OTs ***********/
  /********************************************/
  std::cout << "All Base OTs Done" << std::endl;

  /************ Generate Test Data ************/
  /********************************************/

  // char fix_key[] = "\x61\x7e\xcd\xa2\xa0\x51\x1e\x96"
  //                      "\x5e\x41\xc2\x9b\x15\x3f\xc7\x7a";

  PRG128 prg(fix_key);
  using scalar_ty = uint64_t;
  Bert_ckks bert_ckks(party, port,address);
  // Bert bt(party, port, address, "/ezpc_dir/EzPC/dataset/quantize/mrpc/weights_txt/", false);
  std::random_device rdv;
  std::vector<double> x(4096);
  vector<scalar_ty> x_mpc(4096);
  vector<double> ifft_r_complex(8192);
  vector<uint64_t> des(8192);
  vector<uint64_t> des_seal(8192);
  vector<scalar_ty> y(8192);
  Plaintext y_ntt;
  vector<uint64_t> y_ntt2(8192);
  vector<double> y_double(8192);
  std::uniform_real_distribution<double> uniform(1.0, 4.0);
  // std::generate_n(x.data(), 4096, [&]() { return uniform(rdv);});
  // y = bert_ckks.linear.ckks_decode(x, 8);
  // y = bert_ckks.linear.ckks_encode(x, 4);
  des = read_from_file<uint64_t>("/ezpc_dir/EzPC/SCI/tests/bert_bolt/output/destination.txt");
  des_seal = read_from_file<uint64_t>("/ezpc_dir/EzPC/SCI/tests/bert_bolt/output/destination_seal.txt");
  // for(int i=0;i<des.size();i++){
  //   if(des[i] != des_seal[i]){
  //     cout << "error" << endl;
  //     cout << i << endl;
  //     cout << des[i] << endl;
  //     cout << des_seal[i] << endl;
  //   }
  // }

  x = read_from_file<double>("/ezpc_dir/EzPC/SCI/tests/bert_bolt/output/r.txt");
  ifft_r_complex = read_from_file<double>("/ezpc_dir/EzPC/SCI/tests/bert_bolt/output/ifft_r_complex.txt");
  // write_to_file2("/ezpc_dir/EzPC/SCI/tests/bert_bolt/output/my_x.txt", x);
  cout<< "ok1" << endl;
  // bert_ckks.linear.set_HE(poly_modulus_degree,vector<int>{31,20,31},20);
  // field_to_ring_test(party,bert_ckks);
  // ring_to_field_test(party,bert_ckks);
  // mpc_to_ckks_test(party,bert_ckks);
  // square_test(party,bert_ckks);
  // layernorm_test(party,bert_ckks);
  // part2_test(party,bert_ckks);
  // gelu_test(party,bert_ckks);
  // part4_test(party,bert_ckks);
  // softmax_test(party,bert_ckks);
  transformer_block_test(party,bert_ckks);
  // ifft_test(party,bert_ckks);
  return 0;
  cout<< "ok2" << endl;
  for(int i=0;i<x.size();i++){
    x_mpc[i] = EncodeToFxp<scalar_ty>(x[i],20);
  }
  // write_to_file2("/ezpc_dir/EzPC/SCI/tests/bert_bolt/output/my_x_mpc.txt", x_mpc);
  y = bert_ckks.linear.he1->ckks_encode(x_mpc, 4096);
  // for(int i=0;i<y.size();i++){
  //   y[i] = std::round(ifft_r_complex[i]);
  // }
  // y[0] = 1048832;
  write_to_file2("/ezpc_dir/EzPC/SCI/tests/bert_bolt/output/my_y.txt", y);
  bert_ckks.linear.he1->ckks_ntt(y,8192,pow(2,20),y_ntt);
  vector<complex<scalar_ty>> y_intt;
  y_intt = bert_ckks.linear.he1->ckks_intt(y_ntt);
  write_to_file2("/ezpc_dir/EzPC/SCI/tests/bert_bolt/output/my_y_intt.txt", y_intt);
  y = bert_ckks.linear.he1->ckks_decode(y_intt, 8192);
  write_to_file2("/ezpc_dir/EzPC/SCI/tests/bert_bolt/output/my_y2.txt", y);
  // for(int i=0;i<4096;i++){
  //   y_double[i] = (y[i]/pow(2,20));
  // }
  // cout << "y_ntt.coeff_count():" << y_ntt.coeff_count() << endl;
  // for(int i=0;i<8192;i++){
  //   y_ntt2[i] = y_ntt.data()[i];
  // }
  // write_to_file2("/ezpc_dir/EzPC/SCI/tests/bert_bolt/output/my_y_double2.txt", y_double);
  // write_to_file2("/ezpc_dir/EzPC/SCI/tests/bert_bolt/output/my_y_ntt.txt", y_ntt2);
  // write_to_file2("/ezpc_dir/EzPC/SCI/tests/bert_bolt/output/my_inv_root_n.txt", bert_ckks.linear.he->inv_root_powers_n);
  // write_to_file2("/ezpc_dir/EzPC/SCI/tests/bert_bolt/output/my_inv_root.txt", bert_ckks.linear.he->inv_root_powers_2n);
  cout << "done" << endl;
  return 0;
}
