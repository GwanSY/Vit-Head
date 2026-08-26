#ifndef BERT_H__
#define BERT_H__

#include <fstream>
#include <iostream>
#include <thread>
#include <math.h>
// #include "linear.h"
#include "linear_ckks.h"
#include "nonlinear.h"

#define NL_NTHREADS 32
#define NL_ELL 37

#define GELU_ELL 20
#define GELU_SCALE 11
#define NL_SCALE 12
#define NUM_CLASS 2

// #define BERT_DEBUG
#define BERT_PERF
#define XTS
// #define BERT_SAVE_RESULTS


using namespace std;
using namespace sci;


class Bert_ckks
{
public:
    int party;
    string address;
    int port;

    sci::NetIO *io;
    double n_rounds = 0;
    double n_comm = 0;
    double ct_transfer_comm = 0;
    double ct_transfer_round = 0;
    double conversion_comm = 0;
    double conversion_round = 0;
    double other_nonlinear_comm = 0;
    double other_nonlinear_round = 0;
    double total_la = 0;
    double ct_transfer_la = 0;
    double conversion_la = 0;
    double other_nonlinear_la = 0;
    double he_compute_la = 0;
    Linear_ckks linear;
    NonLinear nonlinear;
    uint64_t poly_modulus_degree = 8192;
    bool prune;

    Bert_ckks(int party, int port, string address);
    ~Bert_ckks();
    Bert_ckks();

    inline uint64_t get_comm(){
        uint64_t total_comm = io->counter;
        for(int i = 0; i < MAX_THREADS; i++){
            total_comm += nonlinear.iopackArr[i]->get_comm();
        }
        uint64_t ret_comm = total_comm - n_comm;
        n_comm = total_comm;
        return ret_comm;
    }

    inline uint64_t get_round(){
        uint64_t total_round = io->num_rounds;
        for(int i = 0; i < MAX_THREADS; i++){
            total_round += nonlinear.iopackArr[i]->get_rounds();
        }
        uint64_t ret_round = total_round - n_rounds;
        n_rounds = total_round;
        return ret_round;
    }

    vector<vector<HE_CKKS::scalar_t>> ckks_to_mpc(vector<Plaintext> pts, int degree, uint64_t Q, int bit_out, int s_out, HE_CKKS* he, bool half=false);

    // size是slot_num
    vector<Plaintext> mpc_to_ckks(vector<vector<HE_CKKS::scalar_t>> input, int slot, uint64_t Q, int bit_out, int s_in, int s_out, HE_CKKS* he,bool half=false);
    
    FixArray element_wise_square(int32_t dim, FixArray input);

    FixArray div_batch_he(const FixArray& nm, const FixArray& dn, int batch_dn_size, int l_out, int s_out, bool normalized_dn = false);

    FixArray div_batch_he_V2(const FixArray& nm, const FixArray& dn, int batch_dn_size, int l_out, int s_out, bool normalized_dn = false);

    FixArray element_wise_mul(FixArray x, FixArray y);

    FixArray element_wise_square_bfv(int32_t dim, FixArray input);

    map<char*,FixArray> layer_norm_nonlinear(map<char*,FixArray> input);

    vector<vector<HE_CKKS::scalar_t>> layer_norm_part2_attention(std::map<char*,FixArray> input);

    std::map<char*,FixArray> gelu_part1(uint64_t *x, int num_ops, int bitwidth, int s);

    FixArray gelu_nonlinear(std::map<char*,FixArray> input);
    // my softmax
    FixArray softmax_ot(FixArray input);

    FixArray softmax_he(FixArray input);

    vector<FixArray> softmax_fix_he(int party, FixArray x, FPMath* fpmath[]);

    FixArray exp_he(const FixArray &x);

    vector<Plaintext> eval_gelu_poly_old(FixArray x);

    vector<Ciphertext> eval_gelu_poly(vector<Ciphertext> x, HE_CKKS* he);
    vector<Ciphertext> ct_ct_mul_blb(vector<Ciphertext> x,vector<Ciphertext> y, HE_CKKS* he);
    vector<Ciphertext> ct_ct_mul_powerformer(vector<Ciphertext> x,vector<Ciphertext> y, HE_CKKS* he);
    std::map<char*,FixArray> Softmax_v_linear_layer_norm_part1(FixArray input, int layer_id);
    std::map<char*,FixArray> ln_part2_linear3_gelu_part1(std::map<char*,FixArray> input);
    std::map<char*,FixArray> linear4_add_layernorm_part1(FixArray input, int layer_id);
	void mp2ml_test();
};

#endif
