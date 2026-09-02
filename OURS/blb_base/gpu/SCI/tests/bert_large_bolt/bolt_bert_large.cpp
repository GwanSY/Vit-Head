/*
*/

#include "bert.h"
#include <fstream>
#define ALICE sci::ALICE
using namespace std;
using namespace seal;
using namespace sci;

int party = 0;
int port = 8000;
string address = "127.0.0.1";
int num_threads = 4;
int bitlength = 37;

string path = "../../dataset/quantize/mrpc";
string output_file_path = "../tests/output/ppnlp_test.txt";
int num_class = 2;
int sample_id = 0;
int num_sample = 1;

bool pruning = false;

// void layernorm_bolt_test(int party, Bert bert){
//     vector<uint64_t> x(128*768);
//     vector<uint64_t> y(128*768);
//     map<char*,FixArray> result;
//     NonLinear nl = bert.nl;
//     // Layer Norm
//     nl.layer_norm(
//         1,
//         x.data(),
//         y.data(),
//         ln_weight_2,
//         ln_bias_2,
//         data.image_size,
//         COMMON_DIM,
//         NL_ELL,
//         NL_SCALE
//     );

//     // wx
//     if(party == ALICE){
//         vector<Ciphertext> ln = ss_to_he_server(
//             lin.he_8192_ln, 
//             ln_2_output_row,
//             ln_2_input_size, 
//             NL_ELL);
//         vector<Ciphertext> ln_w = lin.w_ln(lin.he_8192_ln, ln, lin.w_ln_2_pt[layer_id]);
//         he_to_ss_server(lin.he_8192_ln, ln_w, ln_2_wx, true);
//     } else{
//         ss_to_he_client(
//             lin.he_8192_ln, 
//             ln_2_output_row,
//             ln_2_input_size, 
//             NL_ELL);
//         int cts_size = ln_2_input_size / lin.he_8192_ln->poly_modulus_degree;
//         he_to_ss_client(lin.he_8192_ln, ln_2_wx, cts_size ,data);
//     }

//     nl.gt_p_sub(
//         NL_NTHREADS,
//         ln_2_wx,
//         lin.he_8192_ln->plain_mod,
//         ln_2_wx,
//         ln_2_input_size,
//         NL_ELL,
//         2*NL_SCALE,
//         NL_SCALE
//     );

//     uint64_t ell_mask = (1ULL << (NL_ELL)) - 1;

//     for(int i = 0; i < ln_2_input_size; i++){
//         ln_2_wx[i] += ln_bias_2[i] & ell_mask;
//     }
//     // update H1
//     memcpy(h1_cache_12, ln_2_wx, ln_2_input_size*sizeof(uint64_t));

//     // Rescale
//     nl.right_shift(
//         NL_NTHREADS,
//         ln_2_wx,
//         12 - 5,
//         ln_2_output_row,
//         ln_2_input_size,
//         NL_ELL,
//         NL_SCALE
//     );
//     lin.plain_col_packing_preprocess(
//         ln_2_output_row,
//         ln_2_output_col,
//         lin.he_8192_tiny->plain_mod,
//         data.image_size,
//         COMMON_DIM
//     );
//     if(party == ALICE){
//         h1 = ss_to_he_server(
//         lin.he_8192, 
//         ln_2_output_col,
//         ln_2_input_size, 
//         NL_ELL);
//     } else{
//         ss_to_he_client(
//             lin.he_8192, 
//             ln_2_output_col, 
//             ln_2_input_size, 
//             NL_ELL);
//     }
//     if(party == ALICE){
//         cout << "-> Layer - " << layer_id << ": Linear #1 HE" << endl;
//         vector<Ciphertext> q_k_v = lin.linear_1(
//             lin.he_8192,
//             h1,
//             lin.pp_1[layer_id],
//             data
//         );
//         cout << "-> Layer - " << layer_id << ": Linear #1 done HE" << endl;

//         int qk_offset = qk_size / data.slot_count;

//         enc_v = { q_k_v.begin() + qk_offset, q_k_v.end()};

//         parms_id_type parms_id = q_k_v[0].parms_id();
//         shared_ptr<const SEALContext::ContextData> context_data = lin.he_8192->context->get_context_data(parms_id);
        
//         #pragma omp parallel for
//         for (int i = 0; i < qk_offset; i++) {
//             flood_ciphertext(q_k_v[i], context_data, SMUDGING_BITLEN_bert1);
//             lin.he_8192->evaluator->mod_switch_to_next_inplace(q_k_v[i]);
//             lin.he_8192->evaluator->mod_switch_to_next_inplace(q_k_v[i]);
//         }

//         vector<Ciphertext> q_k = {q_k_v.begin(), q_k_v.begin() + qk_offset};
//         // vector<Ciphertext> v = { q_k_v.begin() + qk_offset, q_k_v.end()};

//         he_to_ss_server(lin.he_8192, q_k, qk_v_cross, true);
//         // he_to_ss_server(lin.he_8192, v, &qk_v_cross[qk_size], false);
//     } else{
//         int qk_cts_len = qk_size / lin.he_8192->poly_modulus_degree;
//         int v_cts_len = v_size / lin.he_8192->poly_modulus_degree;
//         he_to_ss_client(lin.he_8192, qk_v_cross, qk_cts_len, data);
//         // he_to_ss_client(lin.he_8192, &qk_v_cross[qk_size], v_cts_len, data);
//     }

//     lin.plain_cross_packing_postprocess(
//         qk_v_cross, 
//         softmax_input_row,
//         // we need row packing
//         false,
//         data);

//     // Scale: Q*V 22, he to ss
//     nl.gt_p_sub(
//         NL_NTHREADS,
//         softmax_input_row,
//         lin.he_8192->plain_mod,
//         softmax_input_row,
//         qk_size,
//         NL_ELL,
//         22,
//         22
//     );
//     nl.right_shift(
//         NL_NTHREADS,
//         softmax_input_row,
//         22 - NL_SCALE,
//         softmax_input_row,
//         qk_size,
//         NL_ELL,
//         22
//     );

//   cout << "comm:" << bert_ckks.get_comm() << endl;
//   cout << "round:" << bert_ckks.get_round() << endl;
// }

int main(int argc, char **argv) {
    ArgMapping amap;
    amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
    amap.arg("p", port, "Port Number");
    amap.arg("ip", address, "IP Address of server (ALICE)");
    amap.arg("path", path, "Path of dataset and model");
    amap.arg("num_class", num_class, "Number of classification labels");
    amap.arg("id", sample_id, "Index of first sample");
    amap.arg("num_sample", num_sample, "Length of test data");
    amap.arg("prune", pruning, "Input pruning");
    amap.arg("output", output_file_path, "Path of output");
    amap.parse(argc, argv);

    cout << ">>> Evaluating Bert" << endl;
    cout << "-> Role: " << party << endl;
    cout << "-> Address: " << address << endl;
    cout << "-> Port: " << port << endl;
    cout << "<<<" << endl << endl;
    pruning = false;
    Bert bt(party, port, address, path + "/weights_txt/", pruning);

    auto start = high_resolution_clock::now();
    
    vector<vector<double>> inference_results;
    vector<int> predicted_labels;

    if(party == ALICE){
        for(int i = sample_id; i < sample_id + num_sample; i++ ){
            cout << "==>> Inference sample #" << i << endl;
            vector<double> result = bt.run("", "");
        }
    } else{
        ofstream file(output_file_path);
        if (!file) {
            std::cerr << "Could not open the file: " << output_file_path <<std::endl;
            return {};
        }
        for(int i = sample_id; i < sample_id + num_sample; i++ ){
            cout << "==>> Inference sample #" << i << endl;
            vector<double> result = bt.run(
                path + "/weights_txt/inputs_" + to_string(i) + "_data.txt",
                path + "/weights_txt/inputs_" + to_string(i) +  "_mask.txt"
                );
            if(result.size() == 1){
                file << result[0]<< endl;
            } else if(result.size() == 2){
                // inference_results.push_back(result);
                auto max_ele = max_element(result.begin(), result.end());
                int max_index = distance(result.begin(), max_ele);
                // predicted_labels.push_back(max_index);
                file << max_index << "," 
                        << result[0]<< "," 
                        << result[1] << endl;
            }
        }
        file.close();
    }
    
    auto end = high_resolution_clock::now();
    auto interval = (end - start)/1e+9;
    
    cout << "-> End to end takes: " << interval.count() << "sec" << endl;

    return 0;
}
