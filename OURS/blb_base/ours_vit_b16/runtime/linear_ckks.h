#define LINEAR_H__

#include "he.h"
#include "FloatingPoint/fp-math.h"
#include "FloatingPoint/fixed-point.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <math.h>
#include "bert_utils.h"
// #include "complex.h"
// PACKING_NUM is the number of heads, OUTPUT_DIM = COMMON_DIM / PACKING_NUM
//Bert-base
#define PACKING_NUM 12
#define INPUT_DIM 128
#define COMMON_DIM 768
#define OUTPUT_DIM 64
#define INTER_DIM 3072
#define ATTENTION_LAYERS 12
// //gpt2
// #define PACKING_NUM 12
// #define INPUT_DIM 64
// #define COMMON_DIM 768
// #define OUTPUT_DIM 64
// #define INTER_DIM 3072
// //bert-large
// #define PACKING_NUM 16
// #define INPUT_DIM 128
// #define COMMON_DIM 1024
// #define OUTPUT_DIM 64
// #define INTER_DIM 4096

using namespace sci;
using namespace std;
using namespace seal;

struct FCMetadata
{
	int slot_count;
	int32_t pack_num;
	int32_t inp_ct;
	// Filter is a matrix
	int32_t filter_h;
	int32_t filter_w;
	int32_t filter_size;
	// Image is a matrix
	int32_t image_size;
};

// The official BLB GPU path calls Phantom's encoder from packed slot values.
// Keep the same values alongside the CPU-only SEAL representation so a real
// checkpoint can use that path instead of the upstream all-one placeholder.
using SemanticPlaintext = vector<double>;
using SemanticPlainMatrix = vector<vector<SemanticPlaintext>>;
using SemanticPackedWeight =
    pair<SemanticPlainMatrix, SemanticPlainMatrix>;

struct PreprocessParams_1{
	vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> wq_pack;
	vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> wk_pack;
	vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> wv_pack;

	vector<vector<Plaintext>> bq_pack;
	vector<vector<Plaintext>> bk_pack;
	vector<vector<Plaintext>> bv_pack;

	vector<Plaintext> cross_masks;

	vector<SemanticPackedWeight> device_wq_pack;
	vector<SemanticPackedWeight> device_wk_pack;
	vector<SemanticPackedWeight> device_wv_pack;
	SemanticPlainMatrix device_bq_pack;
	SemanticPlainMatrix device_bk_pack;
	SemanticPlainMatrix device_bv_pack;
	std::size_t weight_chain = 0;
	std::size_t bias_chain = 0;
	double weight_scale = 0.0;
	double bias_scale = 0.0;
	double input_scale = 0.0;
	bool device_semantic_packed = false;

	// Checkpoint tensors are signed fixed-point integers. Packing converts them
	// directly to their real-number semantics before either the SEAL or Phantom
	// evaluator consumes them. These bits describe the checkpoint's fixed-point
	// denominator; they are deliberately independent of Plaintext::scale().
	int weight_fractional_bits = 6;
	int bias_fractional_bits = 11;
	bool semantic_values_materialized = false;
};

struct PreprocessParams_2{
    pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>> cross_mat_single;
    vector<Plaintext> cross_bias_single;
	SemanticPackedWeight device_cross_mat_single;
	vector<SemanticPlaintext> device_cross_bias_single;
	std::size_t target_chain = 0;
	double weight_scale = 0.0;
	double bias_scale = 0.0;
	double input_scale = 0.0;
	bool device_semantic_packed = false;

	// Set by weights_preprocess for pp_2/pp_3/pp_4.  A negative value is an
	// invalid/unconfigured stage and is rejected instead of silently treating a
	// fixed-point checkpoint integer as a real model value.
	int weight_fractional_bits = -1;
	int bias_fractional_bits = -1;
	bool semantic_values_materialized = false;
};


class Linear_ckks
{
public:
	int party;
	sci::NetIO *io;
	// Profiling-only callbacks. They label existing evaluator regions and do
	// not alter the evaluator graph or any protocol data.
	void *profile_context = nullptr;
	void (*profile_switch_callback)(void *, int) = nullptr;
	void (*profile_he_callback)(void *, int, double) = nullptr;
	/*
	1# LN+Attention, [60, 55, 60, 60, 60], 1.6*e-12
	2# BMM2+LN, 8.3*e-18 [60, 20, 60, 60, 60, 60]
	3# LN+Linear+GeLU, [60, 32, 60, 60, 60, 60, 60, 60], 6.9*e-14, scale: 46
	4# Linear2+ '+' + LN, [60, 20, 60, 60, 60, 60], 8.22*e-18
	scale: 41 (#3 is 46), output range: 30 
	*/
	HE_CKKS *he1;
	HE_CKKS *he2;
	HE_CKKS *he3;
	HE_CKKS *he4;
	HE_CKKS *he_softmax;
	HE_CKKS *he_onetime;
	// NATIVE is an explicit circuit choice made before key generation.  It gives
	// he1 one additional data level for native QK->AV->WO; B--SELECTIVE keep the
	// published BLB Block-2 chain byte-for-byte unchanged.
	bool native_linear_enabled = false;
	bool bert_post_norm = false;

	FCMetadata data_lin1_0;
	FCMetadata data_lin1_1;

    FCMetadata data_lin2;
    FCMetadata data_lin3;
    FCMetadata data_lin4;


	// Attention
	vector<PreprocessParams_1> pp_1;
	vector<PreprocessParams_2> pp_2;
	// Assignment-specific output projections. pp_2 remains the original BLB
	// cache for B--SELECTIVE.  In NATIVE, pp_2_mpc contains only Softmax/ReLU input rows
	// (and owns b_o), while pp_2_native contains only Linear rows with alpha_h
	// fused into W_O and has zero bias.
	vector<PreprocessParams_2> pp_2_mpc;
	vector<PreprocessParams_2> pp_2_native;
	vector<vector<int>> heterogeneous_protocols;
	vector<vector<double>> heterogeneous_alphas;
	vector<PreprocessParams_2> pp_3;
	vector<PreprocessParams_2> pp_4;

	// Layer Norm
    vector<vector<uint64_t>> w_ln_1;
    vector<vector<uint64_t>> b_ln_1;

    vector<vector<uint64_t>> w_ln_2;
    vector<vector<uint64_t>> b_ln_2;


	// Pooling
    vector<vector<uint64_t>> w_p;
    vector<uint64_t> b_p;

    // Classification
	vector<vector<uint64_t>> w_c;
	vector<uint64_t> b_c;

	uint64_t rotation_count = 0;
	// Paper Table 5 audit counters for the BERT-large microbenchmarks.
	// These counters only expose rotations already emitted by the original
	// evaluator path; they do not change packing or execution.
	uint64_t qk_rotation_count = 0;
	uint64_t softmax_v_rotation_count = 0;
	uint64_t ct_ct_multiply_count = 0;
	uint64_t ct_pt_multiply_count = 0;
	// Profiling counter for the original CKKS evaluator path.  This is a
	// counter only; it does not alter the evaluator schedule.
	uint64_t rescale_count = 0;
	// Time spent copying the already-packed real checkpoint plaintexts from
	// the host representation to Phantom.  This profiles the existing backend
	// boundary and does not cache, repack, or change evaluator inputs.
	double plaintext_h2d_time = 0.0;
	double plaintext_device_encode_time = 0.0;
	double plaintext_device_release_time = 0.0;
	double ciphertext_h2d_time = 0.0;
	double ciphertext_d2h_time = 0.0;
	int sequence_length = INPUT_DIM;

	Linear_ckks();
	Linear_ckks(int party, sci::NetIO *io, LOCATION backend,
	            bool enable_native_linear = false);

	// BERT's packing constants were written for 128 tokens.  ViT-B/16 has
	// 197 logical tokens, but the original BLB packing requires the physical
	// sequence to divide the CKKS slot layout; the faithful padded choice is
	// therefore normally 256.  This only changes metadata and cache packing.
	void configure_sequence_length(int tokens);

	PreprocessParams_1 params_preprocessing_ct_ct(
		HE_CKKS* he,
		const vector<vector<vector<uint64_t>>> &w_q,
		const vector<vector<vector<uint64_t>>> &w_k,
		const vector<vector<vector<uint64_t>>> &w_v,
		const vector<vector<uint64_t>> &b_q,
		const vector<vector<uint64_t>> &b_k,
		const vector<vector<uint64_t>> &b_v,
		const FCMetadata &data,
		std::size_t input_chain,
		double input_scale
	);

	PreprocessParams_2 params_preprocessing_ct_pt(
		HE_CKKS* he,
		int32_t input_dim, 
		int32_t common_dim, 
		int32_t output_dim,
		const vector<vector<uint64_t>> &w,
		const vector<uint64_t> &b,
		const FCMetadata &data,
		int weight_fractional_bits,
		int bias_fractional_bits,
		std::size_t input_chain,
		double input_scale
	);
	// Softmax*V emits each packed head pair as
	// [head0.low, head1.low, head0.high, head1.high], while BSGS2 consumes
	// contiguous 64-feature inputs.  Permute only the input rows of W_O so
	// the online ciphertext layout still implements the checkpoint matrix.
	vector<vector<uint64_t>> attention_output_projection_weights_for_bsgs(
		const vector<vector<uint64_t>> &weights) const;
	void weights_preprocess(const BertModel &bm);
	// Profiling-only scope control for offline cache construction.  The
	// default overload above still preprocesses all twelve layers; this
	// overload does not alter any online BLB operator or parameter.
	void weights_preprocess(const BertModel &bm, int layers_to_preprocess);
	void load_model_tensors(const BertModel &bm);
	void configure_heterogeneous_layer(int layer_id,
	                                  const vector<int> &protocols,
	                                  const vector<double> &alphas);
	void weights_preprocess_layer(const BertModel &bm, int layer_id);
	void release_preprocessed_layer(int layer_id);

	void linear_1(
		HE_CKKS* he,
		const vector<UnifiedCiphertext> &input, 
		PreprocessParams_1 &pp,
		const FCMetadata &data,
		vector<UnifiedCiphertext> &output,
		vector<UnifiedCiphertext> *b_qkv_observer = nullptr
		);
	
	void linear_2(
		HE_CKKS* he,
		const vector<UnifiedCiphertext> &input_cts, 
		PreprocessParams_2 &pp,
		const FCMetadata &data,
		vector<UnifiedCiphertext> &Cipher_plain_results
		);

	// concat on dim1
	// output: dim2 x (dim1xdim3)
	vector<vector<uint64_t>> concat_vec(
		uint64_t* att,
		int n,
		int dim1,
		int dim2);

	// concat on dim1
	// output: dim2 x (dim1xdim3)
	void concat(
		uint64_t* input,
		uint64_t* output,
		int n,
		int dim1,
		int dim2);

 	vector<Plaintext> generate_cross_packing_masks(HE_CKKS* he, const FCMetadata &data);

	vector<Ciphertext> rotation_by_one_depth3(
	HE_CKKS* he,
	const FCMetadata &data, 
	const Ciphertext &ct, 
	int k);

	vector<Ciphertext>
	bert_efficient_preprocess_vec(
		HE_CKKS* he,
		vector<int64_t> &input,
		const FCMetadata &data);
	
	
	uint64_t* bert_cross_packing_postprocess(
		HE_CKKS* he,
		vector<Ciphertext> &cts, 
		const FCMetadata &data);
	
	void plain_cross_packing_postprocess(
		uint64_t* input, 
		uint64_t * output,
		bool col_packing,
		const FCMetadata &data);
	
	void plain_cross_packing_postprocess_v(
		uint64_t* input, 
		uint64_t * output,
		bool col_packing,
		const FCMetadata &data);
	
	void plain_col_packing_preprocess(
		uint64_t* input, 
		uint64_t * output,
		uint64_t plain_mod,
		int common_dim,
		int input_dim);
	
	void plain_col_packing_preprocess_vec(
		vector<vector<uint64_t>> input, 
		uint64_t * output,
		uint64_t plain_mod,
		int common_dim,
		int input_dim);
	
	void plain_col_packing_postprocess(
		uint64_t* input, 
		uint64_t * output,
		bool col_packing,
		const FCMetadata &data);

	pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>
	bert_cross_packing_matrix(
		HE_CKKS* he,
		const uint64_t *const *matrix1,
		const uint64_t *const *matrix2,
		const FCMetadata &data);

	vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>>
	bert_cross_packing_single_matrix(
		HE_CKKS* he,
		const vector<vector<vector<uint64_t>>> &weights, 
		const FCMetadata &data,
		int fractional_bits,
		const parms_id_type &target_level,
		double target_scale,
		vector<SemanticPackedWeight> *device_values);
	
	pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>
	bert_cross_packing_single_matrix_2(
		HE_CKKS* he,
		const uint64_t *const *matrix1,
		const uint64_t *const *matrix2,
		const FCMetadata &data,
		int fractional_bits,
		const parms_id_type &target_level,
		double target_scale,
		SemanticPackedWeight *device_values);

	vector<vector<Plaintext>> bert_cross_packing_bias(
		HE_CKKS* he,
		const vector<vector<uint64_t>> &bias, 
		const FCMetadata &data,
		int fractional_bits,
		const parms_id_type &target_level,
		double target_scale,
		SemanticPlainMatrix *device_values);

	
	vector<Plaintext> bert_cross_packing_bias_2(
		HE_CKKS* he,
		const uint64_t *matrix, 
		const FCMetadata &data,
		int fractional_bits,
		const parms_id_type &target_level,
		double target_scale,
		vector<SemanticPlaintext> *device_values);

	// Cipher * Plain with BSGS for the first layer (Q,K,V)
	void bert_cipher_plain_bsgs(
		HE_CKKS* he,
		const vector<UnifiedCiphertext> &cts, 
		PreprocessParams_1 &pp,
		const FCMetadata &data, 
		vector<UnifiedCiphertext> &result);

	// Cipher * Plain with BSGS for other linear layers
	void bert_cipher_plain_bsgs_2(
		HE_CKKS* he,
		const vector<UnifiedCiphertext> &cts, 
		PreprocessParams_2 &pp,
		const FCMetadata &data, 
		vector<UnifiedCiphertext> &result);

	// Cipher * Cipher in the attention mechanism
	void bert_cipher_cipher_cross_packing(
		HE_CKKS* he,
		const FCMetadata &data,
		vector<UnifiedCiphertext> Cipher_plain_result,
		vector<UnifiedCiphertext> &results);

	// Softmax * V

	vector<vector<vector<Plaintext>>> bert_softmax_v_packing_single_matrix(
		HE_CKKS* he,
		const vector<vector<vector<uint64_t>>> &weights, 
		const FCMetadata &data);

	void softmax_v(
		HE_CKKS* he,
		const vector<vector<vector<UnifiedCiphertext>>> &softmax_s2, 
		const vector<UnifiedCiphertext> &V, 
		const FCMetadata &data, 
		vector<UnifiedCiphertext> &result,
		double output_mask_value = 1.0,
		const vector<bool> *active_heads = nullptr,
		bool post_av_rescale = true);

	void preprocess_softmax_s1_ct_ct(
		HE_CKKS* he, const vector<UnifiedCiphertext> &matrix,
		const FCMetadata &data,
		vector<vector<vector<UnifiedCiphertext>>> &s2_pack,
		const vector<bool> *active_heads = nullptr);

	// for different applications, we need to set different HE parameters
	void set_HE(LOCATION backend, bool enable_native_linear = false);
	// void element_wise_mult(int32_t dim, FixArray input, FixArray weight, FixArray output);

	// void ckks_to_mpc(Plaintext pt, uint64_t* output, int length, uint64_t Q);


};
