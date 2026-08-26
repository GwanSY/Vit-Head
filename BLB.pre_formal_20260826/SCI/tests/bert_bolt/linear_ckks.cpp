#include "linear_ckks.h"

vector<Plaintext> Linear_ckks::generate_cross_packing_masks(HE_CKKS* he, const FCMetadata &data) {
    vector<Plaintext> result(data.image_size * 2);
    #pragma omp parallel for
    for (int i = 0; i < data.image_size; i++) {
        vector<double> mask1(data.slot_count, 0ULL);
        vector<double> mask2(data.slot_count, 0ULL);
        for (int k = 0; k < data.image_size - i; k++) {
            mask1[k + (i * data.image_size) % (data.slot_count / 2)] = 1;
            mask1[k + data.slot_count / 2 + (i * data.image_size) % (data.slot_count / 2)] = 1;
        }
        for (int k = data.image_size - i; k < data.image_size; k++) {
            mask2[k + (i * data.image_size) % (data.slot_count / 2)] = 1;
            mask2[k + data.slot_count / 2 + (i * data.image_size) % (data.slot_count / 2)] = 1;
        }
        Plaintext pt1, pt2;
        he->encoder->encode(mask1, pow(2,he->he_scale), pt1);
        he->encoder->encode(mask2, pow(2,he->he_scale), pt2);
        result[i] = pt1;
        result[i + data.image_size] = pt2;
    }
    return result;
}

PreprocessParams_1 Linear_ckks::params_preprocessing_ct_ct(
    HE_CKKS* he,
    vector<vector<vector<uint64_t>>> w_q,
    vector<vector<vector<uint64_t>>> w_k,
    vector<vector<vector<uint64_t>>> w_v,
    vector<vector<uint64_t>> b_q,
    vector<vector<uint64_t>> b_k,
    vector<vector<uint64_t>> b_v,
    const FCMetadata &data
){
    PreprocessParams_1 pp;

    cout << "filter.h, filter.w:" << data.filter_h << "," << data.filter_w << endl;
    // vector<vector<vector<uint64_t>>> weights_q(12, vector<vector<uint64_t>>(data.filter_h, vector<uint64_t>(data.filter_w)));
    // vector<vector<vector<uint64_t>>> weights_k(12, vector<vector<uint64_t>>(data.filter_h, vector<uint64_t>(data.filter_w)));
    // vector<vector<vector<uint64_t>>> weights_v(12, vector<vector<uint64_t>>(data.filter_h, vector<uint64_t>(data.filter_w)));

    // vector<vector<uint64_t>> bias_q(12, vector<uint64_t>(data.filter_w));
    // vector<vector<uint64_t>> bias_k(12, vector<uint64_t>(data.filter_w));
    // vector<vector<uint64_t>> bias_v(12, vector<uint64_t>(data.filter_w));
    
    pp.wq_pack = bert_cross_packing_single_matrix(he, w_q, data);
    pp.wk_pack = bert_cross_packing_single_matrix(he, w_k, data);
    pp.wv_pack = bert_cross_packing_single_matrix(he, w_v, data);
    // cout << "OK here" << endl;
    // exit(0);
    pp.bq_pack = bert_cross_packing_bias(he, b_q, data);
    pp.bk_pack = bert_cross_packing_bias(he, b_k, data);
    pp.bv_pack = bert_cross_packing_bias(he, b_v, data);

	pp.cross_masks = generate_cross_packing_masks(he, data);

    return pp;
}

PreprocessParams_2 Linear_ckks::params_preprocessing_ct_pt(
    HE_CKKS* he,
    int32_t input_dim, 
    int32_t common_dim, 
    int32_t output_dim,
    vector<vector<uint64_t>> w,
    vector<uint64_t> b,
    const FCMetadata &data
){
    PreprocessParams_2 pp;

    vector<uint64_t *> matrix_mod_p1(common_dim);
    vector<uint64_t *> matrix_mod_p2(common_dim);

    vector<uint64_t *> matrix1(common_dim);
    vector<uint64_t *> matrix2(common_dim);
    cout << "common_dim, output_dim/2:" << common_dim << "," << output_dim/2 << endl;
    #pragma omp parallel for
    for (int i = 0; i < common_dim; i++) {
        matrix_mod_p1[i] = new uint64_t[output_dim / 2];
        matrix_mod_p2[i] = new uint64_t[output_dim / 2];

        matrix1[i] = new uint64_t[output_dim / 2];
        matrix2[i] = new uint64_t[output_dim / 2];
        #pragma omp parallel for
        for (int j = 0; j < output_dim / 2; j++) {
            matrix_mod_p1[i][j] = w[i][j];
            matrix_mod_p2[i][j] = w[i][j + output_dim / 2];
        }
    }
    
    pp.cross_mat_single = bert_cross_packing_single_matrix_2(he, matrix_mod_p1.data(), matrix_mod_p2.data(), data);
    cout << "here OK" << endl;
    pp.cross_bias_single = bert_cross_packing_bias_2(he, b.data(), data);
    return pp;
}

void Linear_ckks::weights_preprocess(BertModel &bm){
    omp_set_num_threads(4);
    #pragma omp parallel for
    for(int i = 0; i < ATTENTION_LAYERS; i++){
        cout << "Preprocessing layer " << i << endl;
        FCMetadata data = data_lin1_0;
        if(i > 0){
            data = data_lin1_1;
        }
        pp_1[i] = params_preprocessing_ct_ct(
            he1,
            bm.w_q[i],
            bm.w_k[i],
            bm.w_v[i],
            bm.b_q[i],
            bm.b_k[i],
            bm.b_v[i],
            data
        );
        cout << "pp_1 done" << endl;
        // exit(0);
        pp_2[i] = params_preprocessing_ct_pt(
            he2,
            INPUT_DIM,
            COMMON_DIM,
            COMMON_DIM,
            bm.w_o[i],
            bm.b_o[i],
            data_lin2
        );
        cout << "pp_2 done" << endl;
        pp_3[i] = params_preprocessing_ct_pt(
            he3,
            INPUT_DIM,
            COMMON_DIM,
            INTER_DIM,
            bm.w_i_1[i],
            bm.b_i_1[i],
            data_lin3
        );
        cout << "pp_3 done" << endl;
        pp_4[i] = params_preprocessing_ct_pt(
            he4,
            INPUT_DIM,
            INTER_DIM,
            COMMON_DIM,
            bm.w_i_2[i],
            bm.b_i_2[i],
            data_lin4
        );
        cout << "pp_4 done" << endl;
        int slot_count = he1->poly_modulus_degree/2;
        int w_length = data_lin2.image_size*COMMON_DIM;
        int64_t* w_1 = new int64_t[w_length];
        int64_t* w_2 = new int64_t[w_length];

        vector<Plaintext> pt_w_1(w_length / slot_count);
        vector<Plaintext> pt_w_2(w_length / slot_count);
        for(int j = 0; j < data_lin2.image_size; j++){
            memcpy(&w_1[j*COMMON_DIM], bm.w_ln_1[i].data(), COMMON_DIM*sizeof(uint64_t));
            memcpy(&w_2[j*COMMON_DIM], bm.w_ln_2[i].data(), COMMON_DIM*sizeof(uint64_t));
        }
        cout << "half done" << endl;
        for(int i = 0; i < pt_w_1.size(); i++){
            vector<double> tmp_1(&w_1[i*slot_count], &w_1[(i+1)*slot_count]);
            Plaintext pt_1;
            he1->encoder->encode(tmp_1, pow(2,he1->he_scale), pt_1);
            pt_w_1[i] = pt_1;

            vector<double> tmp_2(&w_2[i*slot_count], &w_2[(i+1)*slot_count]);
            Plaintext pt_2;
            he2->encoder->encode(tmp_2, pow(2,he2->he_scale), pt_2);
            pt_w_2[i] = pt_2;
        }

        w_ln_1_pt[i] = pt_w_1;
        w_ln_2_pt[i] = pt_w_2;
    }

    w_ln_1 = bm.w_ln_1;
    b_ln_1 = bm.b_ln_1;

    w_ln_2 = bm.w_ln_2;
    b_ln_2 = bm.b_ln_2;

    w_c = bm.w_c;
    b_c = bm.b_c;
    w_p = bm.w_p;
    b_p = bm.b_p;
}

vector<Ciphertext> Linear_ckks::linear_1(
		HE_CKKS* he,
		vector<Ciphertext> input_cts, 
		PreprocessParams_1 &pp,
		const FCMetadata &data) {

    cout << "input_cts.size: " << input_cts.size() << endl;
	vector<Ciphertext> Cipher_plain_results(data.image_size * data.filter_w * 3 * 12 / data.slot_count);
	// Q,K,V
    bert_cipher_plain_bsgs(
        he, 
        input_cts, 
        pp.wq_pack,
        pp.wk_pack,
        pp.wv_pack,
        pp.bq_pack,
        pp.bk_pack,
        pp.bv_pack,
        data, 
        Cipher_plain_results
    );
    for(int i=0;i<Cipher_plain_results.size();i++){
        // he->evaluator->rescale_to_next_inplace(Cipher_plain_results[i]);
        he->evaluator->relinearize_inplace(Cipher_plain_results[i], *he->relin_keys);
    }
    cout << "scale after WX: " << log2(Cipher_plain_results[0].scale()) << endl;
    cout << "bert_cipher_plain_bsgs done" << endl;
	vector<Ciphertext> HE_result(
        data.image_size * data.image_size * 12 / data.slot_count 
        + data.image_size * data.filter_w * 12 / data.slot_count
    );
	//QK^T
    bert_cipher_cipher_cross_packing(he, data, Cipher_plain_results, pp.cross_masks, HE_result);

    for (int i = 0; i < data.image_size * data.filter_w * 12 / data.slot_count; i++) {
        HE_result[data.image_size * data.image_size * 12 / data.slot_count + i] = 
            Cipher_plain_results[data.image_size * data.filter_w * 12 * 2 / data.slot_count + i];
    }

    return HE_result;
}

vector<Ciphertext> Linear_ckks::linear_2(
    HE_CKKS* he,
    vector<Ciphertext> input_cts, 
    PreprocessParams_2 &pp,
    const FCMetadata &data
    ){
    vector<Ciphertext> Cipher_plain_results(data.image_size * data.filter_w / data.slot_count);
    bert_cipher_plain_bsgs_2(he, input_cts, pp.cross_mat_single.first, pp.cross_mat_single.second, pp.cross_bias_single, data, Cipher_plain_results);

    return Cipher_plain_results;
}

pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>> 
Linear_ckks::bert_cross_packing_matrix(
	HE_CKKS* he,
	const uint64_t *const *matrix1, 
	const uint64_t *const *matrix2, 
	const FCMetadata &data) {

    vector<vector<Plaintext>> weightMatrix1; // 64 x 48
    vector<vector<Plaintext>> weightMatrix2; // 64 x 48
    vector<double> temp2;
    int num_diag = data.slot_count / data.image_size / 2; // should be 8
    int num_matrix_per_row = data.filter_h / num_diag; // should be 48
    int num_matrix_per_col = data.filter_w / num_diag; // should be 8

    int n1;
    int n2;
    if (data.slot_count == 4096) {
        n1 = 4;
        n2 = 4;
    }
    else {
        n1 = 8;
        n2 = 4;
    }

    for (int col_ind = 0; col_ind < num_matrix_per_col; col_ind++) {
        int matrix_flag = 0;
        for (int l = 0; l < num_diag; l++) {
            vector<Plaintext> temp_matrix_diag(data.filter_h * data.image_size / data.slot_count);
            int matrix_diag_index = 0;
            for (int i = 0; i < num_matrix_per_row; i++) {
                for (int j = 0; j < num_diag; j++) {
                    for (int k = 0; k < data.image_size; k++) {
                        if (matrix_flag == 0)
                            temp2.push_back((int64_t)matrix1[i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                        else
                            temp2.push_back((int64_t)matrix2[i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                    }
                    if (temp2.size() % (data.slot_count / 2) == 0) {
                        matrix_flag = (matrix_flag + 1) % 2;
                        std::rotate(temp2.begin() + (temp2.size() / (data.slot_count / 2) - 1) * data.slot_count / 2, temp2.begin() + temp2.size() - (l % n1) * data.image_size, temp2.end());
                        if (temp2.size() == data.slot_count) {
                            Plaintext pt;
                            he->encoder->encode(temp2,pow(2, he->he_scale), pt);
                            temp_matrix_diag[matrix_diag_index] = pt;
                            matrix_diag_index++;
                            temp2.clear();
                        }
                    }
                }
            }
            weightMatrix1.push_back(temp_matrix_diag);
        }
    }

    for (int col_ind = 0; col_ind < num_matrix_per_col; col_ind++) {
        int matrix_flag = 0;
        for (int l = 0; l < num_diag; l++) {
            vector<Plaintext> temp_matrix_diag(data.filter_h * data.image_size / data.slot_count);
            int matrix_diag_index = 0;
            for (int i = 0; i < num_matrix_per_row; i++) {
                for (int j = 0; j < num_diag; j++) {
                    for (int k = 0; k < data.image_size; k++) {
                        if (matrix_flag == 0)
                            temp2.push_back((int64_t)matrix2[i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                        else
                            temp2.push_back((int64_t)matrix1[i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                    }
                    if (temp2.size() % (data.slot_count / 2) == 0) {
                        matrix_flag = (matrix_flag + 1) % 2;
                        std::rotate(temp2.begin() + (temp2.size() / (data.slot_count / 2) - 1) * data.slot_count / 2, temp2.begin() + temp2.size() - (l % n1) * data.image_size, temp2.end());
                        if (temp2.size() == data.slot_count) {
                            std::rotate(temp2.begin(), temp2.begin() + temp2.size() / 2, temp2.end());
                            Plaintext pt;
                            he->encoder->encode(temp2, pow(2, he->he_scale), pt);
                            temp_matrix_diag[matrix_diag_index] = pt;
                            matrix_diag_index++;
                            temp2.clear();
                        }
                    }
                }
            }
            weightMatrix2.push_back(temp_matrix_diag);
        }
    }
    return std::make_pair(weightMatrix1, weightMatrix2);
}

vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>>
Linear_ckks::bert_cross_packing_single_matrix(HE_CKKS* he, const vector<vector<vector<uint64_t>>> &weights, const FCMetadata &data) {
    vector<pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>> result(6);
    int num_diag = data.slot_count / data.image_size / 2;

    int n1 = 8;
    int n2 = 4;
    if (data.image_size == 64) {
        n1 = 16;
        n2 = 4;
    }

    int weight_height = data.filter_h;

    int num_matrix_per_row = weight_height / num_diag;
    int num_matrix_per_col = data.filter_w / num_diag;
    // cout << "num_matrix_per_col,num_matrix_per_row,num_diag,data.image_size: " << num_matrix_per_col << "," << num_matrix_per_row << "," << num_diag << "," << data.image_size << endl;
    
    #pragma omp parallel for
    for (int packing_ind = 0; packing_ind < 6; packing_ind++) {
        vector<double> temp2;
        vector<vector<Plaintext>> weightMatrix1;
        vector<vector<Plaintext>> weightMatrix2;
        int cot = 0;
        for (int col_ind = 0; col_ind < num_matrix_per_col; col_ind++) {
            int matrix_flag = 0;
            // #pragma omp parallel for num_threads(4)
            for (int l = 0; l < num_diag; l++) {
                vector<Plaintext> temp_matrix_diag2(weight_height * data.image_size / data.slot_count);
                vector<vector<double>> temp_matrix_diag(weight_height * data.image_size / data.slot_count);
                int matrix_diag_index = 0;
                for (int i = 0; i < num_matrix_per_row; i++) {
                    for (int j = 0; j < num_diag; j++) {
                        for (int k = 0; k < data.image_size; k++) {
                            if (matrix_flag == 0)
                                temp2.push_back(weights[packing_ind * 2][i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                            else
                                temp2.push_back(weights[packing_ind * 2 + 1][i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                        }
                        if (temp2.size() % (data.slot_count / 2) == 0) {
                            matrix_flag = (matrix_flag + 1) % 2;
                            std::rotate(temp2.begin() + (temp2.size() / (data.slot_count / 2) - 1) * data.slot_count / 2, temp2.begin() + temp2.size() - (l % n1) * data.image_size, temp2.end());
                            if (temp2.size() == data.slot_count) {
                                temp_matrix_diag[matrix_diag_index] = temp2;
                                matrix_diag_index++;
                                temp2.clear();
                            }
                        }
                        // if (matrix_flag == 0)
                        //     std::fill(temp2.begin() + cot*data.image_size, temp2.begin()+(cot+1)*data.image_size, (int64_t)weights[packing_ind * 2][i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                        // else
                        //     std::fill(temp2.begin() + cot*data.image_size, temp2.begin()+(cot+1)*data.image_size, (int64_t)weights[packing_ind * 2 + 1][i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                        // cot++;
                        
                        // if (cot==(data.slot_count / 2/ data.image_size)) {
                        //     matrix_flag = (matrix_flag + 1) % 2;
                        //     std::rotate(temp2.begin(), temp2.begin() + data.slot_count / 2 - (l % n1) * data.image_size, temp2.begin()+data.slot_count / 2);
                        // }
                        // else if (cot==(data.slot_count / data.image_size)) {
                        //     matrix_flag = (matrix_flag + 1) % 2;
                        //     std::rotate(temp2.begin() + (temp2.size() / (data.slot_count / 2) - 1) * data.slot_count / 2, temp2.begin() + temp2.size() - (l % n1) * data.image_size, temp2.end());
                        //     Plaintext pt;
                        //     vector<double> temp3(temp2.size());
                        //     he->encoder->encode(temp2, pow(2, he->he_scale), pt);
                        //     temp_matrix_diag[matrix_diag_index] = pt;
                        //     matrix_diag_index++;
                        //     cot = 0;
                        // }
                    }
                }
                
                #pragma omp parallel for num_threads(temp_matrix_diag.size())
                for(int i=0;i<temp_matrix_diag.size();i++){
                    Plaintext pt;
                    he->encoder->encode(temp_matrix_diag[i], pow(2, he->he_scale), pt);
                    temp_matrix_diag2[i] = pt;
                }
                weightMatrix1.push_back(temp_matrix_diag2);
            }
        }
        // cout << "OK HERE hhh" << endl;
        for (int col_ind = 0; col_ind < num_matrix_per_col; col_ind++) {
            int matrix_flag = 0;
            // #pragma omp parallel for num_threads(4)
            for (int l = 0; l < num_diag; l++) {
                vector<Plaintext> temp_matrix_diag2(weight_height * data.image_size / data.slot_count);
                vector<vector<double>> temp_matrix_diag(weight_height * data.image_size / data.slot_count);
                int matrix_diag_index = 0;
                for (int i = 0; i < num_matrix_per_row; i++) {
                    for (int j = 0; j < num_diag; j++) {
                        for (int k = 0; k < data.image_size; k++) {
                            if (matrix_flag == 0)
                                temp2.push_back(weights[packing_ind * 2 + 1][i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                            else
                                temp2.push_back(weights[packing_ind * 2][i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                        }
                        if (temp2.size() % (data.slot_count / 2) == 0) {
                            matrix_flag = (matrix_flag + 1) % 2;
                            std::rotate(temp2.begin() + (temp2.size() / (data.slot_count / 2) - 1) * data.slot_count / 2, temp2.begin() + temp2.size() - (l % n1) * data.image_size, temp2.end());
                            if (temp2.size() == data.slot_count) {
                                std::rotate(temp2.begin(), temp2.begin() + temp2.size() / 2, temp2.end());
                                temp_matrix_diag[matrix_diag_index] = temp2;
                                matrix_diag_index++;
                                temp2.clear();
                            }
                        }
                        // if (matrix_flag == 0)
                        //     std::fill(temp2.begin() + cot*data.image_size, temp2.begin()+(cot+1)*data.image_size, (int64_t)weights[packing_ind * 2 + 1][i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                        // else
                        //     std::fill(temp2.begin() + cot*data.image_size, temp2.begin()+(cot+1)*data.image_size, (int64_t)weights[packing_ind * 2][i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                        // cot++;
                        // if (cot==(data.slot_count / 2/ data.image_size)) {
                        //     matrix_flag = (matrix_flag + 1) % 2;
                        //     std::rotate(temp2.begin(), temp2.begin() + data.slot_count / 2 - (l % n1) * data.image_size, temp2.begin()+data.slot_count / 2);
                        // }
                        // else if (cot==(data.slot_count / data.image_size)) {
                        //     matrix_flag = (matrix_flag + 1) % 2;
                        //     std::rotate(temp2.begin() + (temp2.size() / (data.slot_count / 2) - 1) * data.slot_count / 2, temp2.begin() + temp2.size() - (l % n1) * data.image_size, temp2.end());
                        //     std::rotate(temp2.begin(), temp2.begin() + temp2.size() / 2, temp2.end());
                        //     Plaintext pt;
                        //     he->encoder->encode(temp2, pow(2, he->he_scale), pt);
                        //     temp_matrix_diag[matrix_diag_index] = pt;
                        //     matrix_diag_index++;
                        //     cot = 0;
                        // }
                    }
                }
                #pragma omp parallel for num_threads(temp_matrix_diag.size())
                for(int i=0;i<temp_matrix_diag.size();i++){
                    Plaintext pt;
                    he->encoder->encode(temp_matrix_diag[i], pow(2, he->he_scale), pt);
                    temp_matrix_diag2[i] = pt;
                }
                weightMatrix2.push_back(temp_matrix_diag2);
            }
        }
        // cout << "weightMatrix1.size: " << weightMatrix1.size() << "," << weightMatrix1[0].size() << endl;
        // cout << "weightMatrix2.size: " << weightMatrix2.size() << "," << weightMatrix2[0].size() << endl;
        result[packing_ind] = std::make_pair(weightMatrix1, weightMatrix2);
    }
    return result;
}

pair<vector<vector<Plaintext>>, vector<vector<Plaintext>>>
Linear_ckks::bert_cross_packing_single_matrix_2(
    HE_CKKS* he,
    const uint64_t *const *matrix1,
    const uint64_t *const *matrix2,
    const FCMetadata &data){
    
    vector<vector<Plaintext>> weightMatrix1; // 64 x 48
    vector<vector<Plaintext>> weightMatrix2; // 64 x 48
    
    int num_diag = data.slot_count / data.image_size / 2; // should be 8
    int num_matrix_per_row = data.filter_h / num_diag; // should be 48
    int num_matrix_per_col = data.filter_w / num_diag / 2; // should be 8

    int n1;
    int n2;
    if (data.filter_h == 3072 && data.filter_w == 768) {
        n1 = 2;
        n2 = 16;
        if (data.image_size == 64) {
            n1 = 4;
            n2 = 16;
        }
    }
    else if (data.filter_h == 768 && data.filter_w == 3072) {
        n1 = 8;
        n2 = 4;
        if (data.image_size == 64) {
            n1 = 16;
            n2 = 4;
        }
    }
    else if (data.filter_h == 768 && data.filter_w == 768) {
        n1 = 4;
        n2 = 8;
        if (data.image_size == 64) {
            n1 = 8;
            n2 = 8;
        }
    }
    else {
        assert (0);
    }
    cout << "num_matrix_per_col: " << num_matrix_per_col << endl;
    int cot = 0;
    vector<double> temp2(data.slot_count);

    for (int col_ind = 0; col_ind < num_matrix_per_col; col_ind++) {
        int matrix_flag = 0;
        cout << "col_ind: " << col_ind << endl;
        for (int l = 0; l < num_diag; l++) {
            vector<vector<double>> temp_matrix_diag(data.filter_h * data.image_size / data.slot_count);
            vector<Plaintext> temp_matrix_diag2(data.filter_h * data.image_size / data.slot_count);
            int matrix_diag_index = 0;
            for (int i = 0; i < num_matrix_per_row; i++) {
                for (int j = 0; j < num_diag; j++) {
                    if (matrix_flag == 0)
                        std::fill(temp2.begin() + cot*data.image_size, temp2.begin()+(cot+1)*data.image_size, (int64_t)matrix1[i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                    else
                        std::fill(temp2.begin() + cot*data.image_size, temp2.begin()+(cot+1)*data.image_size, (int64_t)matrix2[i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                    cot++;
                    if (cot==(data.slot_count / 2/ data.image_size)) {
                        matrix_flag = (matrix_flag + 1) % 2;
                        std::rotate(temp2.begin(), temp2.begin() + data.slot_count / 2 - (l % n1) * data.image_size, temp2.begin()+data.slot_count / 2);
                    }
                    else if (cot==(data.slot_count / data.image_size)) {
                        matrix_flag = (matrix_flag + 1) % 2;
                        std::rotate(temp2.begin() + (temp2.size() / (data.slot_count / 2) - 1) * data.slot_count / 2, temp2.begin() + temp2.size() - (l % n1) * data.image_size, temp2.end());
                        temp_matrix_diag[matrix_diag_index] = temp2;
                        matrix_diag_index++;
                        cot = 0;
                    }
                }
            }
            // cout << "temp_matrix_diag[].size: " << temp_matrix_diag.size() << endl;
            #pragma omp parallel for num_threads(temp_matrix_diag.size())
            for(int i=0;i<temp_matrix_diag.size();i++){
                Plaintext pt;
                he->encoder->encode(temp_matrix_diag[i], pow(2, he->he_scale), pt);
                temp_matrix_diag2[i] = pt;
            }
            weightMatrix1.push_back(temp_matrix_diag2);
        }
    }
    
    for (int col_ind = 0; col_ind < num_matrix_per_col; col_ind++) {
        int matrix_flag = 0;
        for (int l = 0; l < num_diag; l++) {
            vector<vector<double>> temp_matrix_diag(data.filter_h * data.image_size / data.slot_count);
            vector<Plaintext> temp_matrix_diag2(data.filter_h * data.image_size / data.slot_count);
            int matrix_diag_index = 0;
            for (int i = 0; i < num_matrix_per_row; i++) {
                for (int j = 0; j < num_diag; j++) {
                    if (matrix_flag==0)
                        std::fill(temp2.begin() + cot*data.image_size, temp2.begin()+(cot+1)*data.image_size, (int64_t)matrix2[i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                    else
                        std::fill(temp2.begin() + cot*data.image_size, temp2.begin()+(cot+1)*data.image_size, (int64_t)matrix1[i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                        cot++;
                    if (cot==(data.slot_count / 2/ data.image_size)) {
                        matrix_flag = (matrix_flag + 1) % 2;
                        std::rotate(temp2.begin(), temp2.begin() + data.slot_count / 2 - (l % n1) * data.image_size, temp2.begin()+data.slot_count / 2);
                    }
                    else if (cot==(data.slot_count / data.image_size)) {
                        matrix_flag = (matrix_flag + 1) % 2;
                        std::rotate(temp2.begin() + (temp2.size() / (data.slot_count / 2) - 1) * data.slot_count / 2, temp2.begin() + temp2.size() - (l % n1) * data.image_size, temp2.end());
                        std::rotate(temp2.begin(), temp2.begin() + temp2.size() / 2, temp2.end());
                        temp_matrix_diag[matrix_diag_index] = temp2;
                        matrix_diag_index++;
                        cot = 0;
                    }
                }
            }
            
            #pragma omp parallel for num_threads(temp_matrix_diag.size())
            for(int i=0;i<temp_matrix_diag.size();i++){
                Plaintext pt;
                he->encoder->encode(temp_matrix_diag[i], pow(2, he->he_scale), pt);
                temp_matrix_diag2[i] = pt;
            }
            weightMatrix2.push_back(temp_matrix_diag2);
        }
    }
    
    return std::make_pair(weightMatrix1, weightMatrix2);
}


vector<vector<Plaintext>> Linear_ckks::bert_cross_packing_bias(
    HE_CKKS* he,
    const vector<vector<uint64_t>> &bias, 
    const FCMetadata &data) {
    vector<vector<Plaintext>> cross_bias_packing(6);
    int current_packing = 0, matrix1_pointer = 0, matrix2_pointer = 0;
    while(current_packing < 6) {
        vector<double> temp(data.slot_count, 0);
        int next_flag = 0;
        int row = 0;
        if (matrix1_pointer == data.filter_w && matrix2_pointer == data.filter_w) {
            matrix1_pointer = 0;
            matrix2_pointer = 0;
            current_packing += 1;
            if (current_packing >= 6)
                break;
        }
        while (row < data.slot_count) {
            if (row >= data.slot_count / 2) {
                next_flag = 1;
            }
            if (next_flag == 0) {
                for (int i = 0; i < data.image_size; i++) {
                    temp[row + i] = (int64_t)bias[current_packing * 2][matrix1_pointer];
                }
                matrix1_pointer++;
            }
            else {
                for (int i = 0; i < data.image_size; i++) {
                    temp[row + i] = (int64_t)bias[current_packing * 2 + 1][matrix2_pointer];
                }
                matrix2_pointer++;
            }
            row += data.image_size;
        }
        Plaintext pt;
        he->encoder->encode(temp,pow(2, he->he_scale), pt);
        cross_bias_packing[current_packing].push_back(pt);
    }
    return cross_bias_packing;
}

vector<Plaintext> Linear_ckks::bert_cross_packing_bias_2(
    HE_CKKS* he,
    const uint64_t *matrix, 
    const FCMetadata &data){

    vector<Plaintext> cross_bias_packing(data.image_size * data.filter_w / data.slot_count);
    int matrix_pointer1 = 0;
    int matrix_pointer2 = data.filter_w / 2;
    cout << "num for loop: " << data.image_size * data.filter_w / data.slot_count << endl;
    #pragma omp parallel for
    for (int packing_num = 0; packing_num < data.image_size * data.filter_w / data.slot_count; packing_num++) {
        vector<double> temp(data.slot_count, 0);
        int right_flag = 0;
        int row = 0;
        while (row < data.slot_count) {
            if (row < data.slot_count / 2) {
                for (int i = 0; i < data.image_size; i++) {
                    temp[row + i] = (int64_t)matrix[matrix_pointer1];
                }
                matrix_pointer1++;
            }
            else {
                for (int i = 0; i < data.image_size; i++) {
                    temp[row + i] = (int64_t)matrix[matrix_pointer2];
                }
                matrix_pointer2++;
            }
            row += data.image_size;
        }
        Plaintext pt;
        he->encoder->encode(temp,pow(2, he->he_scale), pt);
        cross_bias_packing[packing_num] = pt;
        temp.clear();
    }
    return cross_bias_packing;
}

// ckks only support rotate_vector, hence we change all rotate_row and rotate_column to rotate_vector
void Linear_ckks::bert_cipher_plain_bsgs(
    HE_CKKS* he,
    const vector<Ciphertext> &cts, 
    const vector<pair<vector<vector<Plaintext>>, 
    vector<vector<Plaintext>>>> &wq_pack, 
    const vector<pair<vector<vector<Plaintext>>,
    vector<vector<Plaintext>>>> &wk_pack, 
    const vector<pair<vector<vector<Plaintext>>, 
    vector<vector<Plaintext>>>> &wv_pack, 
    const vector<vector<Plaintext>> &bq_pack, 
    const vector<vector<Plaintext>> &bk_pack, 
    const vector<vector<Plaintext>> &bv_pack, 
    const FCMetadata &data, 
    vector<Ciphertext> &result) {

    int n1 = 8;
    int n2 = 4;
    if (data.image_size == 64) {
        n1 = 16;
        n2 = 4;
    }
    vector<vector<Ciphertext>> rotatedIR(cts.size(), vector<Ciphertext>(2 * n1));

    int num_diag = data.slot_count / data.image_size / 2;
    int num_matrix_per_col = data.filter_w / num_diag;
    
    #pragma omp parallel for num_threads(32)
    for (int k = 0; k < cts.size() * n1; k++) {
        int i = k % cts.size();
        int j = k / cts.size();
        Ciphertext temp_rot;
        if (j == 0)
            rotatedIR[i][j] = cts[i];
        else {
            he->evaluator->rotate_vector(cts[i], (num_diag - j) * data.image_size, *(he->gal_keys), temp_rot);
            rotatedIR[i][j] = temp_rot;
        }
        he->evaluator->rotate_vector(rotatedIR[i][j],he->poly_modulus_degree/4, *(he->gal_keys), temp_rot);
        rotatedIR[i][j + n1] = temp_rot;
    }
    cout << "OK1" << endl;
    vector<vector<Ciphertext>> temp_results(data.image_size * data.filter_w * 3 * 12 / data.slot_count, vector<Ciphertext>(n2));

    int temp_result_size = data.image_size * data.filter_w * 2 / data.slot_count;
    int omp_thread1 = 2, omp_thread2 = 16;
    if (data.image_size == 64) {
        omp_thread1 = 3;
        omp_thread2 = 8;
    }

    omp_set_nested(1);
    #pragma omp parallel for num_threads(omp_thread1)
    for (int packing_index = 0; packing_index < 6; packing_index++) {
        //compute matrix multiplication
        vector<vector<Ciphertext>> temp_results(temp_result_size * 3, vector<Ciphertext>(n2));
        vector<vector<Ciphertext>> temp_results_q(temp_result_size, vector<Ciphertext>(n2 * cts.size()));
        vector<vector<Ciphertext>> temp_results_k(temp_result_size, vector<Ciphertext>(n2 * cts.size()));
        vector<vector<Ciphertext>> temp_results_v(temp_result_size, vector<Ciphertext>(n2 * cts.size()));
        vector<vector<Plaintext>> enc_weights_q1 = wq_pack[packing_index].first;
        vector<vector<Plaintext>> enc_weights_q2 = wq_pack[packing_index].second;
        vector<vector<Plaintext>> enc_weights_k1 = wk_pack[packing_index].first;
        vector<vector<Plaintext>> enc_weights_k2 = wk_pack[packing_index].second;
        vector<vector<Plaintext>> enc_weights_v1 = wv_pack[packing_index].first;
        vector<vector<Plaintext>> enc_weights_v2 = wv_pack[packing_index].second;

        #pragma omp parallel for num_threads(omp_thread2)
        for (int k = 0; k < cts.size() * n2; k++) {
            int j = k / cts.size();
            int ct_i = k % cts.size();
            for (int l = 0; l < data.image_size * data.filter_w * 2 / data.slot_count; l++) {
                for (int i = 0; i < n1; i++) {
                    Ciphertext ct_l_q, ct_r_q, ct_l_k, ct_r_k, ct_l_v, ct_r_v;
                    // 现场encode weight
                    Plaintext w1, w2, w3,w4,w5,w6;
                    vector<double> vec_w1(data.slot_count,1), vec_w2(data.slot_count,1), vec_w3(data.slot_count,1), vec_w4(data.slot_count,1), vec_w5(data.slot_count,1), vec_w6(data.slot_count,1);
                    he->encoder->encode(vec_w1,rotatedIR[ct_i][i].parms_id(), pow(2,he->he_scale), w1);
                    he->encoder->encode(vec_w2,rotatedIR[ct_i][i + n1].parms_id(), pow(2,he->he_scale), w2);
                    he->encoder->encode(vec_w3,rotatedIR[ct_i][i].parms_id(), pow(2,he->he_scale), w3);
                    he->encoder->encode(vec_w4,rotatedIR[ct_i][i + n1].parms_id(), pow(2,he->he_scale), w4);
                    he->encoder->encode(vec_w5,rotatedIR[ct_i][i].parms_id(), pow(2,he->he_scale), w5);
                    he->encoder->encode(vec_w6,rotatedIR[ct_i][i + n1].parms_id(), pow(2,he->he_scale), w6);

                    he->evaluator->multiply_plain(rotatedIR[ct_i][i], w1, ct_l_q);
                    he->evaluator->multiply_plain(rotatedIR[ct_i][i + n1], w2, ct_r_q);
                    he->evaluator->multiply_plain(rotatedIR[ct_i][i], w3, ct_l_k);
                    he->evaluator->multiply_plain(rotatedIR[ct_i][i + n1], w4, ct_r_k);
                    he->evaluator->multiply_plain(rotatedIR[ct_i][i], w5, ct_l_v);
                    he->evaluator->multiply_plain(rotatedIR[ct_i][i + n1], w6, ct_r_v);
                    // cout << "mul done" << endl;
                    if (i == 0) {
                        he->evaluator->add(ct_l_q, ct_r_q, temp_results_q[l][k]);
                        he->evaluator->add(ct_l_k, ct_r_k, temp_results_k[l][k]);
                        he->evaluator->add(ct_l_v, ct_r_v, temp_results_v[l][k]);
                    }
                    else {
                        Ciphertext temp_add_ct;
                        he->evaluator->add(ct_l_q, ct_r_q, temp_add_ct);
                        he->evaluator->add_inplace(temp_results_q[l][k], temp_add_ct);
                        he->evaluator->add(ct_l_k, ct_r_k, temp_add_ct);
                        he->evaluator->add_inplace(temp_results_k[l][k], temp_add_ct);
                        he->evaluator->add(ct_l_v, ct_r_v, temp_add_ct);
                        he->evaluator->add_inplace(temp_results_v[l][k], temp_add_ct);
                    }
                    // cout << "add done" << endl;
                }

                he->evaluator->rescale_to_next_inplace(temp_results_q[l][k]);
                he->evaluator->rescale_to_next_inplace(temp_results_k[l][k]);
                he->evaluator->rescale_to_next_inplace(temp_results_v[l][k]);
            }
        }
        // cout << "part 1 ok" << endl;
        #pragma omp parallel for num_threads(n2)
        for (int j = 0; j < n2; j++) {
            for (int ct_i = 0; ct_i < cts.size(); ct_i++) {
                for (int l = 0; l < temp_result_size; l++) {
                    if (ct_i == 0) {
                        temp_results[l][j] = temp_results_q[l][j * cts.size() + ct_i];
                        temp_results[l + temp_result_size][j] = temp_results_k[l][j * cts.size() + ct_i];
                        temp_results[l + temp_result_size * 2][j] = temp_results_v[l][j * cts.size() + ct_i];
                    }
                    else {
                        he->evaluator->add_inplace(temp_results[l][j], temp_results_q[l][j * cts.size() + ct_i]);
                        he->evaluator->add_inplace(temp_results[l + temp_result_size][j], temp_results_k[l][j * cts.size() + ct_i]);
                        he->evaluator->add_inplace(temp_results[l + temp_result_size * 2][j], temp_results_v[l][j * cts.size() + ct_i]);
                    }
                }
            }
        }
        // cout << "OK2" << endl;
        #pragma omp parallel for
        for (int l = 0; l < temp_result_size; l++) {
            Ciphertext ct_q, ct_k, ct_v;
            for (int k = 0; k < n2; k++) {
                if (k == 0) {
                    ct_q = temp_results[l][0];
                    ct_k = temp_results[l + temp_result_size][0];
                    ct_v = temp_results[l + temp_result_size * 2][0];
                }
                else {
                    Ciphertext temp_rot_ct;
                    he->evaluator->rotate_vector(temp_results[l][k], -n1 * k * data.image_size, *(he->gal_keys), temp_rot_ct);
                    he->evaluator->add_inplace(ct_q, temp_rot_ct);
                    he->evaluator->rotate_vector(temp_results[l + temp_result_size][k], -n1 * k * data.image_size, *(he->gal_keys), temp_rot_ct);
                    he->evaluator->add_inplace(ct_k, temp_rot_ct);
                    he->evaluator->rotate_vector(temp_results[l + temp_result_size * 2][k], -n1 * k * data.image_size, *(he->gal_keys), temp_rot_ct);
                    he->evaluator->add_inplace(ct_v, temp_rot_ct);
                }
            }
            result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count] = ct_q;
            result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count + data.image_size * data.filter_w * 12 / data.slot_count] = ct_k;
            result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count + data.image_size * data.filter_w * 24 / data.slot_count] = ct_v;
            
            // 现场encode bias
            Plaintext b1,b2,b3;
            vector<double> vec_b1(data.slot_count,1), vec_b2(data.slot_count,1), vec_b3(data.slot_count,1);
            he->encoder->encode(vec_b1,result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count].parms_id(), result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count].scale(), b1);
            he->encoder->encode(vec_b2,result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count + data.image_size * data.filter_w * 12 / data.slot_count].parms_id(), result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count + data.image_size * data.filter_w * 12 / data.slot_count].scale(), b2);
            he->encoder->encode(vec_b3,result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count + data.image_size * data.filter_w * 24 / data.slot_count].parms_id(), result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count + data.image_size * data.filter_w * 24 / data.slot_count].scale(), b3);

            he->evaluator->add_plain_inplace(result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count], b1);
            he->evaluator->add_plain_inplace(result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count + data.image_size * data.filter_w * 12 / data.slot_count], b2);
            he->evaluator->add_plain_inplace(result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count + data.image_size * data.filter_w * 24 / data.slot_count], b3);
        }
    }
    cout << "bert_cipher_plain_bsgs done" << endl;
}

/*
    ckks only support rotate_vector, hence we change all rotate_row and rotate_column to rotate_vector
*/
void Linear_ckks::bert_cipher_plain_bsgs_2(
    HE_CKKS* he,
    const vector<Ciphertext> &cts, 
    const vector<vector<Plaintext>> &enc_mat1, 
    const vector<vector<Plaintext>> &enc_mat2, 
    const vector<Plaintext> &enc_bias, 
    const FCMetadata &data, 
    vector<Ciphertext> &result){
    int n1;
    int n2;
    if (data.filter_h == 3072 && data.filter_w == 768) {
        n1 = 2;
        n2 = 16;
        if (data.image_size == 64) {
            n1 = 4;
            n2 = 16;
        }
    }
    else if (data.filter_h == 768 && data.filter_w == 3072) {
        n1 = 8;
        n2 = 4;
        if (data.image_size == 64) {
            n1 = 16;
            n2 = 4;
        }
    }
    else if (data.filter_h == 768 && data.filter_w == 768) {
        n1 = 4;
        n2 = 8;
        if (data.image_size == 64) {
            n1 = 8;
            n2 = 8;
        }
    }
    else {
        assert (0);
    }
    int num_diag = data.slot_count / data.image_size / 2;

    vector<vector<Ciphertext>> rotatedIR(cts.size(), vector<Ciphertext>(n1 * 2));
    
    #pragma omp parallel for num_threads(32)
    for (int k = 0; k < cts.size() * n1; k++) {
        int i = k % cts.size();
        int j = k / cts.size();
        Ciphertext temp_rot;
        if (j == 0)
            rotatedIR[i][j] = cts[i];
        else {
            he->evaluator->rotate_vector(cts[i], (num_diag - j) * data.image_size, *(he->gal_keys), temp_rot);
            rotatedIR[i][j] = temp_rot;
        }
        he->evaluator->rotate_vector(rotatedIR[i][j], he->poly_modulus_degree/4, *(he->gal_keys), temp_rot);
        rotatedIR[i][j + n1] = temp_rot;
    }
    cout << "OK1" << endl;
    //compute matrix multiplication
    vector<vector<Ciphertext>> temp_results(data.image_size * data.filter_w / data.slot_count, vector<Ciphertext>(n2));
    vector<vector<Ciphertext>> temp_results1(data.image_size * data.filter_w / data.slot_count, vector<Ciphertext>(n2 * cts.size()));
    
    // rotatedIR 48 x 16, enc_mat 64 x 48

    #pragma omp parallel for num_threads(32)
    for (int k = 0; k < cts.size() * n2; k++) {
        int j = k / cts.size();
        int ct_i = k % cts.size();
        for (int l = 0; l < data.image_size * data.filter_w / data.slot_count; l++) {
            for (int i = 0; i < n1; i++) {
                // 现场encode weight, fake实现
                Ciphertext ct1_l;
                Ciphertext ct1_r;
                Plaintext w1,w2;
                vector<double> vec_w1(data.slot_count,1), vec_w2(data.slot_count,1);
                he->encoder->encode(vec_w1,rotatedIR[ct_i][i].parms_id(), pow(2,he->he_scale), w1);
                he->encoder->encode(vec_w2,rotatedIR[ct_i][i + n1].parms_id(), pow(2,he->he_scale), w2);
                // cout << "before scale: " << log2(rotatedIR[ct_i][i].scale()) << endl;
                he->evaluator->multiply_plain(rotatedIR[ct_i][i], w1, ct1_l);
                he->evaluator->multiply_plain(rotatedIR[ct_i][i + n1], w2, ct1_r);
                // cout << "after scale: " << log2(ct1_l.scale()) << endl;
                if (i == 0)
                    he->evaluator->add(ct1_l, ct1_r, temp_results1[l][k]);
                else {
                    Ciphertext temp_add_ct;
                    he->evaluator->add(ct1_l, ct1_r, temp_add_ct);
                    he->evaluator->add_inplace(temp_results1[l][k], temp_add_ct);
                }
            }
        }
    }

    cout << "OK2" << endl;
    // FIXME: optimize this
    #pragma omp parallel for num_threads(n2)
    for (int j = 0; j < n2; j++) {
        for (int ct_i = 0; ct_i < cts.size(); ct_i++) {
            for (int l = 0; l < data.image_size * data.filter_w / data.slot_count; l++) {
                if (ct_i == 0)
                    temp_results[l][j] = temp_results1[l][j * cts.size() + ct_i];
                else
                    he->evaluator->add_inplace(temp_results[l][j], temp_results1[l][j * cts.size() + ct_i]);
            }
        }
        
    }
    cout << "OK3" << endl;
    #pragma omp parallel for num_threads(data.image_size * data.filter_w / data.slot_count)
    for (int l = 0; l < data.image_size * data.filter_w / data.slot_count; l++) {
        Ciphertext ct;
        for (int k = 0; k < n2; k++) {
            if (k == 0)
                ct = temp_results[l][0];
            else {
                Ciphertext temp_rot_ct;
                he->evaluator->rotate_vector(temp_results[l][k], -n1 * k * data.image_size, *(he->gal_keys), temp_rot_ct);
                he->evaluator->add_inplace(ct, temp_rot_ct);
            }
        }
        result[l] = ct;
        // 现场encode bias
        Plaintext b;
        vector<double> vec_b(data.slot_count,1);
        he->encoder->encode(vec_b,result[l].parms_id(), result[l].scale(), b);
        he->evaluator->add_plain_inplace(result[l], b);
    }

    parms_id_type parms_id = result[0].parms_id();
    shared_ptr<const SEALContext::ContextData> context_data = he->context->get_context_data(parms_id);
    cout << "OK4" << endl;
    cout << "scale: " << log2(result[0].scale()) << endl;
    #pragma omp parallel for
    for (int i = 0; i < result.size(); i++) {
        flood_ciphertext(result[i], context_data, SMUDGING_BITLEN_bert2);
        // he->evaluator->rescale_to_next_inplace(result[i]);
    }
}

// tianshi: need to understand!!
// Compute QK^T
// 1. rotate rhs for 128 x 1-step rotations
// 2. mult with lhs (producing 128 cts)
// 3. for each of the 128 cts, rotate for log(32) times, sum together + 1 time batch rotation
// 4. mult masks (1, 0 (x31), 1, 0 (x31), ... ) and sum together (do the first 32 (1st batch) and then the second batch).

void Linear_ckks::bert_cipher_cipher_cross_packing(
	HE_CKKS* he,
	const FCMetadata &data, 
	const vector<Ciphertext> &Cipher_plain_result, 
	const vector<Plaintext> &cross_masks, 
	vector<Ciphertext> &results) {
    uint32_t num_rot=0;
    int packing_gap = data.image_size * data.filter_w / data.slot_count * 3;
    int temp_result_size = data.image_size * data.filter_w * 2 / data.slot_count;

    #pragma omp parallel for num_threads(2)
    for (int packing_index = 0; packing_index < 6; packing_index++) {
        vector<Ciphertext> rotation_results(data.image_size * 2);
        for (int l = 0; l < temp_result_size; l++) {
            Ciphertext Qi = Cipher_plain_result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count];
            Ciphertext Ki = Cipher_plain_result[l + packing_index * data.image_size * data.filter_w * 2 / data.slot_count + data.image_size * data.filter_w * 12 / data.slot_count];
            #pragma omp parallel for num_threads(16)
            for (int i = 0; i < data.image_size; i++) {
                vector<Ciphertext> temp_mult = rotation_by_one_depth3(he, data, Ki, i);
                if (l == 0) {
                    he->evaluator->multiply(Qi, temp_mult[0], rotation_results[i]);
                    he->evaluator->multiply(Qi, temp_mult[1], rotation_results[i + data.image_size]);
                }
                else {
                    Ciphertext temp_qk;
                    he->evaluator->multiply(Qi, temp_mult[0], temp_qk);
                    he->evaluator->add_inplace(rotation_results[i], temp_qk);
                    he->evaluator->multiply(Qi, temp_mult[1], temp_qk);
                    he->evaluator->add_inplace(rotation_results[i + data.image_size], temp_qk);
                }

            }
        }
        // cout << "part 1 ok" << endl;
        #pragma omp parallel for num_threads(16)
        for (int i = 0; i < data.image_size * 2; i++) {
            he->evaluator->relinearize_inplace(rotation_results[i], *(he->relin_keys));
            he->evaluator->rescale_to_next_inplace(rotation_results[i]);
        }
        // cout << "part 2 ok" << endl;
        int local_rotation = std::ceil(std::log2(data.slot_count / data.image_size / 2));

        #pragma omp parallel for num_threads(16)
        for (int i = 0; i < data.image_size * 2; i++) {
            for (int k = 0; k < local_rotation; k++) {
                Ciphertext temp2;
                he->evaluator->rotate_vector(rotation_results[i], (int32_t) pow(2, k) * data.image_size, *(he->gal_keys), temp2);
                he->evaluator->add_inplace(rotation_results[i], temp2);
                num_rot++;
            }
            // 现场encode mask
            // Plaintext mask;
            // vector<double> vec_mask(data.slot_count,1.0);
            // he->encoder->encode(vec_mask, rotation_results[i].parms_id(), 1, mask);
            // he->evaluator->multiply_plain_inplace(rotation_results[i], mask);
        }
        // cout << "part 3 ok" << endl;
        int num_cts_per_res = data.image_size * data.image_size * 2 / data.slot_count; // 1 or 4
        int num_col_per_ct = data.slot_count / 2 / data.image_size; // 64 or 32

        #pragma omp parallel for num_threads(num_cts_per_res)
        for (int i = 0; i < num_cts_per_res; i++) {
            he->evaluator->add(rotation_results[num_col_per_ct * i], rotation_results[num_col_per_ct * i + data.image_size], results[packing_index * num_cts_per_res + i]);
            for (int j = 1; j < num_col_per_ct; j++) {
                he->evaluator->add_inplace(results[packing_index * num_cts_per_res + i], rotation_results[num_col_per_ct * i + j]);
                he->evaluator->add_inplace(results[packing_index * num_cts_per_res + i], rotation_results[num_col_per_ct * i + j + data.image_size]);
            }
        }
        // cout << "part 4 ok" << endl;
    }
    cout << "num rotations: " << num_rot << endl;
    cout << "bert_cipher_cipher_cross_packing done" << endl;
}

vector<Ciphertext> Linear_ckks::rotation_by_one_depth3(
	HE_CKKS* he,
	const FCMetadata &data, 
	const Ciphertext &ct, 
	int k) {

    int m = -(data.image_size - k);
    Ciphertext ct1;
    Ciphertext ct2;
    he->evaluator->rotate_vector(ct, k, *(he->gal_keys), ct1);
    he->evaluator->rotate_vector(ct, m, *(he->gal_keys), ct2);

    return {ct1, ct2};
}

// column-wise packing
vector<Ciphertext> Linear_ckks::bert_efficient_preprocess_vec(
	HE_CKKS* he,
	vector<int64_t> &input, 
	const FCMetadata &data) {

    vector<Ciphertext> cts((data.image_size * data.filter_h) / data.slot_count);

    #pragma omp parallel for
    for (int i = 0; i < (data.image_size * data.filter_h) / data.slot_count; i++)
    {
        vector<double> pod_matrix(data.slot_count, 0);
        pod_matrix = vector<double>(input.begin() + i * data.slot_count, input.begin() + (i+1) * data.slot_count);
        Ciphertext ct;
        Plaintext pt;
        he->encoder->encode(pod_matrix,pow(2, he->he_scale), pt);
        he->encryptor->encrypt(pt, ct);
        cts[i] = ct;
    }
    return cts;
}

uint64_t* Linear_ckks::bert_cross_packing_postprocess(
    HE_CKKS* he,
    vector<Ciphertext> &cts, 
    const FCMetadata &data) {
    uint64_t *result = new uint64_t[data.image_size*data.image_size*12];
    int num_cts_per_mat = data.image_size * data.image_size / data.slot_count;
    for (int packing_num = 0; packing_num < 12; packing_num++) {
        for (int i = 0; i < num_cts_per_mat; i++) {
            vector<double> plain(data.slot_count, 0);
            Plaintext tmp;
            he->decryptor->decrypt(cts[i + packing_num * num_cts_per_mat], tmp);
            he->encoder->decode(tmp, plain);

            #pragma omp parallel for
            for (int row = 0; row < data.slot_count; row++) {
                int j = row / data.image_size;
                int k = row % data.image_size;
                if (j < 32) { // k, (k + j) % 128
                    result[k + ((k + j + i * 32) % data.image_size) * data.image_size + packing_num * data.image_size * data.image_size] = plain[row];
                }
                else if (j == 32 && i == 0) { // (64 + k) % 128, k
                    result[((k + 64) % data.image_size) + k * data.image_size + packing_num * data.image_size * data.image_size] = plain[row];
                }
                else { // (k - 32 + j) % 128, k
                    result[k * data.image_size + (k + j - 32 + i * 32) % 128 + packing_num * data.image_size * data.image_size] = plain[row];
                }
            }
        }
    }
    return result;
}


void Linear_ckks::plain_cross_packing_postprocess(
    uint64_t* input, 
    uint64_t * output,
    bool col_packing,
    const FCMetadata &data) {

    int cts_size = 12*data.image_size*data.image_size / data.slot_count;

    int num_cts_per_res = data.image_size * data.image_size * 2 / data.slot_count; // 1 or 4
    int num_col_per_ct = data.slot_count / 2 / data.image_size; // 64 or 32

    omp_set_nested(1);
    #pragma omp parallel for
    for (int ct_ind = 0; ct_ind < cts_size; ct_ind++) {
        // cout << "ct_ind: " << ct_ind << endl;
        vector<uint64_t> plain(&input[ct_ind* data.slot_count], &input[(ct_ind + 1) * data.slot_count]);

        int current_col = ct_ind % num_cts_per_res;
        int current_packing = ct_ind / num_cts_per_res;
        
        if (col_packing) {
            #pragma omp parallel for
            for (int row = 0; row < data.slot_count; row++) {
                int j = row / data.image_size + current_col * num_col_per_ct;
                int k = row % data.image_size;
                int next_flag = 0;
                if (row >= data.slot_count / 2) {
                    next_flag = data.image_size * data.image_size;
                    j -= data.slot_count / 2 / data.image_size;
                }
                output[k + (k + j) % data.image_size * data.image_size + current_packing * data.image_size * data.image_size * 2 + next_flag] = plain[row];
            }
        }
        else {
            #pragma omp parallel for
            for (int row = 0; row < data.slot_count; row++) {
                int j = row / data.image_size + current_col * num_col_per_ct;
                int k = row % data.image_size;
                int next_flag = 0;
                if (row >= data.slot_count / 2) {
                    next_flag = data.image_size * data.image_size;
                    j -= data.slot_count / 2 / data.image_size;
                }
                output[k * data.image_size + (k + j) % data.image_size + current_packing * data.image_size * data.image_size * 2 + next_flag] = plain[row];
            }
        }
    }
}

void Linear_ckks::plain_cross_packing_postprocess_v(
    uint64_t* input, 
    uint64_t * output,
    bool col_packing,
    const FCMetadata &data) {
    int cts_size = 12*data.image_size*OUTPUT_DIM / data.slot_count; 
    int num_V_per_cts = data.slot_count / (data.image_size * data.filter_w);

    omp_set_nested(1);
    #pragma omp parallel for
    for (int ct_ind = 0; ct_ind < cts_size; ct_ind++) {
        vector<uint64_t> plain(&input[ct_ind* data.slot_count], &input[(ct_ind + 1) * data.slot_count]);

        if (col_packing) {
            #pragma omp parallel for
            for (int row = 0; row < data.slot_count; row++) {
                int j = row / data.image_size;
                int k = row % data.image_size;
                if (row >= data.slot_count / 2) {
                    j -= data.slot_count / data.image_size / 2;
                    j += data.filter_w;
                }
                if (num_V_per_cts == 1) {
                    output[k + j * data.image_size + (ct_ind / 2) * data.image_size * data.filter_w * 2 + (ct_ind % 2) * data.image_size * data.filter_w / 2] = plain[row];
                }
                else if (num_V_per_cts == 2) {
                    output[k + j * data.image_size + ct_ind * data.image_size * data.filter_w * 2] = plain[row];
                }
            }
        }
        else {
            #pragma omp parallel for
            for (int row = 0; row < data.slot_count; row++) {
                int j = row / data.image_size;
                int k = row % data.image_size;
                int next_flag = 0;
                if (row >= data.slot_count / 2) {
                    j -= data.slot_count / data.image_size / 2;
                    next_flag = data.filter_w * data.image_size;
                }
                if (num_V_per_cts == 1) {
                    output[k * data.filter_w + j + next_flag + (ct_ind / 2) * data.image_size * data.filter_w * 2 + (ct_ind % 2) * data.filter_w / 2] = plain[row];
                }
                else if (num_V_per_cts == 2) {
                    output[k * data.filter_w + j + next_flag + ct_ind * data.image_size * data.filter_w * 2] = plain[row];
                }
            }
        }
    }
}

void Linear_ckks::plain_col_packing_preprocess(
    uint64_t* input, 
    uint64_t * output,
    uint64_t plain_mod,
    int input_dim,
    int common_dim){
    #pragma omp parallel for
    for (int j = 0; j < common_dim; j++)
            for (int i = 0; i < input_dim; i++)
                output[j*input_dim + i] = input[i*common_dim +j];
}

void Linear_ckks::plain_col_packing_preprocess_vec(
    vector<vector<uint64_t>> input, 
    uint64_t * output,
    uint64_t plain_mod,
    int input_dim,
    int common_dim){
    for (int j = 0; j < common_dim; j++)
            for (int i = 0; i < input_dim; i++)
                output[j*input_dim + i] = input[i][j];
}

void Linear_ckks::plain_col_packing_postprocess(
    uint64_t* input, 
    uint64_t * output,
    bool col_packing,
    const FCMetadata &data){
    for (int i = 0; i < data.image_size * data.filter_w / data.slot_count; i++) {
        vector<uint64_t> plain(&input[i* data.slot_count], &input[(i+1)* data.slot_count]);
        if (col_packing) {
            #pragma omp parallel for
            for (int row = 0; row < data.slot_count; row++) {
                int j = row / data.image_size;
                int k = row % data.image_size;
                if (row >= data.slot_count / 2) {
                    j -= data.slot_count / data.image_size / 2;
                    j += data.filter_w / 2;
                }
                output[k + j * data.image_size + i * data.slot_count / 2] = plain[row];
            }
        }
        else {
            #pragma omp parallel for
            for (int row = 0; row < data.slot_count; row++) {
                int j = row / data.image_size;
                int k = row % data.image_size;
                if (row >= data.slot_count / 2) {
                    j -= data.slot_count / data.image_size / 2;
                    j += data.filter_w / 2;
                }
                j += i * data.slot_count / data.image_size / 2;
                output[k * data.filter_w + j] = plain[row];
            }
        }
    }
}

vector<vector<uint64_t>> Linear_ckks::concat_vec(
    uint64_t* att,
    int n,
    int dim1,
    int dim2){
    
    vector<vector<uint64_t>> res;
    for(int j = 0; j < dim1; j++){
        vector<uint64_t> row;
        for(int i = 0; i < n; i++){
            row.insert(row.end(), &att[i*dim1*dim2 + j*dim2], &att[i*dim1*dim2 + j*dim2 + dim2]);
        }
        res.push_back(row);
    }
    return res;
}

void Linear_ckks::concat( 
    uint64_t* input,
    uint64_t* output,
    int n,
    int dim1,
    int dim2){

    for(int j = 0; j < dim1; j++){
        for(int i = 0; i < n; i++){
            memcpy(&output[j*n*dim2 + i*dim2], &input[i*dim1*dim2 + j*dim2], dim2*sizeof(uint64_t));
        }
    }
}

// matrix is row-packed with 12 * 128 rows and 128 cols
vector<Ciphertext> Linear_ckks::preprocess_softmax_s1(HE_CKKS* he, uint64_t* matrix, const FCMetadata &data) {
    int num_cts_per_res = data.image_size * data.image_size * 2 / data.slot_count; // 1 or 4
    int num_col_per_ct = data.slot_count / 2 / data.image_size; // 64 or 32

    int total_cts = 12 * data.image_size * data.image_size / data.slot_count;
    vector<Ciphertext> enc_softmax(total_cts);

    #pragma omp parallel for
    for (int ct_ind = 0; ct_ind < total_cts; ct_ind++) {
        int current_col = ct_ind % num_cts_per_res;
        int current_packing = ct_ind / num_cts_per_res;
        vector<double> pod_matrix(data.slot_count);

        for (int row = 0; row < data.slot_count; row++) {
            int j = row / data.image_size + current_col * num_col_per_ct;
            int k = row % data.image_size;
            int next_flag = 0;
            if (row >= data.slot_count / 2) {
                next_flag = data.image_size * data.image_size;
                j -= data.slot_count / 2 / data.image_size;
            }
            pod_matrix[row] = (int64_t)matrix[k * data.image_size + j + current_packing * data.image_size * data.image_size * 2 + next_flag];
        }
        Ciphertext ct;
        Plaintext pt;
        he->encoder->encode(pod_matrix,pow(2, he->he_scale), pt);
        he->encryptor->encrypt(pt, ct);
        enc_softmax[ct_ind] = ct;
    }
    return enc_softmax;
}

void Linear_ckks::client_S1_V_R(HE_CKKS* he, const uint64_t *softmax_s1, uint64_t* V, uint64_t* result, const FCMetadata &data){
    int total_packing = 12;
    for (int packing_num = 0; packing_num < total_packing; packing_num++) {
        #pragma omp parallel for
        for(int i = 0; i < data.image_size; i++) {
            for(int j = 0; j < data.filter_w; j++) {
                result[packing_num * data.image_size * data.filter_w + i + j * data.image_size] = 0;
                for(int k = 0; k < data.image_size; k++) {
                    result[packing_num * data.image_size * data.filter_w + i + j * data.image_size] += softmax_s1[packing_num * data.image_size * data.image_size + i * data.image_size + k] * V[k + j * data.image_size + data.image_size * data.filter_w * packing_num];
                    result[packing_num * data.image_size * data.filter_w + i + j * data.image_size] = result[packing_num * data.image_size * data.filter_w + i + j * data.image_size];
                }
            }
        }
    }
}

void Linear_ckks::bert_postprocess_V(HE_CKKS* he, uint64_t* input, uint64_t* result, const FCMetadata &data, const bool &col_packing){
    // int total_packing = cts.size() * data.slot_count / (data.image_size * data.filter_w);
    int num_cts_per_mat_V = data.image_size * data.filter_w / data.slot_count;
    for (int packing_num = 0; packing_num < 12; packing_num++) {
        for (int i = 0; i < num_cts_per_mat_V; i++) {
            // vector<uint64_t> plain(data.slot_count, 0ULL);
            // Plaintext pt;
            // he->decryptor->decrypt(cts[i + packing_num * num_cts_per_mat_V], pt);
            // he->encoder->decode(pt, plain);

            int offset = i + packing_num * num_cts_per_mat_V;
            vector<uint64_t> plain(&input[offset* data.slot_count], &input[offset* data.slot_count + data.slot_count]);

            if (col_packing) {
                #pragma omp parallel for
                for (int row = 0; row < data.slot_count; row++) {
                    int j = row / data.image_size;
                    int k = row % data.image_size;
                    if (row >= data.slot_count / 2) {
                        j -= data.slot_count / data.image_size / 2;
                        j += data.filter_w / 2;
                    }
                    result[k + j * data.image_size + i * data.slot_count / 2 + packing_num * data.image_size * data.filter_w] = plain[row];
                }
            }
            else {
                #pragma omp parallel for
                for (int row = 0; row < data.slot_count; row++) {
                    int j = row / data.image_size;
                    int k = row % data.image_size;
                    if (row >= data.slot_count / 2) {
                        j -= data.slot_count / data.image_size / 2;
                        j += data.filter_w / 2;
                    }
                    j += i * data.slot_count / data.image_size / 2;
                    result[k * data.filter_w + j + packing_num * data.image_size * data.filter_w] = plain[row];
                }
            }
        }
    }
}

void Linear_ckks::bert_postprocess_V_enc(HE_CKKS* he, vector<Ciphertext> cts, uint64_t* result, const FCMetadata &data, const bool &col_packing){
    // int total_packing = cts.size() * data.slot_count / (data.image_size * data.filter_w);
    int num_cts_per_mat_V = data.image_size * data.filter_w / data.slot_count;
    for (int packing_num = 0; packing_num < 12; packing_num++) {
        for (int i = 0; i < num_cts_per_mat_V; i++) {
            vector<double> plain(data.slot_count, 0);
            Plaintext pt;
            he->decryptor->decrypt(cts[i + packing_num * num_cts_per_mat_V], pt);
            he->encoder->decode(pt, plain);

            if (col_packing) {
                #pragma omp parallel for
                for (int row = 0; row < data.slot_count; row++) {
                    int j = row / data.image_size;
                    int k = row % data.image_size;
                    if (row >= data.slot_count / 2) {
                        j -= data.slot_count / data.image_size / 2;
                        j += data.filter_w / 2;
                    }
                    result[k + j * data.image_size + i * data.slot_count / 2 + packing_num * data.image_size * data.filter_w] = plain[row];
                }
            }
            else {
                #pragma omp parallel for
                for (int row = 0; row < data.slot_count; row++) {
                    int j = row / data.image_size;
                    int k = row % data.image_size;
                    if (row >= data.slot_count / 2) {
                        j -= data.slot_count / data.image_size / 2;
                        j += data.filter_w / 2;
                    }
                    j += i * data.slot_count / data.image_size / 2;
                    result[k * data.filter_w + j + packing_num * data.image_size * data.filter_w] = plain[row];
                }
            }
        }
    }
}

vector<vector<vector<uint64_t>>> softmax_mask_ckks(const FCMetadata &data) {
    vector<vector<vector<uint64_t>>> mask(2, vector<vector<uint64_t>>(data.image_size, vector<uint64_t>(data.image_size)));
    #pragma omp parallel for
    for (int i = 0; i < data.image_size; i++) {
        vector<uint64_t> mask1(data.image_size, 0ULL);
        vector<uint64_t> mask2(data.image_size, 0ULL);
        for (int j = 0; j < data.image_size - i; j++) {
            mask1[j] = 1;
        }
        for (int j = data.image_size - i; j < data.image_size; j++) {
            mask2[j] = 1;
        }
        mask[0][i] = mask1;
        mask[1][i] = mask2;
    }
    return mask;
}

vector<vector<vector<Plaintext>>> 
Linear_ckks::preprocess_softmax_s2(HE_CKKS* he, const uint64_t *matrix, const FCMetadata &data){
    
    auto mask = softmax_mask_ckks(data);

    int num_diag = data.image_size;
    int num_diag_per_ct = data.slot_count / data.image_size / 2;
    vector<vector<vector<Plaintext>>> s2_pack(6);

    #pragma omp parallel for num_threads(2)
    for (int packing_ind = 0; packing_ind < 6; packing_ind++) {
        vector<vector<Plaintext>> weightMatrix1(2, vector<Plaintext>(num_diag));
        #pragma omp parallel for
        for (int diag_ind = 0; diag_ind < num_diag; diag_ind++) {
            vector<int64_t> temp2, temp3;
            vector<double> r1(data.image_size), r2(data.image_size), r3(data.image_size), r4(data.image_size);
            for (int j = 0; j < num_diag; j++) {
                temp2.push_back(matrix[((j + diag_ind) % num_diag) + j * data.image_size + packing_ind * 2 * data.image_size * data.image_size]);
                temp3.push_back(matrix[((j + diag_ind) % num_diag) + j * data.image_size + (packing_ind * 2 + 1) * data.image_size * data.image_size]);
            }
            // std::rotate(temp2.begin(), temp2.begin() + temp2.size() - diag_ind, temp2.end());
            std::transform(temp2.begin(), temp2.end(), mask[0][diag_ind].begin(), r1.begin(), std::multiplies<uint64_t>());
            std::transform(temp2.begin(), temp2.end(), mask[1][diag_ind].begin(), r2.begin(), std::multiplies<uint64_t>());
            std::transform(temp3.begin(), temp3.end(), mask[0][diag_ind].begin(), r3.begin(), std::multiplies<uint64_t>());
            std::transform(temp3.begin(), temp3.end(), mask[1][diag_ind].begin(), r4.begin(), std::multiplies<uint64_t>());
            for (int j = 0; j < std::log2(num_diag_per_ct); j++) {
                r1.reserve(r1.size() + distance(r1.begin(), r1.end()));
                r1.insert(r1.end(), r1.begin(), r1.end());
                r2.reserve(r2.size() + distance(r2.begin(), r2.end()));
                r2.insert(r2.end(), r2.begin(), r2.end());
                r3.reserve(r3.size() + distance(r3.begin(), r3.end()));
                r3.insert(r3.end(), r3.begin(), r3.end());
                r4.reserve(r4.size() + distance(r4.begin(), r4.end()));
                r4.insert(r4.end(), r4.begin(), r4.end());
            }
            r1.insert(r1.end(), r3.begin(), r3.end());
            r2.insert(r2.end(), r4.begin(), r4.end());

            Plaintext pt;
            he->encoder->encode(r1, pow(2, he->he_scale), pt);
            weightMatrix1[0][diag_ind] = pt;
            he->encoder->encode(r2, pow(2, he->he_scale), pt);
            weightMatrix1[1][diag_ind] = pt;
        }
        s2_pack[packing_ind] = weightMatrix1;
    }
    return s2_pack;
}

vector<vector<vector<Plaintext>>> Linear_ckks::bert_softmax_v_packing_single_matrix(
    HE_CKKS* he,
    const vector<vector<vector<uint64_t>>> &weights, 
    const FCMetadata &data) 
    {
    vector<vector<vector<Plaintext>>> result(6);
    int num_diag = data.slot_count / data.image_size / 2;

    int n1 = 4;
    int n2 = 8;
    if (data.image_size == 64) {
        n1 = 8;
        n2 = 8;
    }

    int weight_height = data.image_size;
    int num_matrix_per_row = weight_height / num_diag; // 1 or 4
    int num_matrix_per_col = data.filter_w / num_diag; // 1 or 2

    omp_set_nested(1);
    #pragma omp parallel for num_threads(2)
    for (int packing_ind = 0; packing_ind < 6; packing_ind++) {
        vector<vector<Plaintext>> weightMatrix1(num_matrix_per_col * num_diag);
        for (int col_ind = 0; col_ind < num_matrix_per_col; col_ind++) {
            #pragma omp parallel for
            for (int l = 0; l < num_diag; l++) {
                vector<double> temp2, temp3;
                vector<Plaintext> temp_matrix_diag(num_matrix_per_row);
                int matrix_diag_index = 0;
                for (int i = 0; i < num_matrix_per_row; i++) {
                    for (int j = 0; j < num_diag; j++) {
                        for (int k = 0; k < data.image_size; k++) {
                                temp2.push_back((int64_t)weights[packing_ind * 2][i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                                temp3.push_back((int64_t)weights[packing_ind * 2 + 1][i * num_diag + j][(j + l) % num_diag + col_ind * num_diag]);
                        }
                    }
                    std::rotate(temp2.begin(), temp2.begin() + temp2.size() - (l % n1) * data.image_size, temp2.end());
                    std::rotate(temp3.begin(), temp3.begin() + temp3.size() - (l % n1) * data.image_size, temp3.end());
                    temp2.insert(temp2.end(), temp3.begin(), temp3.end());
                    Plaintext pt;
                    he->encoder->encode(temp2, pow(2, he->he_scale), pt);
                    temp_matrix_diag[matrix_diag_index] = pt;
                    matrix_diag_index++;
                    temp2.clear();
                    temp3.clear();
                }
                weightMatrix1[col_ind * num_diag + l] = temp_matrix_diag;
            }
        }
        result[packing_ind] = weightMatrix1;
    }
    return result;
}

vector<vector<vector<Plaintext>>>
Linear_ckks::preprocess_softmax_v_r(HE_CKKS* he, const uint64_t *matrix, const FCMetadata &data) {
    vector<vector<vector<uint64_t>>> weights_r(12, vector<vector<uint64_t>>(data.image_size, vector<uint64_t>(data.filter_w)));

    for (int packing_ind = 0; packing_ind < 12; packing_ind++) {
        #pragma omp parallel for
        for (int i = 0; i < data.image_size; i++) {
            for (int j = 0; j < data.filter_w; j++) {
                weights_r[packing_ind][i][j] = matrix[i + j * data.image_size + packing_ind * data.image_size * data.filter_w];
            }
        }
    }
    vector<vector<vector<Plaintext>>> R_pack = bert_softmax_v_packing_single_matrix(he, weights_r, data);
    return R_pack;
}

void Linear_ckks::bert_softmax_V(HE_CKKS* he, vector<Ciphertext> &softmax_s1, vector<vector<vector<Plaintext>>> &softmax_s2, vector<Ciphertext> &V, vector<vector<vector<Plaintext>>> &R, const FCMetadata &data, vector<Ciphertext> &result) {
    // FIXME: pack R according to ours ctxpt
    // FIXME: compute softmax_s1 x R

    // #pragma omp parallel for
    // for (int i = 0; i < V.size(); i++) {
    //     he->evaluator->mod_switch_to_next_inplace(V[i]);
    // }
    int n1 = 4;
    int n2 = 8;
    if (data.image_size == 64) {
        n1 = 8;
        n2 = 8;
    }

    #pragma omp parallel for num_threads(2)
    for (int packing_ind = 0; packing_ind < 6; packing_ind++) {
        int num_diag = data.slot_count / data.image_size / 2;
        int num_matrix_per_row = data.image_size / num_diag; // 1 or 4
        int num_matrix_per_col = data.filter_w / num_diag; // 1 or 2

        vector<vector<Plaintext>> R1 = R[packing_ind];
        vector<vector<Ciphertext>> rotatedIR(num_matrix_per_row);

        #pragma omp parallel for
        for (int i = 0; i < num_matrix_per_row; i++) {
            vector<Ciphertext> tmp;
            tmp.push_back(softmax_s1[packing_ind * num_matrix_per_row + i]);

            for (int j = 1; j < n1; j++) {
                Ciphertext temp_rot;
                he->evaluator->rotate_rows(softmax_s1[packing_ind * num_matrix_per_row + i], (num_diag - j) * data.image_size, *(he->gal_keys), temp_rot);
                tmp.push_back(temp_rot);
            }

            rotatedIR[i] = tmp;
            tmp.clear();
        }

        //compute matrix multiplication
        vector<vector<Ciphertext>> temp_results(num_matrix_per_col, vector<Ciphertext>(n2));
        vector<vector<Ciphertext>> temp_results1(num_matrix_per_col, vector<Ciphertext>(n2 * num_matrix_per_row));

        #pragma omp parallel for
        for (int k = 0; k < num_matrix_per_row * n2; k++) {
            int j = k / num_matrix_per_row;
            int ct_i = k % num_matrix_per_row;
            for (int l = 0; l < num_matrix_per_col; l++) {
                for (int i = 0; i < n1; i++) {
                    Ciphertext ct1_l;
                    he->evaluator->multiply_plain(rotatedIR[ct_i][i], R1[n1 * j + i + l * num_diag][ct_i], ct1_l);
                    if (i == 0)
                        temp_results1[l][k] = ct1_l;
                    else {
                        he->evaluator->add_inplace(temp_results1[l][k], ct1_l);
                    }
                }
            }
        }

        #pragma omp parallel for
        for (int j = 0; j < n2; j++) {
            for (int ct_i = 0; ct_i < num_matrix_per_row; ct_i++) {
                for (int l = 0; l < num_matrix_per_col; l++) {
                    if (ct_i == 0)
                        temp_results[l][j] = temp_results1[l][j * num_matrix_per_row + ct_i];
                    else
                        he->evaluator->add_inplace(temp_results[l][j], temp_results1[l][j * num_matrix_per_row + ct_i]);
                }
            }
        }

        #pragma omp parallel for
        for (int l = 0; l < num_matrix_per_col; l++) {
            Ciphertext ct;
            for (int k = 0; k < n2; k++) {
                if (k == 0)
                    ct = temp_results[l][0];
                else {
                    Ciphertext temp_rot_ct;
                    he->evaluator->rotate_rows(temp_results[l][k], -n1 * k * data.image_size, *(he->gal_keys), temp_rot_ct);
                    he->evaluator->add_inplace(ct, temp_rot_ct);
                }
            }
            result[packing_ind * data.image_size * data.filter_w * 2 / data.slot_count + l] = ct;
        }

        // FIXME: pack softmax_s2 according to gazelle
        // FIXME: compute softmax_s2 x V

        num_diag = data.image_size;
        for (int ct_ind = 0; ct_ind < num_matrix_per_col; ct_ind++) {
            vector<Ciphertext> rotation_results(num_diag);

            #pragma omp parallel for
            for (int i = 0; i < num_diag; i++) {
                Ciphertext temp1;
                Ciphertext temp2;
                vector<Ciphertext> temp_mult = rotation_by_one_depth3(he, data, V[packing_ind * num_matrix_per_col + ct_ind], i);
                he->evaluator->multiply_plain(temp_mult[0], softmax_s2[packing_ind][0][i], temp1);
                he->evaluator->multiply_plain(temp_mult[1], softmax_s2[packing_ind][1][i], temp2);
                he->evaluator->add(temp1, temp2, rotation_results[i]);
            }
            for (int i = 0; i < num_diag; i++) {
                he->evaluator->add_inplace(result[packing_ind * data.image_size * data.filter_w * 2 / data.slot_count + ct_ind], rotation_results[i]);
            }
            rotation_results.clear();
        }
    }

    parms_id_type parms_id = result[0].parms_id();
    shared_ptr<const SEALContext::ContextData> context_data = he->context->get_context_data(parms_id);

    #pragma omp parallel for
    for (int i = 0; i < result.size(); i++) {
        flood_ciphertext(result[i], context_data, SMUDGING_BITLEN_bert1);
        he->evaluator->mod_switch_to_next_inplace(result[i]);
        he->evaluator->mod_switch_to_next_inplace(result[i]);
    }
}

vector<Ciphertext> Linear_ckks::w_ln(HE_CKKS* he, vector<Ciphertext> ln, vector<Plaintext> w){
    int cts_size = ln.size();
    vector<Ciphertext> result(cts_size);
    #pragma omp parallel for
    for (int i = 0; i < cts_size; i++) {
        he->evaluator->multiply_plain(ln[i], w[i], result[i]);
    }

    parms_id_type parms_id = result[0].parms_id();
    shared_ptr<const SEALContext::ContextData> context_data = he->context->get_context_data(parms_id);

    #pragma omp parallel for
    for (int i = 0; i < cts_size; i++) {
        flood_ciphertext(result[i], context_data, SMUDGING_BITLEN_bert3);
        he->evaluator->mod_switch_to_next_inplace(result[i]);
        he->evaluator->mod_switch_to_next_inplace(result[i]);
    }

    return result;
}


void Linear_ckks::softmax_v(
    HE_CKKS* he,
    vector<vector<vector<Ciphertext>>> &softmax_s2, 
    vector<Ciphertext> &V, 
    const FCMetadata &data, 
    vector<Ciphertext> &result){
    #pragma omp parallel for num_threads(2)
    for (int packing_ind = 0; packing_ind < 6; packing_ind++) {
        int num_diag = data.slot_count / data.image_size / 2;
        int num_matrix_per_col = data.filter_w / num_diag; // 1 or 2

        // FIXME: pack softmax_s2 according to gazelle
        // FIXME: compute softmax_s2 x V

        num_diag = data.image_size;
        for (int ct_ind = 0; ct_ind < num_matrix_per_col; ct_ind++) {
            vector<Ciphertext> rotation_results(num_diag);

            #pragma omp parallel for
            for (int i = 0; i < num_diag; i++) {
                Ciphertext temp1;
                Ciphertext temp2;
                vector<Ciphertext> temp_mult = rotation_by_one_depth3(he, data, V[packing_ind * num_matrix_per_col + ct_ind], i);
                he->evaluator->multiply(temp_mult[0], softmax_s2[packing_ind][0][i], temp1);
                he->evaluator->multiply(temp_mult[1], softmax_s2[packing_ind][1][i], temp2);
                he->evaluator->add(temp1, temp2, rotation_results[i]);
            }
            result[packing_ind * data.image_size * data.filter_w * 2 / data.slot_count + ct_ind] = rotation_results[0];
            for (int i = 1; i < num_diag; i++) {
                he->evaluator->add_inplace(result[packing_ind * data.image_size * data.filter_w * 2 / data.slot_count + ct_ind], rotation_results[i]);
            }
            rotation_results.clear();
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < result.size(); i++) {
        he->evaluator->relinearize_inplace(result[i], *(he->relin_keys));
        // he->evaluator->mod_switch_to_next_inplace(result[i]);
        // he->evaluator->mod_switch_to_next_inplace(result[i]);
    }    
}

void Linear_ckks::preprocess_softmax(const uint64_t *input, uint64_t *output, const FCMetadata &data){
    int num_diag = data.image_size;
    int num_diag_per_ct = data.slot_count / data.image_size / 2;
    // vector<vector<uint64_t>> result_uint(data.image_size * data.image_size * 12 / data.slot_count, vector<uint64_t>(data.slot_count));
    #pragma omp parallel for
    for (int packing_ind = 0; packing_ind < 6; packing_ind++) {
        vector<uint64_t> temp2, temp3;
        // #pragma omp parallel for num_threads(32)
        for (int diag_ind = 0; diag_ind < num_diag; diag_ind++) {
            for (int j = 0; j < num_diag; j++) {
                temp2.push_back(input[((j + diag_ind) % num_diag) + j * data.image_size + packing_ind * 2 * data.image_size * data.image_size]);
                temp3.push_back(input[((j + diag_ind) % num_diag) + j * data.image_size + (packing_ind * 2 + 1) * data.image_size * data.image_size]);
            }
            if (temp2.size() == data.slot_count / 2) {
                temp2.insert(temp2.end(), temp3.begin(), temp3.end());
                // result_uint[packing_ind * data.image_size * data.image_size * 2 / data.slot_count + diag_ind / num_diag_per_ct] = temp2;

                int offset = packing_ind * data.image_size * data.image_size * 2 / data.slot_count + diag_ind / num_diag_per_ct;
                memcpy(&output[offset*data.slot_count], temp2.data(), data.slot_count*sizeof(uint64_t));
                temp2.clear();
                temp3.clear();
            }
        }
    }
}

vector<vector<Plaintext>> Linear_ckks::softmax_mask_ct_ct(HE_CKKS* he, const FCMetadata &data){
    vector<vector<Plaintext>> mask(2, vector<Plaintext>(data.image_size));
    int num_diag = data.image_size;
    int num_diag_per_ct = data.slot_count / data.image_size / 2;

    #pragma omp parallel for num_threads(32)
    for (int i = 0; i < data.image_size; i++) {
        vector<int64_t> mask1(data.image_size, 0);
        vector<int64_t> mask2(data.image_size, 0);
        for (int j = 0; j < data.image_size - i; j++) {
            mask1[j] = 1;
        }
        for (int j = data.image_size - i; j < data.image_size; j++) {
            mask2[j] = 1;
        }
        vector<double> m1(data.slot_count, 0), m2(data.slot_count, 0);
        int start_ind = (i % num_diag_per_ct) * num_diag;
        for (int j = start_ind; j < num_diag + start_ind; j++) {
            m1[j] = mask1[j - start_ind];
            m1[j + data.slot_count / 2] = mask1[j - start_ind];
            m2[j] = mask2[j - start_ind];
            m2[j + data.slot_count / 2] = mask2[j - start_ind];
        }
        Plaintext pt;
        he->encoder->encode(m1, pow(2,he->he_scale), pt);
        mask[0][i] = pt;
        he->encoder->encode(m2, pow(2,he->he_scale), pt);
        mask[1][i] = pt;
    }
    return mask;
}

vector<vector<vector<Ciphertext>>> Linear_ckks::preprocess_softmax_s1_ct_ct(HE_CKKS* he, const vector<Ciphertext> &matrix, const FCMetadata &data, vector<vector<Plaintext>> &mask) {

    int num_diag = data.image_size;
    int num_diag_per_ct = data.slot_count / data.image_size / 2;
    vector<vector<vector<Ciphertext>>> s2_pack(6);

    #pragma omp parallel for num_threads(2)
    for (int packing_ind = 0; packing_ind < 6; packing_ind++) {
        // cout << "packing_ind: " << packing_ind << endl;
        vector<vector<Ciphertext>> weightMatrix1(2, vector<Ciphertext>(num_diag));
        #pragma omp parallel for
        for (int diag_ind = 0; diag_ind < num_diag; diag_ind++) {

            // int cur_diag = (packing_ind * num_diag_per_ct + diag_ind) % num_diag;
            int cts_ind = packing_ind * data.image_size * data.image_size * 2 / data.slot_count + diag_ind / num_diag_per_ct;
            Ciphertext cur_ct = matrix[cts_ind];
            Plaintext mask1 = mask[0][diag_ind];
            Plaintext mask2 = mask[1][diag_ind];
            Ciphertext cur_ct_l, cur_ct_r;
            cur_ct_l = Ciphertext(cur_ct);
            cur_ct_r = Ciphertext(cur_ct);
            // he->evaluator->multiply_plain(cur_ct, mask1, cur_ct_l);
            // he->evaluator->multiply_plain(cur_ct, mask2, cur_ct_r);
            for (int j = 0; j < std::log2(num_diag_per_ct); j++) {
                Ciphertext temp_ct;
                he->evaluator->rotate_vector(cur_ct_l, (int64_t)num_diag * std::pow(2, j), *(he->gal_keys), temp_ct);
                he->evaluator->add_inplace(cur_ct_l, temp_ct);
                he->evaluator->rotate_vector(cur_ct_r, (int64_t)num_diag * std::pow(2, j), *(he->gal_keys), temp_ct);
                he->evaluator->add_inplace(cur_ct_r, temp_ct);
            }
            weightMatrix1[0][diag_ind] = cur_ct_l;
            weightMatrix1[1][diag_ind] = cur_ct_r;
        }
        s2_pack[packing_ind] = weightMatrix1;
    }
    return s2_pack;
}

Linear_ckks::Linear_ckks(){

}

Linear_ckks::Linear_ckks(int party, NetIO *io) {
	this->party = party;
	this->io = io;
    pp_1.resize(ATTENTION_LAYERS);
    pp_2.resize(ATTENTION_LAYERS);
    pp_3.resize(ATTENTION_LAYERS);
    pp_4.resize(ATTENTION_LAYERS);

    w_ln_1_pt.resize(ATTENTION_LAYERS);
    w_ln_2_pt.resize(ATTENTION_LAYERS);

    data_lin1_0.filter_h = COMMON_DIM;
    data_lin1_0.filter_w = OUTPUT_DIM;
    data_lin1_0.image_size = INPUT_DIM;
    data_lin1_0.slot_count = 16384;

    int input_dim = INPUT_DIM;

    data_lin1_1.filter_h = COMMON_DIM;
    data_lin1_1.filter_w = OUTPUT_DIM;
    data_lin1_1.image_size = input_dim;
    data_lin1_1.slot_count = 16384;

    data_lin2.filter_h = COMMON_DIM;
    data_lin2.filter_w = COMMON_DIM;
    data_lin2.image_size = input_dim;
    data_lin2.slot_count = 16384;

    data_lin3.filter_h = COMMON_DIM;
    data_lin3.filter_w = INTER_DIM;
    data_lin3.image_size = input_dim;
    data_lin3.slot_count = 16384;

    data_lin4.filter_h = INTER_DIM;
    data_lin4.filter_w = COMMON_DIM;
    data_lin4.image_size = input_dim;
    data_lin4.slot_count = 8192;
    set_HE();
    // exit(0);
}

void Linear_ckks::set_HE(){
    cout << "he0 ok" << endl;
	this->he1 = new HE_CKKS(party, io, 32768, std::vector<int>{60, 22, 60, 60, 60, 60},20,44);
    cout << "he1 ok" << endl;
    this->he2 = new HE_CKKS(party, io, 32768, std::vector<int>{60, 40, 60, 60, 60, 60, 60},20,43);
    cout << "he2 ok" << endl;
    this->he3 = new HE_CKKS(party, io, 32768, std::vector<int>{60, 32, 60, 60, 60, 60, 60, 60},20,46);
    cout << "he3 ok" << endl;
    this->he4 = new HE_CKKS(party, io, 16384, std::vector<int>{60, 34, 60, 60, 60, 60},20,38);
}

// plain_mod_bitwidth is the bitwidth of plain_mod
void Linear_ckks::set_HE_bfv(size_t poly_modulus_degree,vector<int> coeff_bit_sizes, uint64_t plain_mod_bitwidth){
    this->he_bfv = new HE(party, io, poly_modulus_degree, coeff_bit_sizes, plain_mod_bitwidth, true);
}

