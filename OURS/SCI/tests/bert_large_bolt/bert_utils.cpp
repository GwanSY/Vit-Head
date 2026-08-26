#include "bert_utils.h"

#include <stdexcept>

vector<vector<uint64_t>> read_data(const string& filename) {
    ifstream input_file(filename);
    vector<vector<uint64_t>> data;

    if (!input_file.is_open()) {
        throw runtime_error("required weight is missing: " + filename);
    }

    string line;
    while (getline(input_file, line)) {
        vector<uint64_t> row;
        istringstream line_stream(line);
        string cell;

        while (getline(line_stream, cell, ',')) {
            row.push_back(stoll(cell));
        }

        data.push_back(row);
    }

    input_file.close();
    return data;
}

vector<uint64_t> read_bias(const string& filename, int output_dim) {
    ifstream input_file(filename);
    vector<uint64_t> data;

    if (!input_file.is_open()) {
        throw runtime_error("required bias is missing: " + filename);
    }

    string line;
    vector<uint64_t> sub_mat;
    int row_counting = 0;
    while (getline(input_file, line)) {
        istringstream line_stream(line);
        string cell;

        while (getline(line_stream, cell, ',')) {
            sub_mat.push_back(stoll(cell));
        }
        row_counting++;
        if (row_counting == output_dim) {
            data = sub_mat;
            sub_mat.clear();
            row_counting -= output_dim;
        }
    }

    input_file.close();
    return data;
}

vector<vector<vector<uint64_t>>> read_qkv_weights(const string& filename) {
    ifstream input_file(filename);
    vector<vector<vector<uint64_t>>> data;

    if (!input_file.is_open()) {
        throw runtime_error("required QKV weight is missing: " + filename);
    }

    string line;
    vector<vector<uint64_t>> sub_mat;
    int row_counting = 0;
    while (getline(input_file, line)) {
        vector<uint64_t> row;
        istringstream line_stream(line);
        string cell;

        while (getline(line_stream, cell, ',')) {
            row.push_back(stoll(cell));
        }
        sub_mat.push_back(row);
        row_counting++;
        if (row_counting == COMMON_DIM) {
            data.push_back(sub_mat);
            sub_mat.clear();
            row_counting -= COMMON_DIM;
        }
    }

    input_file.close();
    return data;
}

vector<vector<uint64_t>> read_qkv_bias(const string& filename) {
    ifstream input_file(filename);
    vector<vector<uint64_t>> data;

    if (!input_file.is_open()) {
        throw runtime_error("required QKV bias is missing: " + filename);
    }

    string line;
    vector<uint64_t> sub_mat;
    int row_counting = 0;
    while (getline(input_file, line)) {
        istringstream line_stream(line);
        string cell;

        while (getline(line_stream, cell, ',')) {
            sub_mat.push_back(stoll(cell));
        }
        row_counting++;
        if (row_counting == OUTPUT_DIM) {
            data.push_back(sub_mat);
            sub_mat.clear();
            row_counting -= OUTPUT_DIM;
        }
    }

    input_file.close();
    return data;
}

string replace(string str, string substr1, string substr2) {
    size_t index = str.find(substr1, 0); 
    str.replace(index, substr1.length(), substr2);
    return str;
}

BertModel load_model(string model_dir, int num_class, int layers_to_load,
                     bool require_pooling_and_classifier,
                     bool attention_only){
    if (layers_to_load < 1 || layers_to_load > 12) {
        throw invalid_argument("layers_to_load must be in [1, 12]");
    }
    if (model_dir.empty()) {
        throw invalid_argument("model directory is empty");
    }
    if (model_dir.back() != '/') {
        model_dir.push_back('/');
    }
    BertModel bm;

    bm.w_q.resize(layers_to_load);
    bm.w_k.resize(layers_to_load);
    bm.w_v.resize(layers_to_load);
    bm.w_o.resize(layers_to_load);
    bm.w_i_1.resize(layers_to_load);
    bm.w_i_2.resize(layers_to_load);

    bm.b_q.resize(layers_to_load);
    bm.b_k.resize(layers_to_load);
    bm.b_v.resize(layers_to_load);
    bm.b_o.resize(layers_to_load);
    bm.b_i_1.resize(layers_to_load);
    bm.b_i_2.resize(layers_to_load);

    bm.w_ln_1.resize(layers_to_load);
    bm.w_ln_2.resize(layers_to_load);

    bm.b_ln_1.resize(layers_to_load);
    bm.b_ln_2.resize(layers_to_load);

    
    // Attention
    string wq_fname = 
    "bert.encoder.layer.X.attention.self.query.weight.txt";
    string bq_fname = 
    "bert.encoder.layer.X.attention.self.query.bias.txt";

    string wk_fname = 
    "bert.encoder.layer.X.attention.self.key.weight.txt";
    string bk_fname = 
    "bert.encoder.layer.X.attention.self.key.bias.txt";

    string wv_fname = 
    "bert.encoder.layer.X.attention.self.value.weight.txt";
    string bv_fname = 
    "bert.encoder.layer.X.attention.self.value.bias.txt";

    string wo_fname = 
    "bert.encoder.layer.X.attention.output.dense.weight.txt";
    string bo_fname = 
    "bert.encoder.layer.X.attention.output.dense.bias.txt";

    string wi1_fname = 
    "bert.encoder.layer.X.intermediate.dense.weight.txt";
    string bi1_fname = 
    "bert.encoder.layer.X.intermediate.dense.bias.txt";

    string wi2_fname = 
    "bert.encoder.layer.X.output.dense.weight.txt";
    string bi2_fname = 
    "bert.encoder.layer.X.output.dense.bias.txt";

    // Layer Norm
    string wln1_fname = 
    "bert.encoder.layer.X.attention.output.LayerNorm.weight.txt";
    string bln1_fname = 
    "bert.encoder.layer.X.attention.output.LayerNorm.bias.txt";

    string wln2_fname = 
    "bert.encoder.layer.X.output.LayerNorm.weight.txt";
    string bln2_fname = 
    "bert.encoder.layer.X.output.LayerNorm.bias.txt";

    // Pooling
    string wp_fname = 
    "bert.pooler.dense.weight.txt";
    string bp_fname = 
    "bert.pooler.dense.bias.txt";

    // Classification
    string wc_fname = 
    "classifier.weight.txt";
    string bc_fname = 
    "classifier.bias.txt";
    
    // Attention
    for(int i = 0; i < layers_to_load; i++){
        string lid = to_string(i);
        vector<vector<vector<uint64_t>>> wq = read_qkv_weights(
            model_dir + replace(wq_fname, "X", lid)
        );
        vector<vector<vector<uint64_t>>> wk = read_qkv_weights(
            model_dir + replace(wk_fname, "X", lid)
        );
        vector<vector<vector<uint64_t>>> wv = read_qkv_weights(
            model_dir + replace(wv_fname, "X", lid)
        );
        
        vector<vector<uint64_t>> wo = read_data(
            model_dir + replace(wo_fname, "X", lid)
        );
        vector<vector<uint64_t>> wi1;
        vector<vector<uint64_t>> wi2;
        if (!attention_only) {
            wi1 = read_data(model_dir + replace(wi1_fname, "X", lid));
            wi2 = read_data(model_dir + replace(wi2_fname, "X", lid));
        }

        vector<vector<uint64_t>> bq = read_qkv_bias(
            model_dir + replace(bq_fname, "X", lid)
        );
        vector<vector<uint64_t>> bk = read_qkv_bias(
            model_dir + replace(bk_fname, "X", lid)
        );
        vector<vector<uint64_t>> bv = read_qkv_bias(
            model_dir + replace(bv_fname, "X", lid)
        );

        vector<uint64_t> wln1;
        vector<uint64_t> bln1;
        vector<uint64_t> wln2;
        vector<uint64_t> bln2;
        if (!attention_only) {
            wln1 = read_bias(model_dir + replace(wln1_fname, "X", lid),
                             COMMON_DIM);
            bln1 = read_bias(model_dir + replace(bln1_fname, "X", lid),
                             COMMON_DIM);
            wln2 = read_bias(model_dir + replace(wln2_fname, "X", lid),
                             COMMON_DIM);
            bln2 = read_bias(model_dir + replace(bln2_fname, "X", lid),
                             COMMON_DIM);
        }

        vector<uint64_t> bo = read_bias(
            model_dir + replace(bo_fname, "X", lid), 
            COMMON_DIM
        );
        vector<uint64_t> bi1;
        vector<uint64_t> bi2;
        if (!attention_only) {
            bi1 = read_bias(model_dir + replace(bi1_fname, "X", lid),
                            INTER_DIM);
            bi2 = read_bias(model_dir + replace(bi2_fname, "X", lid),
                            COMMON_DIM);
        }

        auto require_qkv_weight = [i](const auto &value, const string &name) {
            if (value.size() != PACKING_NUM) {
                throw runtime_error("layer " + to_string(i) + " " + name +
                                    " packed head count mismatch");
            }
            for (const auto &head : value) {
                if (head.size() != COMMON_DIM) {
                    throw runtime_error("layer " + to_string(i) + " " + name +
                                        " head input-row count mismatch");
                }
                for (const auto &row : head) {
                    if (row.size() != OUTPUT_DIM) {
                        throw runtime_error("layer " + to_string(i) + " " + name +
                                            " head output width mismatch");
                    }
                }
            }
        };
        auto require_matrix = [i](const auto &value, size_t rows, size_t cols,
                                  const string &name) {
            if (value.size() != rows) {
                throw runtime_error("layer " + to_string(i) + " " + name +
                                    " row count mismatch");
            }
            for (const auto &row : value) {
                if (row.size() != cols) {
                    throw runtime_error("layer " + to_string(i) + " " + name +
                                        " column count mismatch");
                }
            }
        };
        auto require_qkv_bias = [i](const auto &value, const string &name) {
            if (value.size() != PACKING_NUM) {
                throw runtime_error("layer " + to_string(i) + " " + name +
                                    " head count mismatch");
            }
            for (const auto &head : value) {
                if (head.size() != OUTPUT_DIM) {
                    throw runtime_error("layer " + to_string(i) + " " + name +
                                        " head size mismatch");
                }
            }
        };
        auto require_vector = [i](const auto &value, size_t size, const string &name) {
            if (value.size() != size) {
                throw runtime_error("layer " + to_string(i) + " " + name +
                                    " size mismatch");
            }
        };

        require_qkv_weight(wq, "query.weight");
        require_qkv_weight(wk, "key.weight");
        require_qkv_weight(wv, "value.weight");
        require_matrix(wo, COMMON_DIM, COMMON_DIM,
                       "attention.output.dense.weight");
        if (!attention_only) {
            require_matrix(wi1, COMMON_DIM, INTER_DIM,
                           "intermediate.dense.weight");
            require_matrix(wi2, INTER_DIM, COMMON_DIM,
                           "output.dense.weight");
        }
        require_qkv_bias(bq, "query.bias");
        require_qkv_bias(bk, "key.bias");
        require_qkv_bias(bv, "value.bias");
        require_vector(bo, COMMON_DIM, "attention.output.dense.bias");
        if (!attention_only) {
            require_vector(bi1, INTER_DIM, "intermediate.dense.bias");
            require_vector(bi2, COMMON_DIM, "output.dense.bias");
            require_vector(wln1, COMMON_DIM,
                           "attention.output.LayerNorm.weight");
            require_vector(bln1, COMMON_DIM,
                           "attention.output.LayerNorm.bias");
            require_vector(wln2, COMMON_DIM, "output.LayerNorm.weight");
            require_vector(bln2, COMMON_DIM, "output.LayerNorm.bias");
        }

        bm.w_q[i] =  wq;
        bm.w_k[i] =  wk;
        bm.w_v[i] =  wv;
        bm.w_o[i] =  wo;
        bm.w_i_1[i] =  wi1;
        bm.w_i_2[i] =  wi2;

        bm.w_ln_1[i] =  wln1;
        bm.b_ln_1[i] =  bln1;
        bm.w_ln_2[i] =  wln2;
        bm.b_ln_2[i] =  bln2;

        bm.b_q[i] =  bq;
        bm.b_k[i] =  bk;
        bm.b_v[i] =  bv;
        bm.b_o[i] =  bo;
        bm.b_i_1[i] =  bi1;
        bm.b_i_2[i] =  bi2;
    }

    if (require_pooling_and_classifier) {
        bm.w_p = read_data(model_dir + wp_fname);
        bm.b_p = read_bias(model_dir + bp_fname, 768);
        bm.w_c = read_data(model_dir + wc_fname);
        bm.b_c = read_bias(model_dir + bc_fname, num_class);
        if (bm.w_p.size() != 768 || bm.b_p.size() != 768 ||
            bm.w_c.size() != 768 || bm.b_c.size() != static_cast<size_t>(num_class)) {
            throw runtime_error("pooling/classifier shape mismatch");
        }
    }

    return bm;
}
