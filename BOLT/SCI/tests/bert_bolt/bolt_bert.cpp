/*
*/

#include "bert.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <system_error>
#include <vector>

using namespace std;
using namespace seal;
using namespace sci;

int party = 0;
int port = 8000;
string address = "127.0.0.1";
int num_threads = 4;
int bitlength = 37;

string path;
string output_file_path = "bolt_bert_results.txt";
int num_class = 2;
int sample_id = 0;
int num_sample = 1;

bool pruning = false;

struct BertE2EMetrics {
    std::uint64_t communication_bytes = 0;
    std::uint64_t logical_rounds = 0;
    std::uint64_t inference_communication_bytes = 0;
    std::uint64_t inference_logical_rounds = 0;
    double inference_time_ms = 0.0;
    double end_to_end_time_ms = 0.0;
};

namespace {

namespace fs = std::filesystem;

vector<fs::path> bert_model_files(const fs::path &weights_dir) {
    vector<fs::path> files;
    const vector<string> layer_suffixes = {
        "attention.self.query.weight.txt",
        "attention.self.query.bias.txt",
        "attention.self.key.weight.txt",
        "attention.self.key.bias.txt",
        "attention.self.value.weight.txt",
        "attention.self.value.bias.txt",
        "attention.output.dense.weight.txt",
        "attention.output.dense.bias.txt",
        "intermediate.dense.weight.txt",
        "intermediate.dense.bias.txt",
        "output.dense.weight.txt",
        "output.dense.bias.txt",
        "attention.output.LayerNorm.weight.txt",
        "attention.output.LayerNorm.bias.txt",
        "output.LayerNorm.weight.txt",
        "output.LayerNorm.bias.txt",
    };
    for (int layer = 0; layer < 12; ++layer) {
        for (const string &suffix : layer_suffixes) {
            files.push_back(weights_dir /
                            ("bert.encoder.layer." + to_string(layer) + "." +
                             suffix));
        }
    }
    files.push_back(weights_dir / "bert.pooler.dense.weight.txt");
    files.push_back(weights_dir / "bert.pooler.dense.bias.txt");
    files.push_back(weights_dir / "classifier.weight.txt");
    files.push_back(weights_dir / "classifier.bias.txt");
    return files;
}

bool is_nonempty_regular_file(const fs::path &file) {
    std::error_code error;
    const bool regular = fs::is_regular_file(file, error);
    if (error || !regular) {
        return false;
    }
    const auto size = fs::file_size(file, error);
    return !error && size > 0;
}

bool report_missing_files(const vector<fs::path> &files) {
    vector<fs::path> missing;
    for (const fs::path &file : files) {
        if (!is_nonempty_regular_file(file)) {
            missing.push_back(file);
        }
    }
    if (missing.empty()) {
        return true;
    }
    cerr << "BOLT_BERT_PREFLIGHT=FAIL missing_or_empty=" << missing.size()
         << endl;
    for (const fs::path &file : missing) {
        cerr << "  " << file.string() << endl;
    }
    return false;
}

}  // namespace

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

    if (party != ALICE && party != BOB) {
        cerr << "BOLT_BERT_PREFLIGHT=FAIL reason=invalid_role expected=1_or_2 got="
             << party << endl;
        return 2;
    }
    if (path.empty()) {
        cerr << "BOLT_BERT_PREFLIGHT=FAIL reason=missing_path" << endl;
        return 2;
    }
    if (num_class != NUM_CLASS) {
        cerr << "BOLT_BERT_PREFLIGHT=FAIL reason=unsupported_num_class expected="
             << NUM_CLASS << " got=" << num_class << endl;
        return 2;
    }
    if (sample_id < 0 || num_sample <= 0) {
        cerr << "BOLT_BERT_PREFLIGHT=FAIL reason=invalid_sample_range id="
             << sample_id << " num_sample=" << num_sample << endl;
        return 2;
    }

    const fs::path data_root(path);
    const fs::path weights_dir = data_root / "weights_txt";
    vector<fs::path> required_files;
    if (party == ALICE) {
        required_files = bert_model_files(weights_dir);
    } else {
        for (int i = sample_id; i < sample_id + num_sample; ++i) {
            required_files.push_back(weights_dir /
                                     ("inputs_" + to_string(i) + "_data.txt"));
            required_files.push_back(weights_dir /
                                     ("inputs_" + to_string(i) + "_mask.txt"));
        }
        const fs::path output_path(output_file_path);
        if (!output_path.parent_path().empty()) {
            std::error_code error;
            fs::create_directories(output_path.parent_path(), error);
            if (error) {
                cerr << "BOLT_BERT_PREFLIGHT=FAIL reason=output_directory path="
                     << output_path.parent_path().string()
                     << " error=" << error.message() << endl;
                return 2;
            }
        }
    }
    if (!report_missing_files(required_files)) {
        return 2;
    }
    cout << "BOLT_BERT_PREFLIGHT=PASS role="
         << (party == ALICE ? "ALICE" : "BOB")
         << " checked_files=" << required_files.size() << endl;

    cout << ">>> Evaluating Bert" << endl;
    cout << "-> Role: " << party << endl;
    cout << "-> Address: " << address << endl;
    cout << "-> Port: " << port << endl;
    cout << "<<<" << endl << endl;

    const auto e2e_start = high_resolution_clock::now();
    Bert bt(party, port, address, (weights_dir.string() + "/"), pruning);

    const std::uint64_t local_before_bytes = bt.total_comm();
    const std::uint64_t local_before_rounds = bt.total_rounds();
    const auto inference_start = high_resolution_clock::now();
    
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
            return 2;
        }
        for(int i = sample_id; i < sample_id + num_sample; i++ ){
            cout << "==>> Inference sample #" << i << endl;
            vector<double> result = bt.run(
                (weights_dir / ("inputs_" + to_string(i) + "_data.txt")).string(),
                (weights_dir / ("inputs_" + to_string(i) + "_mask.txt")).string()
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
    
    const auto end = high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> inference_elapsed =
        end - inference_start;
    const std::chrono::duration<double, std::milli> end_to_end_elapsed =
        end - e2e_start;
    BertE2EMetrics local_metrics{
        bt.total_comm(),
        bt.total_rounds(),
        bt.total_comm() - local_before_bytes,
        bt.total_rounds() - local_before_rounds,
        inference_elapsed.count(),
        end_to_end_elapsed.count()};

    // Counters are send-only per endpoint.  Exchange the snapshots after the
    // inference interval so the metric itself is not counted as inference
    // communication.  ALICE reports the bidirectional sum and max logical
    // rounds/time; no endpoint is double-counted.
    BertE2EMetrics peer_metrics{};
    if (party == ALICE) {
        bt.io->send_data(&local_metrics, sizeof(local_metrics));
        bt.io->flush();
        bt.io->recv_data(&peer_metrics, sizeof(peer_metrics));
    } else {
        bt.io->recv_data(&peer_metrics, sizeof(peer_metrics));
        bt.io->send_data(&local_metrics, sizeof(local_metrics));
        bt.io->flush();
    }

    if (party == ALICE) {
        const std::uint64_t total_bytes =
            local_metrics.communication_bytes + peer_metrics.communication_bytes;
        const std::uint64_t total_rounds =
            std::max(local_metrics.logical_rounds, peer_metrics.logical_rounds);
        const std::uint64_t inference_bytes =
            local_metrics.inference_communication_bytes +
            peer_metrics.inference_communication_bytes;
        const std::uint64_t inference_rounds = std::max(
            local_metrics.inference_logical_rounds,
            peer_metrics.inference_logical_rounds);
        const double inference_ms =
            std::max(local_metrics.inference_time_ms, peer_metrics.inference_time_ms);
        const double total_ms = std::max(local_metrics.end_to_end_time_ms,
                                         peer_metrics.end_to_end_time_ms);
        cout << "BOLT_BERT_E2E_METRICS communication_bytes=" << total_bytes
             << " end_to_end_time_ms=" << std::fixed << std::setprecision(3)
             << total_ms << " logical_rounds=" << total_rounds
             << " inference_communication_bytes=" << inference_bytes
             << " inference_time_ms=" << inference_ms
             << " inference_logical_rounds=" << inference_rounds
             << " samples=" << num_sample << " status=PASS" << endl;

        const auto emit_counts = [](const char *name,
                                    const sci::bolt::BoltOperationCounts &counts) {
            cout << name
                 << " ciphertexts=" << counts.ciphertexts
                 << " rotations=" << counts.rotations
                 << " multiply_plain=" << counts.multiply_plain
                 << " evaluator_multiply=" << counts.evaluator_multiply
                 << " relinearizations=" << counts.relinearizations
                 << " modulus_switches=" << counts.modulus_switches
                 << " noise_floodings=" << counts.noise_floodings << endl;
        };
        emit_counts("BOLT_BERT_HE_COUNTS qkv", bt.lin.bolt_qkv_counts());
        emit_counts("BOLT_BERT_HE_COUNTS linear2", bt.lin.bolt_linear2_counts());
        emit_counts("BOLT_BERT_HE_COUNTS linear3", bt.lin.bolt_linear3_counts());
        emit_counts("BOLT_BERT_HE_COUNTS linear4", bt.lin.bolt_linear4_counts());
        emit_counts("BOLT_BERT_HE_COUNTS qk", bt.lin.bolt_qk_counts());
        emit_counts("BOLT_BERT_HE_COUNTS softmax_v", bt.lin.bolt_softmax_v_counts());
    }

    cout << "-> Inference takes: " << std::fixed << std::setprecision(3)
         << inference_elapsed.count() / 1000.0 << "sec" << endl;

    return 0;
}
