#pragma once

#ifdef BLB_VIT_VALIDATION_OUTPUTS

#include "FloatingPoint/fp-math.h"
#include "utils/io_pack.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class VitValidationObserver {
public:
  VitValidationObserver(int party, sci::IOPack &io, FPMath &fp,
                        const std::string &manifest_path,
                        const std::string &model_path,
                        const std::string &input_path,
                        const std::string &reference_path,
                        const std::string &runtime_source_sha256,
                        std::size_t logical_tokens,
                        std::size_t executed_blocks,
                        const std::string &output_dir);

  void observe(const std::string &name, const FixArray &secure,
               std::size_t physical_rows, std::size_t cols,
               std::size_t valid_rows);
  void finalize();

private:
  struct ShareFile {
    std::string filename;
    std::string sha256;
  };

  struct Stage {
    std::string name;
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t elements = 0;
    int source_scale = 0;
    std::string packing;
    ShareFile fixed;
    ShareFile clear;
  };

  struct Metrics {
    std::string name;
    std::size_t elements = 0;
    double fixed_max_abs = 0.0;
    double fixed_mean_abs = 0.0;
    double fixed_rmse = 0.0;
    double clear_max_abs = 0.0;
    double clear_mean_abs = 0.0;
    double clear_rmse = 0.0;
    int rmse_square_scale = 0;
    int rmse_split_bits = 0;
    double milliseconds = 0.0;
    std::uint64_t bytes = 0;
    std::uint64_t logical_rounds = 0;
  };

  std::vector<std::uint64_t> load_share(const Stage &stage,
                                        const ShareFile &share) const;
  void load_manifest(const std::string &manifest_path,
                     const std::string &model_path,
                     const std::string &input_path,
                     const std::string &reference_path,
                     const std::string &runtime_source_sha256,
                     std::size_t logical_tokens,
                     std::size_t executed_blocks);

  int party_;
  sci::IOPack &io_;
  FPMath &fp_;
  std::string manifest_path_;
  std::string manifest_dir_;
  std::string output_dir_;
  int ell_ = 0;
  int scale_ = 0;
  double clear_quantization_bound_ = 0.0;
  std::vector<Stage> stages_;
  std::vector<Metrics> metrics_;
  std::size_t next_stage_ = 0;
};

#endif
