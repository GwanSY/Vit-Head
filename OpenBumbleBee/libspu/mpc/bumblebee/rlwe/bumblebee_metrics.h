// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>

namespace spu::mpc::bumblebee {

enum class BumblebeeMetricKind : std::uint8_t {
  kHe,
  kConversion,
  kH2D,
  kD2H,
  // CPU time spent submitting Phantom/CUDA evaluation work. This deliberately
  // excludes time blocked in an explicit CUDA synchronization.
  kKernelSubmit,
  // Host wall time blocked in cudaStreamSynchronize/cudaDeviceSynchronize.
  kSyncWait,
  // CPU-side bridge/cache preparation that is neither a transfer nor a CUDA
  // evaluation call (metadata lookup, cache population, object preparation).
  kGpuCpu,
  // Object-level counters used to distinguish new uploads from cache hits.
  kCipherUpload,
  kPlainUpload,
  kGaloisUpload,
  kCipherCacheHit,
  kPlainCacheHit,
};

struct BumblebeeMetric {
  double seconds = 0.0;
  std::size_t calls = 0;
  std::size_t bytes = 0;
};

struct BumblebeeMetrics {
  BumblebeeMetric he;
  BumblebeeMetric conversion;
  BumblebeeMetric h2d;
  BumblebeeMetric d2h;
  BumblebeeMetric kernel_submit;
  BumblebeeMetric sync_wait;
  BumblebeeMetric gpu_cpu;
  BumblebeeMetric cipher_upload;
  BumblebeeMetric plain_upload;
  BumblebeeMetric galois_upload;
  BumblebeeMetric cipher_cache_hit;
  BumblebeeMetric plain_cache_hit;
};

void ResetBumblebeeMetrics();
BumblebeeMetrics GetBumblebeeMetrics();
void RecordBumblebeeMetric(BumblebeeMetricKind kind, double seconds,
                           std::size_t bytes = 0,
                           std::size_t calls = 1);

class ScopedBumblebeeMetric final {
 public:
  explicit ScopedBumblebeeMetric(BumblebeeMetricKind kind,
                                 std::size_t bytes = 0)
      : kind_(kind), bytes_(bytes), start_(Clock::now()) {}

  ScopedBumblebeeMetric(const ScopedBumblebeeMetric&) = delete;
  ScopedBumblebeeMetric& operator=(const ScopedBumblebeeMetric&) = delete;

  ~ScopedBumblebeeMetric();

 private:
  using Clock = std::chrono::steady_clock;

  BumblebeeMetricKind kind_;
  std::size_t bytes_;
  Clock::time_point start_;
};

}  // namespace spu::mpc::bumblebee
