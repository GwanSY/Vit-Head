// Copyright 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "libspu/mpc/bumblebee/rlwe/bumblebee_metrics.h"

#include <array>
#include <atomic>
#include <chrono>

namespace spu::mpc::bumblebee {
namespace {

constexpr std::size_t kMetricCount =
    static_cast<std::size_t>(BumblebeeMetricKind::kPlainCacheHit) + 1;

struct AtomicMetric {
  std::atomic<std::uint64_t> nanoseconds{0};
  std::atomic<std::size_t> calls{0};
  std::atomic<std::size_t> bytes{0};
};

struct MetricsState {
  std::array<AtomicMetric, kMetricCount> metrics;
};

MetricsState& State() {
  static MetricsState state;
  return state;
}

BumblebeeMetric& SelectMetric(BumblebeeMetrics& metrics,
                              BumblebeeMetricKind kind) {
  switch (kind) {
    case BumblebeeMetricKind::kHe:
      return metrics.he;
    case BumblebeeMetricKind::kConversion:
      return metrics.conversion;
    case BumblebeeMetricKind::kH2D:
      return metrics.h2d;
    case BumblebeeMetricKind::kD2H:
      return metrics.d2h;
    case BumblebeeMetricKind::kKernelSubmit:
      return metrics.kernel_submit;
    case BumblebeeMetricKind::kSyncWait:
      return metrics.sync_wait;
    case BumblebeeMetricKind::kGpuCpu:
      return metrics.gpu_cpu;
    case BumblebeeMetricKind::kCipherUpload:
      return metrics.cipher_upload;
    case BumblebeeMetricKind::kPlainUpload:
      return metrics.plain_upload;
    case BumblebeeMetricKind::kGaloisUpload:
      return metrics.galois_upload;
    case BumblebeeMetricKind::kCipherCacheHit:
      return metrics.cipher_cache_hit;
    case BumblebeeMetricKind::kPlainCacheHit:
      return metrics.plain_cache_hit;
  }
  return metrics.he;
}

}  // namespace

void ResetBumblebeeMetrics() {
  auto& state = State();
  for (auto& metric : state.metrics) {
    metric.nanoseconds.store(0, std::memory_order_relaxed);
    metric.calls.store(0, std::memory_order_relaxed);
    metric.bytes.store(0, std::memory_order_relaxed);
  }
}

BumblebeeMetrics GetBumblebeeMetrics() {
  auto& state = State();
  BumblebeeMetrics result;
  for (std::size_t index = 0; index < state.metrics.size(); ++index) {
    const auto& source = state.metrics[index];
    auto& target = SelectMetric(result, static_cast<BumblebeeMetricKind>(index));
    target.seconds = static_cast<double>(
                         source.nanoseconds.load(std::memory_order_relaxed)) /
                     1.0e9;
    target.calls = source.calls.load(std::memory_order_relaxed);
    target.bytes = source.bytes.load(std::memory_order_relaxed);
  }
  return result;
}

void RecordBumblebeeMetric(BumblebeeMetricKind kind, double seconds,
                           std::size_t bytes, std::size_t calls) {
  auto& state = State();
  auto& metric = state.metrics[static_cast<std::size_t>(kind)];
  const auto nanoseconds = static_cast<std::uint64_t>(seconds * 1.0e9);
  metric.nanoseconds.fetch_add(nanoseconds, std::memory_order_relaxed);
  metric.calls.fetch_add(calls, std::memory_order_relaxed);
  metric.bytes.fetch_add(bytes, std::memory_order_relaxed);
}

ScopedBumblebeeMetric::~ScopedBumblebeeMetric() {
  const auto elapsed = Clock::now() - start_;
  RecordBumblebeeMetric(
      kind_, std::chrono::duration_cast<std::chrono::duration<double>>(elapsed)
                 .count(),
      bytes_);
}

}  // namespace spu::mpc::bumblebee
