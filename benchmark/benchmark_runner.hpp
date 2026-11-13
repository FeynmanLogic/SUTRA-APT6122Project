
/*
Written by Dhruv Kulkarni, 6122APT, last modified on Nov 10
*/
#pragma once

#include "benchmark_config.hpp"
#include "../common/inference_types.hpp"
#include "../worker/worker.hpp"

#include <string>
#include <vector>

namespace sutra {

struct BenchmarkSummary {
    int completed_requests{0};
    int failed_requests{0};
    int generated_tokens{0};

    double wall_time_seconds{0.0};
    double requests_per_second{0.0};
    double tokens_per_second{0.0};

    double mean_latency_ms{0.0};
    double p50_latency_ms{0.0};
    double p95_latency_ms{0.0};
    double p99_latency_ms{0.0};

    double p50_ttft_ms{0.0};
    double p95_ttft_ms{0.0};
};

class BenchmarkRunner {
public:
    explicit BenchmarkRunner(
        InferenceWorker& worker
    );

    BenchmarkSummary run(
        RequestBatch requests,
        const BenchmarkConfig& config
    );

private:
    struct BenchmarkRecord {
        InferenceResponse response;
        int batch_size{0};
        double client_latency_ms{0.0};
    };

    void runWarmups(
        const InferenceRequest& request,
        int warmup_count
    );

    std::vector<RequestBatch> createBatches(
        const RequestBatch& requests,
        int batch_size
    ) const;

    void writeCsv(
        const std::vector<BenchmarkRecord>& records,
        const BenchmarkConfig& config
    ) const;

    BenchmarkSummary calculateSummary(
        const std::vector<BenchmarkRecord>& records,
        double wall_time_seconds
    ) const;

    static double percentile(
        std::vector<double> values,
        double fraction
    );

    static std::string escapeCsv(
        const std::string& value
    );

    InferenceWorker& worker_;
};

}  // namespace sutra