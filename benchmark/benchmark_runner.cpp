/*
Written by Dhruv Kulkarni, 6122APT, last modified on Nov 10
*/

#include "benchmark_runner.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace sutra {

namespace {

std::int64_t currentUnixMilliseconds() {
    const auto now =
        std::chrono::system_clock::now();

    return std::chrono::duration_cast<
        std::chrono::milliseconds
    >(now.time_since_epoch()).count();
}

}  // namespace

BenchmarkRunner::BenchmarkRunner(
    InferenceWorker& worker
)
    : worker_(worker) {}

void BenchmarkRunner::runWarmups(
    const InferenceRequest& request,
    int warmup_count
) {
    for (int index = 0;
         index < warmup_count;
         ++index) {
        InferenceRequest warmup = request;
        warmup.request_id = 0;
        warmup.arrival_time_ms =
            currentUnixMilliseconds();

        worker_.processBatch({warmup});
    }
}

std::vector<RequestBatch>
BenchmarkRunner::createBatches(
    const RequestBatch& requests,
    int batch_size
) const {
    if (batch_size <= 0) {
        throw std::invalid_argument(
            "Batch size must be positive."
        );
    }

    std::vector<RequestBatch> batches;

    for (std::size_t start = 0;
         start < requests.size();
         start += batch_size) {
        const std::size_t finish = std::min(
            start +
                static_cast<std::size_t>(batch_size),
            requests.size()
        );

        batches.emplace_back(
            requests.begin() + start,
            requests.begin() + finish
        );
    }

    return batches;
}

BenchmarkSummary BenchmarkRunner::run(
    RequestBatch requests,
    const BenchmarkConfig& config
) {
    if (requests.empty()) {
        throw std::invalid_argument(
            "Benchmark requires at least one request."
        );
    }

    if (config.concurrency <= 0) {
        throw std::invalid_argument(
            "Concurrency must be positive."
        );
    }

    runWarmups(
        requests.front(),
        config.warmup_requests
    );

    std::vector<RequestBatch> batches =
        createBatches(
            requests,
            config.batch_size
        );

    std::vector<BenchmarkRecord> records;
    records.reserve(requests.size());

    std::atomic<std::size_t> next_batch{0};
    std::mutex records_mutex;

    const int thread_count = std::min(
        config.concurrency,
        static_cast<int>(batches.size())
    );

    const auto benchmark_start =
        std::chrono::steady_clock::now();

    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    for (int thread_index = 0;
         thread_index < thread_count;
         ++thread_index) {
        threads.emplace_back([&]() {
            while (true) {
                const std::size_t batch_index =
                    next_batch.fetch_add(1);

                if (batch_index >= batches.size()) {
                    break;
                }

                RequestBatch batch =
                    batches[batch_index];

                for (auto& request : batch) {
                    request.arrival_time_ms =
                        currentUnixMilliseconds();
                }

                const auto submission_time =
                    std::chrono::steady_clock::now();

                ResponseBatch responses =
                    worker_.processBatch(batch);

                const auto completion_time =
                    std::chrono::steady_clock::now();

                const double client_latency_ms =
                    std::chrono::duration<
                        double,
                        std::milli
                    >(
                        completion_time -
                        submission_time
                    ).count();

                std::lock_guard<std::mutex> lock(
                    records_mutex
                );

                for (auto& response : responses) {
                    BenchmarkRecord record;
                    record.response =
                        std::move(response);
                    record.batch_size =
                        static_cast<int>(
                            batch.size()
                        );
                    record.client_latency_ms =
                        client_latency_ms;

                    records.push_back(
                        std::move(record)
                    );
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    const auto benchmark_finish =
        std::chrono::steady_clock::now();

    const double wall_time_seconds =
        std::chrono::duration<double>(
            benchmark_finish -
            benchmark_start
        ).count();

    std::sort(
        records.begin(),
        records.end(),
        [](const BenchmarkRecord& left,
           const BenchmarkRecord& right) {
            return left.response.request_id <
                   right.response.request_id;
        }
    );

    writeCsv(records, config);

    return calculateSummary(
        records,
        wall_time_seconds
    );
}

double BenchmarkRunner::percentile(
    std::vector<double> values,
    double fraction
) {
    if (values.empty()) {
        return 0.0;
    }

    std::sort(
        values.begin(),
        values.end()
    );

    const std::size_t index =
        static_cast<std::size_t>(
            std::ceil(
                fraction * values.size()
            )
        ) - 1;

    return values[
        std::min(index, values.size() - 1)
    ];
}

BenchmarkSummary
BenchmarkRunner::calculateSummary(
    const std::vector<BenchmarkRecord>& records,
    double wall_time_seconds
) const {
    BenchmarkSummary summary;
    summary.wall_time_seconds =
        wall_time_seconds;

    std::vector<double> latencies;
    std::vector<double> ttfts;

    double total_latency = 0.0;

    for (const auto& record : records) {
        if (!record.response.success) {
            ++summary.failed_requests;
            continue;
        }

        ++summary.completed_requests;

        summary.generated_tokens +=
            record.response.generated_tokens;

        latencies.push_back(
            record.client_latency_ms
        );

        ttfts.push_back(
            record.response.time_to_first_token_ms
        );

        total_latency +=
            record.client_latency_ms;
    }

    if (wall_time_seconds > 0.0) {
        summary.requests_per_second =
            summary.completed_requests /
            wall_time_seconds;

        summary.tokens_per_second =
            summary.generated_tokens /
            wall_time_seconds;
    }

    if (summary.completed_requests > 0) {
        summary.mean_latency_ms =
            total_latency /
            summary.completed_requests;
    }

    summary.p50_latency_ms =
        percentile(latencies, 0.50);

    summary.p95_latency_ms =
        percentile(latencies, 0.95);

    summary.p99_latency_ms =
        percentile(latencies, 0.99);

    summary.p50_ttft_ms =
        percentile(ttfts, 0.50);

    summary.p95_ttft_ms =
        percentile(ttfts, 0.95);

    return summary;
}

std::string BenchmarkRunner::escapeCsv(
    const std::string& value
) {
    std::string escaped = "\"";

    for (char character : value) {
        if (character == '"') {
            escaped += "\"\"";
        } else if (
            character == '\n' ||
            character == '\r'
        ) {
            escaped += ' ';
        } else {
            escaped += character;
        }
    }

    escaped += '"';
    return escaped;
}

void BenchmarkRunner::writeCsv(
    const std::vector<BenchmarkRecord>& records,
    const BenchmarkConfig& config
) const {
    const std::filesystem::path output_path(
        config.output_csv
    );

    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(
            output_path.parent_path()
        );
    }

    std::ofstream output(config.output_csv);

    if (!output) {
        throw std::runtime_error(
            "Unable to open benchmark CSV: " +
            config.output_csv
        );
    }

    output
        << "run_id,request_id,worker_id,backend,"
    << "batch_size,concurrency,input_tokens,"
    << "output_tokens,queue_time_ms,ttft_ms,"
    << "inference_ms,engine_latency_ms,"
    << "client_latency_ms,tokens_per_second,"
    << "success,error\n";

    output << std::fixed << std::setprecision(3);

    for (const auto& record : records) {
        const auto& response = record.response;

        output
            << escapeCsv(config.run_id) << ','
            << response.request_id << ','
            << response.worker_id << ','
            << escapeCsv(
                   toString(config.backend)
               ) << ','
            << record.batch_size << ','
            << config.concurrency << ','
            << response.input_tokens << ','
<< response.generated_tokens << ','
<< response.queue_time_ms << ','
<< response.time_to_first_token_ms << ','
<< response.inference_time_ms << ','
            << response.latency_ms << ','
            << record.client_latency_ms << ','
            << response.tokens_per_second << ','
            << (response.success ? "true" : "false")
            << ','
            << escapeCsv(
                   response.error_message
               )
            << '\n';
    }
}

}  // namespace sutra