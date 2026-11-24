/*
Written by Dhruv Kulkarni, 6122APT, last modified on Nov 12
*/


#include "gpu_monitor.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace sutra {

GpuMonitor::GpuMonitor(
    std::string output_path,
    int sample_interval_ms
)
    : output_path_(std::move(output_path)),
      sample_interval_(sample_interval_ms) {
    if (output_path_.empty()) {
        throw std::invalid_argument(
            "GPU monitor output path cannot be empty."
        );
    }

    if (sample_interval_ms <= 0) {
        throw std::invalid_argument(
            "GPU sampling interval must be positive."
        );
    }
}

GpuMonitor::~GpuMonitor() {
    stop();
}

std::int64_t GpuMonitor::unixTimeMilliseconds() {
    const auto now =
        std::chrono::system_clock::now();

    return std::chrono::duration_cast<
        std::chrono::milliseconds
    >(now.time_since_epoch()).count();
}

void GpuMonitor::start() {
    bool expected = false;

    if (
        !running_.compare_exchange_strong(
            expected,
            true
        )
    ) {
        throw std::runtime_error(
            "GPU monitor is already running."
        );
    }

    try {
        monitor_thread_ =
            std::thread(
                &GpuMonitor::monitorLoop,
                this
            );
    } catch (...) {
        running_ = false;
        throw;
    }
}

void GpuMonitor::stop() {
    if (!running_.exchange(false)) {
        if (monitor_thread_.joinable()) {
            monitor_thread_.join();
        }

        return;
    }

    wait_condition_.notify_all();

    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
}

bool GpuMonitor::isRunning() const {
    return running_.load();
}

void GpuMonitor::monitorLoop() {
    const std::filesystem::path output_path(
        output_path_
    );

    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(
            output_path.parent_path()
        );
    }

    std::ofstream output(output_path_);

    if (!output) {
        std::cerr
            << "GPU monitor could not open: "
            << output_path_
            << '\n';

        running_ = false;
        return;
    }

    output
        << "unix_time_ms,elapsed_ms,gpu_index,"
        << "gpu_name,gpu_utilization_percent,"
        << "memory_used_mib,memory_total_mib,"
        << "temperature_c,power_draw_w\n";

    output.flush();

    const auto monitoring_start =
        std::chrono::steady_clock::now();

    const char* command =
        "nvidia-smi "
        "--query-gpu=index,name,utilization.gpu,"
        "memory.used,memory.total,temperature.gpu,"
        "power.draw "
        "--format=csv,noheader,nounits "
#ifdef _WIN32
        "2>NUL";
#else
        "2>/dev/null";
#endif

    while (running_.load()) {
        const auto sample_time =
            std::chrono::steady_clock::now();

        const double elapsed_ms =
            std::chrono::duration<
                double,
                std::milli
            >(
                sample_time -
                monitoring_start
            ).count();

#ifdef _WIN32
        FILE* pipe = _popen(command, "r");
#else
        FILE* pipe = popen(command, "r");
#endif

        if (pipe == nullptr) {
            std::cerr
                << "GPU monitor failed to run "
                << "nvidia-smi.\n";
        } else {
            char buffer[4096];

            while (
                std::fgets(
                    buffer,
                    sizeof(buffer),
                    pipe
                ) != nullptr
            ) {
                std::string line(buffer);

                while (
                    !line.empty() &&
                    (
                        line.back() == '\n' ||
                        line.back() == '\r'
                    )
                ) {
                    line.pop_back();
                }

                if (!line.empty()) {
                    output
                        << unixTimeMilliseconds()
                        << ','
                        << elapsed_ms
                        << ','
                        << line
                        << '\n';
                }
            }

#ifdef _WIN32
            _pclose(pipe);
#else
            pclose(pipe);
#endif
            output.flush();
        }

        std::unique_lock<std::mutex> lock(
            wait_mutex_
        );

        wait_condition_.wait_for(
            lock,
            sample_interval_,
            [this]() {
                return !running_.load();
            }
        );
    }
}

}  // namespace sutra
