/*
Written by Dhruv Kulkarni, 6122APT, last modified on Nov 12
*/

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

namespace sutra {

class GpuMonitor {
public:
    GpuMonitor(
        std::string output_path,
        int sample_interval_ms = 200
    );

    ~GpuMonitor();

    GpuMonitor(const GpuMonitor&) = delete;
    GpuMonitor& operator=(const GpuMonitor&) = delete;

    void start();
    void stop();

    bool isRunning() const;

private:
    void monitorLoop();

    static std::int64_t unixTimeMilliseconds();

    std::string output_path_;
    std::chrono::milliseconds sample_interval_;

    std::atomic<bool> running_{false};
    std::thread monitor_thread_;

    std::mutex wait_mutex_;
    std::condition_variable wait_condition_;
};

}  // namespace sutra