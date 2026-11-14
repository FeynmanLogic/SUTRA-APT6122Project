
#include "scheduler/scheduler.hpp"

#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

using json = nlohmann::json;
using namespace std::chrono_literals;

namespace {

constexpr int PORT = 8080;
constexpr int MAX_CLIENTS = 32;
constexpr std::uint32_t MAX_FRAME = 1024 * 1024;

sutra::Scheduler scheduler;
std::atomic<std::uint64_t> next_id{1};
std::atomic<int> active_clients{0};

std::mutex pending_mutex;
std::map<std::uint64_t,
    std::shared_ptr<std::promise<sutra::Completion>>> pending;

std::mutex log_mutex;

void log(const std::string& message) {
    std::lock_guard<std::mutex> lock(log_mutex);
    std::cout << message << std::endl;
}

bool readExact(int fd, void* data, std::size_t size) {
    auto* ptr = static_cast<char*>(data);

    while (size > 0) {
        ssize_t n = recv(fd, ptr, size, 0);

        if (n < 0 && errno == EINTR)
            continue;

        if (n <= 0)
            return false;

        ptr += n;
        size -= static_cast<std::size_t>(n);
    }

    return true;
}

bool writeExact(int fd, const void* data, std::size_t size) {
    const auto* ptr = static_cast<const char*>(data);

    while (size > 0) {
        ssize_t n = send(fd, ptr, size, MSG_NOSIGNAL);

        if (n < 0 && errno == EINTR)
            continue;

        if (n <= 0)
            return false;

        ptr += n;
        size -= static_cast<std::size_t>(n);
    }

    return true;
}

// TCP frame = 4-byte big-endian length + JSON bytes.
bool readFrame(int fd, std::string& payload) {
    std::uint32_t network_length;

    if (!readExact(fd, &network_length, sizeof(network_length)))
        return false;

    std::uint32_t length = ntohl(network_length);

    if (length == 0 || length > MAX_FRAME)
        return false;

    payload.resize(length);
    return readExact(fd, payload.data(), length);
}

bool sendFrame(int fd, const json& message) {
    std::string payload = message.dump();

    if (payload.size() > MAX_FRAME)
        return false;

    std::uint32_t length =
        htonl(static_cast<std::uint32_t>(payload.size()));

    return writeExact(fd, &length, sizeof(length)) &&
           writeExact(fd, payload.data(), payload.size());
}

std::int64_t currentTimeMs() {
    auto now = std::chrono::system_clock::now();

    return std::chrono::duration_cast<
        std::chrono::milliseconds>(
            now.time_since_epoch()
        ).count();
}

json errorMessage(const std::string& error) {
    return {
        {"version", 1},
        {"type", "error"},
        {"error", error}
    };
}

// Return each completed result to the waiting client.
void completionLoop() {
    while (true) {
        auto completion = scheduler.waitForCompletion();

        if (!completion)
            return;

        std::shared_ptr<std::promise<sutra::Completion>> target;

        {
            std::lock_guard<std::mutex> lock(pending_mutex);

            auto it = pending.find(
                completion->response.request_id
            );

            if (it != pending.end()) {
                target = it->second;
                pending.erase(it);
            }
        }

        if (target)
            target->set_value(std::move(*completion));
    }
}

// CPU-only synthetic worker for integration tests.
void syntheticWorkerLoop() {
    while (true) {
        auto batch = scheduler.waitForDispatch();

        if (!batch)
            return;

        log("Dispatching batch " +
            std::to_string(batch->batch_id) +
            " to worker " +
            std::to_string(batch->worker_id));

        for (const auto& request : batch->requests) {
            std::this_thread::sleep_for(30ms);

            sutra::InferenceResponse response;
            response.request_id = request.request_id;
            response.worker_id = batch->worker_id;
            response.success = true;
            response.generated_text =
                "Synthetic response to: " + request.prompt;

            if (!scheduler.complete(
                    batch->batch_id,
                    batch->worker_id,
                    std::move(response))) {
                log("Warning: completion rejected");
            }
        }
    }
}

// Prevent the demo worker from timing out while idle.
void heartbeatLoop() {
    while (true) {
        std::this_thread::sleep_for(500ms);

        if (!scheduler.heartbeat(0))
            return;
    }
}

void handleClient(int fd) {
    struct ClientGuard {
        int fd;
        ~ClientGuard() {
            close(fd);
            --active_clients;
        }
    } guard{fd};

    timeval timeout{};
    timeout.tv_sec = 10;

    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO,
               &timeout, sizeof(timeout));

    // Inference may take longer than the initial read.
    timeout.tv_sec = 130;
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO,
               &timeout, sizeof(timeout));

    std::string payload;

    if (!readFrame(fd, payload))
        return;

    try {
        json message = json::parse(payload);

        if (!message.is_object() ||
            !message.contains("version") ||
            !message["version"].is_number_integer() ||
            message["version"] != 1 ||
            !message.contains("type") ||
            message["type"] != "infer" ||
            !message.contains("payload") ||
            !message["payload"].is_object()) {

            sendFrame(fd, errorMessage("Invalid request envelope"));
            return;
        }

        const json& body = message["payload"];

        if (!body.contains("prompt") ||
            !body["prompt"].is_string()) {

            sendFrame(fd, errorMessage("Invalid prompt"));
            return;
        }

        std::string prompt = body["prompt"].get<std::string>();

        if (prompt.empty() || prompt.size() > 16 * 1024) {
            sendFrame(fd, errorMessage("Prompt size out of range"));
            return;
        }

        int tokens = 64;
        int priority = 0;

        if (body.contains("max_new_tokens")) {
            if (!body["max_new_tokens"].is_number_integer()) {
                sendFrame(fd, errorMessage("Invalid token limit"));
                return;
            }
            tokens = body["max_new_tokens"].get<int>();
        }

        if (body.contains("priority")) {
            if (!body["priority"].is_number_integer()) {
                sendFrame(fd, errorMessage("Invalid priority"));
                return;
            }
            priority = body["priority"].get<int>();
        }

        if (tokens < 1 || tokens > 512 ||
            priority < 0 || priority > 2) {
            sendFrame(fd, errorMessage("Request values out of range"));
            return;
        }

        sutra::InferenceRequest request;

        request.request_id = next_id.fetch_add(1);
        request.prompt = std::move(prompt);
        request.max_new_tokens = tokens;
        request.priority = priority;
        request.arrival_time_ms = currentTimeMs();

        if (request.request_id == 0) {
            sendFrame(fd, errorMessage("Request IDs exhausted"));
            return;
        }

        auto promise =
            std::make_shared<std::promise<sutra::Completion>>();

        auto future = promise->get_future();

        {
            std::lock_guard<std::mutex> lock(pending_mutex);
            pending.emplace(request.request_id, promise);
        }

        auto status = scheduler.submit(request);

        if (status != sutra::SubmitStatus::Accepted) {
            {
                std::lock_guard<std::mutex> lock(pending_mutex);
                pending.erase(request.request_id);
            }

            sendFrame(fd, errorMessage(sutra::toString(status)));
            return;
        }

        log("Accepted request " +
            std::to_string(request.request_id));

        // The completionLoop fulfills this future.
        sutra::Completion completion = future.get();

        const auto& response = completion.response;

        json reply = {
            {"version", 1},
            {"type", "infer_result"},
            {"request_id", response.request_id},
            {"success", response.success},
            {"status", sutra::toString(completion.code)},
            {"worker_id", response.worker_id},
            {"generated_text", response.generated_text},
            {"error_message", response.error_message},
            {"queue_time_ms", response.queue_time_ms},
            {"inference_time_ms", response.inference_time_ms},
            {"latency_ms", response.latency_ms}
        };

        sendFrame(fd, reply);

    } catch (const std::exception& ex) {
        log(std::string("Client error: ") + ex.what());
        sendFrame(fd, errorMessage("Malformed or invalid request"));
    }
}

} // namespace

int main() {
    if (!scheduler.registerWorker({0, 4, 4, 1})) {
        std::cerr << "Worker registration failed\n";
        return 1;
    }

    std::thread(completionLoop).detach();
    std::thread(syntheticWorkerLoop).detach();
    std::thread(heartbeatLoop).detach();

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);

    if (server_fd < 0) {
        perror("socket");
        return 1;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET,
               SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(PORT);

    if (bind(server_fd,
             reinterpret_cast<sockaddr*>(&address),
             sizeof(address)) < 0) {
        perror("bind");
        close(server_fd);
        return 1;
    }

    if (listen(server_fd, MAX_CLIENTS) < 0) {
        perror("listen");
        close(server_fd);
        return 1;
    }

    log("SUTRA router + scheduler demo listening on 127.0.0.1:8080");

    while (true) {
        int fd = accept(server_fd, nullptr, nullptr);

        if (fd < 0) {
            if (errno == EINTR)
                continue;
            perror("accept");
            continue;
        }

        // Atomic reservation prevents exceeding the client limit.
        int count = active_clients.load();

        do {
            if (count >= MAX_CLIENTS)
                break;
        } while (!active_clients.compare_exchange_weak(
            count, count + 1));

        if (count >= MAX_CLIENTS) {
            close(fd);
            continue;
        }

        try {
            std::thread(handleClient, fd).detach();
        } catch (...) {
            close(fd);
            --active_clients;
        }
    }
}
