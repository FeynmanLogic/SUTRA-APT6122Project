
#include "scheduler/scheduler.hpp"

#include <nlohmann/json.hpp>

#include "socket_compat.hpp"
#include "worker/worker.hpp"
#include "inference/mock_engine.hpp"
#ifdef SUTRA_ENABLE_LLAMA
#include "inference/llama_engine.hpp"
#endif
#include <algorithm>
#include <csignal>
#include <vector>

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

constexpr int MAX_CLIENTS = 32;
constexpr std::uint32_t MAX_FRAME = 1024 * 1024;

sutra::Scheduler scheduler;
std::atomic<std::uint64_t> next_id{1};
std::atomic<int> active_clients{0};
volatile std::sig_atomic_t stop_requested = 0;
void requestStop(int) { stop_requested = 1; }

std::mutex pending_mutex;
std::map<std::uint64_t,
    std::shared_ptr<std::promise<sutra::Completion>>> pending;

std::mutex log_mutex;

void log(const std::string& message) {
    std::lock_guard<std::mutex> lock(log_mutex);
    std::cout << message << std::endl;
}

bool readExact(Socket fd, void* data, std::size_t size) {
    auto* ptr = static_cast<char*>(data);

    while (size > 0) {
        int n = receiveBytes(fd, ptr, size);

        if (n < 0 && socketInterrupted())
            continue;

        if (n <= 0)
            return false;

        ptr += n;
        size -= static_cast<std::size_t>(n);
    }

    return true;
}

bool writeExact(Socket fd, const void* data, std::size_t size) {
    const auto* ptr = static_cast<const char*>(data);

    while (size > 0) {
        int n = sendBytes(fd, ptr, size);

        if (n < 0 && socketInterrupted())
            continue;

        if (n <= 0)
            return false;

        ptr += n;
        size -= static_cast<std::size_t>(n);
    }

    return true;
}

// TCP frame = 4-byte big-endian length + JSON bytes.
bool readFrame(Socket fd, std::string& payload) {
    std::uint32_t network_length;

    if (!readExact(fd, &network_length, sizeof(network_length)))
        return false;

    std::uint32_t length = ntohl(network_length);

    if (length == 0 || length > MAX_FRAME)
        return false;

    payload.resize(length);
    return readExact(fd, payload.data(), length);
}

bool sendFrame(Socket fd, const json& message) {
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

// One dispatch loop owns this worker and its model context.
void workerLoop(sutra::InferenceWorker& worker) {
    while (auto batch = scheduler.waitForDispatch()) {
        auto responses = worker.processBatch(batch->requests);
        for (auto& response : responses) {
            if (!scheduler.complete(batch->batch_id, batch->worker_id, std::move(response)))
                log("Warning: late or rejected completion");
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

void handleClient(Socket fd) {
    struct ClientGuard {
        Socket fd;
        ~ClientGuard() {
            closeSocket(fd);
            --active_clients;
        }
    } guard{fd};

    setSocketTimeout(fd, SO_RCVTIMEO, 10);
    setSocketTimeout(fd, SO_SNDTIMEO, 10);

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
            if (body["max_new_tokens"] < 1 || body["max_new_tokens"] > 512) {
                sendFrame(fd, errorMessage("Request values out of range"));
                return;
            }
            tokens = body["max_new_tokens"].get<int>();
        }

        if (body.contains("priority")) {
            if (!body["priority"].is_number_integer()) {
                sendFrame(fd, errorMessage("Invalid priority"));
                return;
            }
            if (body["priority"] < 0 || body["priority"] > 2) {
                sendFrame(fd, errorMessage("Request values out of range"));
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
            {"latency_ms", response.latency_ms},
            {"input_tokens", response.input_tokens},
            {"generated_tokens", response.generated_tokens},
            {"time_to_first_token_ms", response.time_to_first_token_ms},
            {"tokens_per_second", response.tokens_per_second}
        };

        sendFrame(fd, reply);

    } catch (const std::exception& ex) {
        log(std::string("Client error: ") + ex.what());
        sendFrame(fd, errorMessage("Malformed or invalid request"));
    }
}

} // namespace

int main(int argc, char* argv[]) {
    int port = 8080;
    int gpu_layers = 0;
    std::string backend = "mock";
    std::string model;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help") {
                std::cout << "Usage: sutra_router [--port 8080] [--backend mock|llama] "
                             "[--model model.gguf] [--gpu-layers 0]\n";
                return 0;
            }
            if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + arg);
            const std::string value = argv[++i];
            auto number = [&] {
                std::size_t consumed = 0;
                int n = std::stoi(value, &consumed);
                if (consumed != value.size()) throw std::invalid_argument("Invalid number: " + value);
                return n;
            };
            if (arg == "--port") port = number();
            else if (arg == "--backend") backend = value;
            else if (arg == "--model") model = value;
            else if (arg == "--gpu-layers") gpu_layers = number();
            else throw std::invalid_argument("Unknown argument: " + arg);
        }
        if (port < 1 || port > 65535 || gpu_layers < 0)
            throw std::invalid_argument("Invalid port or GPU layer count");
        if (backend != "mock" && backend != "llama")
            throw std::invalid_argument("Backend must be mock or llama");
        if (backend == "mock" && (!model.empty() || gpu_layers != 0))
            throw std::invalid_argument("Model and GPU layers require --backend llama");
        std::unique_ptr<sutra::InferenceEngine> engine;
        if (backend == "mock") {
            engine = std::make_unique<sutra::MockInferenceEngine>();
        } else {
#ifdef SUTRA_ENABLE_LLAMA
            if (model.empty()) throw std::invalid_argument("--backend llama requires --model");
            engine = std::make_unique<sutra::LlamaEngine>(gpu_layers, 4096);
            if (!engine->loadModel(model)) throw std::runtime_error("Failed to load model");
#else
            throw std::runtime_error("Rebuild with SUTRA_ENABLE_LLAMA=ON to use --backend llama");
#endif
        }
        sutra::InferenceWorker worker(0, std::move(engine));
        SocketRuntime sockets;
        if (!sockets.ready) throw std::runtime_error("Socket initialization failed");
        Socket server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd == invalid_socket) throw std::runtime_error("socket failed");
        struct ServerGuard {
            Socket fd;
            ~ServerGuard() { closeSocket(fd); }
        } server_guard{server_fd};
        int opt = 1;
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR,
                   reinterpret_cast<const char*>(&opt), sizeof(opt));
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port = htons(static_cast<std::uint16_t>(port));
        if (bind(server_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0)
            throw std::runtime_error("bind failed (port may already be in use)");
        if (listen(server_fd, MAX_CLIENTS) != 0) throw std::runtime_error("listen failed");
        if (!scheduler.registerWorker({0, 4, 4, 1}))
            throw std::runtime_error("Worker registration failed");
        std::signal(SIGINT, requestStop);
        std::signal(SIGTERM, requestStop);
        struct ClientThread {
            std::thread thread;
            std::shared_ptr<std::atomic<bool>> finished;
        };
        std::vector<ClientThread> clients;
        std::thread completions, execution, heartbeat;
        auto joinAll = [&] {
            scheduler.shutdown();
            if (execution.joinable()) execution.join();
            if (completions.joinable()) completions.join();
            if (heartbeat.joinable()) heartbeat.join();
            for (auto& client : clients) client.thread.join();
        };
        try {
            completions = std::thread(completionLoop);
            execution = std::thread(workerLoop, std::ref(worker));
            heartbeat = std::thread(heartbeatLoop);
            log("SUTRA router listening on 127.0.0.1:" + std::to_string(port) + " backend=" + backend);
            while (!stop_requested) {
                for (auto it = clients.begin(); it != clients.end();) {
                    if (it->finished->load()) {
                        it->thread.join();
                        it = clients.erase(it);
                    } else ++it;
                }
                fd_set readable;
                FD_ZERO(&readable);
                FD_SET(server_fd, &readable);
                timeval timeout{};
                timeout.tv_usec = 200000;
                const int ready = select(static_cast<int>(server_fd + 1), &readable, nullptr, nullptr, &timeout);
                if (ready < 0) {
                    if (socketInterrupted()) continue;
                    throw std::runtime_error("select failed");
                }
                if (ready == 0) continue;
                Socket fd = accept(server_fd, nullptr, nullptr);
                if (fd == invalid_socket) {
                    if (socketInterrupted()) continue;
                    throw std::runtime_error("accept failed");
                }
                if (active_clients.load() >= MAX_CLIENTS) {
                    closeSocket(fd);
                    continue;
                }
                auto finished = std::make_shared<std::atomic<bool>>(false);
                // Allocate the slot before starting the thread, keeping failure cleanup safe.
                try { clients.push_back({std::thread{}, finished}); }
                catch (...) { closeSocket(fd); throw; }
                ++active_clients;
                try {
                    clients.back().thread = std::thread([fd, finished] {
                        handleClient(fd);
                        finished->store(true);
                    });
                } catch (...) {
                    clients.pop_back();
                    closeSocket(fd);
                    --active_clients;
                    throw;
                }
            }
        } catch (...) {
            joinAll();
            throw;
        }
        joinAll();
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
    return 0;
}
