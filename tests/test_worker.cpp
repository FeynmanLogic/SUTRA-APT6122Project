#include "worker/worker.hpp"
#include "inference/mock_engine.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
class BrokenEngine final : public sutra::InferenceEngine {
public:
    explicit BrokenEngine(int mode) : mode_(mode) {}
    bool loadModel(const std::string&) override { return true; }
    sutra::InferenceResponse generatesingle(const sutra::InferenceRequest&) override { return {}; }
    sutra::ResponseBatch generatefullBatch(const sutra::RequestBatch& requests) override {
        if (mode_ == 0) throw std::runtime_error("engine failed");
        if (mode_ == 1) return {};
        sutra::ResponseBatch replies(requests.size());
        for (auto& reply : replies) {
            reply.request_id = 999;
            reply.success = true;
            reply.generated_text = "another request's output";
        }
        return replies;
    }
private:
    int mode_;
};
}

int main() {
    try {
        sutra::RequestBatch requests{{1, "first", 8, 0, 0}, {2, "second", 8, 1, 0}};
        sutra::InferenceWorker worker(7, std::make_unique<sutra::MockInferenceEngine>());
        require(worker.getWorkerId() == 7, "worker ID");
        require(worker.processBatch({}).empty(), "empty batch");
        auto replies = worker.processBatch(requests);
        require(replies.size() == 2, "response count");
        for (std::size_t i = 0; i < replies.size(); ++i) {
            require(replies[i].success && replies[i].worker_id == 7, "successful worker response");
            require(replies[i].request_id == requests[i].request_id, "request correlation");
            require(replies[i].generated_text == "Synthetic response to: " + requests[i].prompt,
                    "prompt correlation");
        }
        for (int mode = 0; mode < 3; ++mode) {
            sutra::InferenceWorker broken(3, std::make_unique<BrokenEngine>(mode));
            replies = broken.processBatch(requests);
            require(replies.size() == requests.size(), "one failure per input");
            for (std::size_t i = 0; i < replies.size(); ++i) {
                require(!replies[i].success && !replies[i].error_message.empty(), "explicit failure");
                require(replies[i].request_id == requests[i].request_id, "failed request correlation");
                require(replies[i].worker_id == 3, "failed worker ID");
                require(replies[i].generated_text.empty(), "discard mismatched output");
            }
        }
        bool rejected = false;
        try { sutra::InferenceWorker invalid(0, nullptr); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "null engine rejected");
        rejected = false;
        try { sutra::InferenceWorker invalid(-1, std::make_unique<sutra::MockInferenceEngine>()); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "negative worker ID rejected");
        std::cout << "Worker contract checks passed\n";
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
