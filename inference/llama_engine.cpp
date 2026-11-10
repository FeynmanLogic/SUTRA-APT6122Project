/*
Written by Dhruv Kulkarni, 6122APT, last modified on Nov 10
*/
#include "llama_engine.hpp"

#include <chrono>
#include <exception>
#include <stdexcept>
#include <vector>

namespace sutra {

LlamaEngine::LlamaEngine(
    int gpu_layers,
    std::uint32_t context_size
)
    : gpu_layers_(gpu_layers),
      context_size_(context_size) {}

LlamaEngine::~LlamaEngine() {
    releaseResources();
}

void LlamaEngine::releaseResources() {
    if (context_ != nullptr) {
        llama_free(context_);
        context_ = nullptr;
    }

    if (model_ != nullptr) {
        llama_model_free(model_);
        model_ = nullptr;
    }

    vocabulary_ = nullptr;

    if (backend_initialized_) {
        llama_backend_free();
        backend_initialized_ = false;
    }
}

bool LlamaEngine::loadModel(
    const std::string& model_path
) {
    std::lock_guard<std::mutex> lock(inference_mutex_);

    releaseResources();

    llama_backend_init();
    backend_initialized_ = true;

    llama_model_params model_parameters =
        llama_model_default_params();

    model_parameters.n_gpu_layers = gpu_layers_;

    model_ = llama_model_load_from_file(
        model_path.c_str(),
        model_parameters
    );

    if (model_ == nullptr) {
        releaseResources();
        return false;
    }

    vocabulary_ = llama_model_get_vocab(model_);

    llama_context_params context_parameters =
        llama_context_default_params();

    context_parameters.n_ctx = context_size_;
    context_parameters.n_batch = context_size_;
    context_parameters.no_perf = false;

    context_ = llama_init_from_model(
        model_,
        context_parameters
    );

    if (context_ == nullptr) {
        releaseResources();
        return false;
    }

    return true;
}

void LlamaEngine::setBatchTokens(
    llama_batch_ext* batch,
    const llama_token* tokens,
    std::int32_t token_count,
    llama_pos starting_position
) {
    llama_batch_ext_clear(batch);

    for (std::int32_t index = 0;
         index < token_count;
         ++index) {
        const std::int32_t batch_index =
            llama_batch_ext_add_token(
                batch,
                0,
                tokens[index]
            );

        const llama_pos position =
            starting_position + index;

        llama_batch_ext_set_pos(
            batch,
            batch_index,
            &position
        );
    }

    llama_batch_ext_set_output_logits(
        batch,
        token_count - 1,
        true
    );
}

std::string LlamaEngine::tokenToText(
    llama_token token
) const {
    std::vector<char> buffer(128);

    int length = llama_token_to_piece(
        vocabulary_,
        token,
        buffer.data(),
        static_cast<int>(buffer.size()),
        0,
        true
    );

    if (length < 0) {
        buffer.resize(static_cast<std::size_t>(-length));

        length = llama_token_to_piece(
            vocabulary_,
            token,
            buffer.data(),
            static_cast<int>(buffer.size()),
            0,
            true
        );
    }

    if (length < 0) {
        throw std::runtime_error(
            "Failed to convert token to text."
        );
    }

    return std::string(
        buffer.data(),
        static_cast<std::size_t>(length)
    );
}

std::string LlamaEngine::formatPrompt(
    const std::string& prompt
) const {
    if (model_ == nullptr) {
        throw std::runtime_error(
            "Cannot format prompt without a loaded model."
        );
    }

    const char* chat_template =
        llama_model_chat_template(
            model_,
            nullptr
        );

    if (chat_template == nullptr) {
        throw std::runtime_error(
            "The model does not contain a chat template."
        );
    }

    const llama_chat_message message = {
        "user",
        prompt.c_str()
    };

    int formatted_length =
        llama_chat_apply_template(
            chat_template,
            &message,
            1,
            true,
            nullptr,
            0
        );

    if (formatted_length < 0) {
        throw std::runtime_error(
            "Failed to determine formatted prompt size."
        );
    }

    std::vector<char> formatted_buffer(
        static_cast<std::size_t>(
            formatted_length
        )
    );

    formatted_length =
        llama_chat_apply_template(
            chat_template,
            &message,
            1,
            true,
            formatted_buffer.data(),
            static_cast<std::int32_t>(
                formatted_buffer.size()
            )
        );

    if (formatted_length < 0) {
        throw std::runtime_error(
            "Failed to apply the model chat template."
        );
    }

    if (
        formatted_length >
        static_cast<int>(formatted_buffer.size())
    ) {
        formatted_buffer.resize(
            static_cast<std::size_t>(
                formatted_length
            )
        );

        formatted_length =
            llama_chat_apply_template(
                chat_template,
                &message,
                1,
                true,
                formatted_buffer.data(),
                static_cast<std::int32_t>(
                    formatted_buffer.size()
                )
            );

        if (formatted_length < 0) {
            throw std::runtime_error(
                "Failed to resize formatted prompt."
            );
        }
    }

    return std::string(
        formatted_buffer.data(),
        static_cast<std::size_t>(
            formatted_length
        )
    );
}

//how to handle one single request 

InferenceResponse LlamaEngine::generatesingle(
    const InferenceRequest& request
) {
    std::lock_guard<std::mutex> lock(inference_mutex_);

    InferenceResponse response;
    response.request_id = request.request_id;

    const auto start =
        std::chrono::steady_clock::now();

    if (model_ == nullptr ||
        context_ == nullptr ||
        vocabulary_ == nullptr) {
        response.success = false;
        response.error_message =
            "The model has not been loaded.";
        return response;
    }

    if (request.prompt.empty()) {
        response.success = false;
        response.error_message =
            "The prompt cannot be empty.";
        return response;
    }

    if (request.max_new_tokens <= 0) {
        response.success = false;
        response.error_message =
            "max_new_tokens must be positive.";
        return response;
    }

    llama_sampler* sampler = nullptr;
    llama_batch_ext* batch = nullptr;

    try {
        llama_memory_clear(
            llama_get_memory(context_),
            true
        );
        const std::string formatted_prompt =
    formatPrompt(request.prompt);
        const int prompt_token_count =
            -llama_tokenize(
                vocabulary_,
               formatted_prompt.c_str(),
formatted_prompt.size(),
                nullptr,
                0,
                true,
                true
            );

        if (prompt_token_count <= 0) {
            throw std::runtime_error(
                "Failed to determine prompt token count."
            );
        }

        if (
            prompt_token_count +
            request.max_new_tokens >
            static_cast<int>(context_size_)
        ) {
            throw std::runtime_error(
                "Prompt and output exceed context size."
            );
        }

        std::vector<llama_token> prompt_tokens(
            prompt_token_count
        );

        const int tokenization_result =
            llama_tokenize(
                vocabulary_,
                formatted_prompt.c_str(),
        formatted_prompt.size(),
                prompt_tokens.data(),
                prompt_tokens.size(),
                true,
                true
            );

        if (tokenization_result < 0) {
            throw std::runtime_error(
                "Prompt tokenization failed."
            );
        }

        response.input_tokens = prompt_token_count;

        llama_sampler_chain_params sampler_parameters =
            llama_sampler_chain_default_params();

        sampler_parameters.no_perf = false;

        sampler = llama_sampler_chain_init(
            sampler_parameters
        );

        llama_sampler_chain_add(
            sampler,
            llama_sampler_init_greedy()
        );

        batch = llama_batch_ext_init(context_);

        setBatchTokens(
            batch,
            prompt_tokens.data(),
            prompt_token_count,
            0
        );

        if (
            llama_process(
                context_,
                LLAMA_PROCESS_TYPE_DECODE,
                batch
            ) != 0
        ) {
            throw std::runtime_error(
                "Prompt evaluation failed."
            );
        }

        llama_pos next_position =
            prompt_token_count;

        bool first_token_recorded = false;

        while (
            response.generated_tokens <
            request.max_new_tokens
        ) {
            const llama_token next_token =
                llama_sampler_sample(
                    sampler,
                    context_,
                    -1
                );

            if (
                llama_vocab_is_eog(
                    vocabulary_,
                    next_token
                )
            ) {
                break;
            }

            response.generated_text +=
                tokenToText(next_token);

            ++response.generated_tokens;

            if (!first_token_recorded) {
                const auto first_token_time =
                    std::chrono::steady_clock::now();

                response.time_to_first_token_ms =
                    std::chrono::duration<
                        double,
                        std::milli
                    >(
                        first_token_time - start
                    ).count();

                first_token_recorded = true;
            }

            setBatchTokens(
                batch,
                &next_token,
                1,
                next_position
            );

            if (
                llama_process(
                    context_,
                    LLAMA_PROCESS_TYPE_DECODE,
                    batch
                ) != 0
            ) {
                throw std::runtime_error(
                    "Token evaluation failed."
                );
            }

            ++next_position;
        }

        const auto finish =
            std::chrono::steady_clock::now();

        response.inference_time_ms =
            std::chrono::duration<
                double,
                std::milli
            >(finish - start).count();

        response.latency_ms =
            response.inference_time_ms;

        if (response.inference_time_ms > 0.0) {
            response.tokens_per_second =
                response.generated_tokens /
                (response.inference_time_ms / 1000.0);
        }

        response.success = true;
    } catch (const std::exception& exception) {
        response.success = false;
        response.error_message = exception.what();
    }

    if (batch != nullptr) {
        llama_batch_ext_free(batch);
    }

    if (sampler != nullptr) {
        llama_sampler_free(sampler);
    }

    return response;
}

}  // namespace sutra