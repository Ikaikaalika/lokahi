/*
 * Lokahi C ABI.
 *
 * Copyright (c) 2025-2026 Common Compute LLC.
 * Licensed under the PolyForm Noncommercial License 1.0.0; see LICENSE.
 *
 * All functions return 0 on success and a nonzero code on failure; call
 * lokahi_last_error() on the same thread for a description. A model handle
 * is not thread-safe; serialize calls per handle.
 */
#ifndef LOKAHI_H
#define LOKAHI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LOKAHI_ABI_VERSION 2

typedef struct lokahi_model lokahi_model;

typedef struct {
  const char* backend; /* "auto", "cpu", or "metal"; NULL means "auto" */
  int32_t max_context; /* tokens; <= 0 means 4096 */
  int32_t prefill_chunk; /* tokens per prefill submission; <= 0 means 512 */
  int32_t cpu_threads; /* <= 0 means hardware concurrency */
} lokahi_load_options;

typedef struct {
  double prefill_seconds; /* admitted prompt to first token available */
  double decode_seconds;  /* first token to last token available */
  int32_t generated_tokens;
} lokahi_timing;

int32_t lokahi_abi_version(void);
const char* lokahi_last_error(void);

int lokahi_model_load(const char* directory, const lokahi_load_options* options,
                      lokahi_model** out_model);
void lokahi_model_free(lokahi_model* model);

const char* lokahi_model_backend(const lokahi_model* model);
int32_t lokahi_model_vocab_size(const lokahi_model* model);
int32_t lokahi_model_position(const lokahi_model* model);
void lokahi_model_reset(lokahi_model* model);

/* Append tokens; copies the last token's logits into logits_out when it is
 * not NULL (vocab_size floats). */
int lokahi_prefill(lokahi_model* model, const int32_t* tokens, int32_t count, float* logits_out);
int lokahi_decode(lokahi_model* model, int32_t token, float* logits_out);

/* Greedy generation from the current position. Writes up to max_new tokens
 * and stores the count in *out_count. */
int lokahi_generate_greedy(lokahi_model* model, const int32_t* prompt, int32_t prompt_count,
                           int32_t max_new, int32_t stop_at_eos, int32_t* out_tokens,
                           int32_t* out_count, lokahi_timing* timing);

/* Seeded sampling with mlx_lm semantics: top-p, min-p and top-k filter the
 * untempered distribution, then a token is drawn with probability
 * proportional to exp(logit / temperature). */
typedef struct {
  float temperature; /* >= 0; 0 is greedy */
  float top_p;       /* (0, 1]; 1 disables */
  int32_t top_k;     /* >= 0; 0 disables */
  float min_p;       /* [0, 1]; 0 disables */
  uint64_t seed;
} lokahi_sampling;

/* Generation from the current position; `sampling` NULL is greedy. Generated
 * token i is drawn with noise keyed by (seed, i, token id), so a seed gives
 * the same tokens on every backend. Writes up to max_new tokens. */
int lokahi_generate(lokahi_model* model, const int32_t* prompt, int32_t prompt_count, int32_t max_new,
                    const lokahi_sampling* sampling, int32_t stop_at_eos, int32_t* out_tokens,
                    int32_t* out_count, lokahi_timing* timing);

/* The host reference sampler: one token from `vocab` logits at noise step `step`. */
int lokahi_sample(const float* logits, int32_t vocab, const lokahi_sampling* sampling, int32_t step,
                  int32_t* out_token);

/* Tokenizer for <directory>/tokenizer.json (Hugging Face format). */
typedef struct lokahi_tokenizer lokahi_tokenizer;

int lokahi_tokenizer_load(const char* directory, lokahi_tokenizer** out_tokenizer);
void lokahi_tokenizer_free(lokahi_tokenizer* tokenizer);

/* Encodes `length` bytes of UTF-8 text. Always stores the full token count in
 * *out_count; fails without writing ids when it exceeds `capacity`. */
int lokahi_tokenize(const lokahi_tokenizer* tokenizer, const char* text, size_t length,
                    int32_t add_special_tokens, int32_t* out_ids, int32_t capacity,
                    int32_t* out_count);

/* Decodes ids to UTF-8. Always stores the byte length (excluding the NUL) in
 * *out_length; fails without writing when length + 1 exceeds `capacity`. */
int lokahi_detokenize(const lokahi_tokenizer* tokenizer, const int32_t* ids, int32_t count,
                      int32_t skip_special_tokens, char* out_text, size_t capacity,
                      size_t* out_length);

/* The Lōkahi wire v1 service: the Common Compute boundary. Requests are
 * encoded `start` and `control` documents; the service answers with encoded
 * `event` and `receipt` documents (integrations/commoncompute/lokahi-wire-v1),
 * each validated before it is delivered, and exactly one receipt per request.
 *
 * config_json is a UTF-8 JSON object:
 *   {"models": [{"id", "revision", "artifact_digest", "adapter_id"?, "directory"}],
 *    "backend"?, "max_context"?, "queue_limit"?, "text_flush_ms"?, "runtime_revision"?} */
typedef struct lokahi_service lokahi_service;

/* Receives one encoded event or receipt: `length` bytes of UTF-8 JSON, not
 * NUL-terminated. Called on a service thread, one call at a time per request;
 * it must not call back into the service. */
typedef void (*lokahi_envelope_fn)(void* context, const char* envelope, size_t length);

int lokahi_service_create(const char* config_json, size_t length, lokahi_service** out_service);

/* Admits a start document. Fails when the document is invalid or its
 * request_id is still in flight (no receipt can be addressed then); every
 * other outcome, rejection included, arrives on `sink`. */
int lokahi_service_start(lokahi_service* service, const char* request, size_t length, lokahi_envelope_fn sink,
                         void* context);

/* Applies a cancel or drain control document. Cancelling an unknown or
 * finished request does nothing. */
int lokahi_service_control(lokahi_service* service, const char* control, size_t length);

/* Cancels outstanding requests, delivers their receipts, and frees the service. */
void lokahi_service_free(lokahi_service* service);

#ifdef __cplusplus
}
#endif

#endif /* LOKAHI_H */
