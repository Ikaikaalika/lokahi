# Chat template fixtures

The real chat templates of the models Lōkahi targets. `tests/test_chat_template.py`
uses them as correctness evidence: the native renderer
(`native/engine/src/chat_template.cpp`) has to match jinja2, set up as
transformers sets it up, byte for byte.

Each directory is laid out like a Hugging Face snapshot, so the native loader
reads it unchanged. A `chat_template.jinja` file is stored verbatim. A
`tokenizer_config.json` file is trimmed to `bos_token`, `chat_template` and
`eos_token`, with values copied unchanged from upstream and written as
`json.dumps(..., indent=2, ensure_ascii=False)` plus a newline. Where upstream
ships `chat_template.jinja`, the trimmed config keeps only the special tokens.
The exception is GLM-4, whose config keeps its own inline template so the
loader's precedence is tested; transformers uses the `.jinja` file. No weights
or tokenizers are stored.

The files were fetched on 2026-10-15 from the pinned revisions through the
Hugging Face Hub file API. Each fetch was checked against the byte size the Hub
reports: the whole file for small files, and an exact byte range for the tail
of large configs, which is where `chat_template` and the special tokens are.

| Directory | Repository | Revision | Upstream files | License |
|---|---|---|---|---|
| `qwen3` | `mlx-community/Qwen3-0.6B-4bit` | `73e3e38d981303bc594367cd910ea6eb48349da8` | `tokenizer_config.json` (9706 B, sha256 `253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0`) | Apache-2.0 |
| `llama3.2` | `mlx-community/Llama-3.2-1B-Instruct-4bit` | `08231374eeacb049a0eade7922910865b8fce912` | `tokenizer_config.json` (54558 B; bytes 49000 to the end read) | Llama 3.2 Community License |
| `gemma3` | `mlx-community/gemma-3-1b-it-qat-4bit` | `15fed4eafb456c6fcb2a1165f19ac609670ed14b` | `tokenizer_config.json` (1156959 B; bytes 1153959 to the end read) | Gemma Terms of Use |
| `mistral-v0.3` | `mistralai/Mistral-7B-Instruct-v0.3` | `e0bc86c23ce5aae1db576c8cca6f06f1f73af2db` | `tokenizer_config.json` (140874 B; bytes 134874 to the end read) | Apache-2.0 |
| `deepseek-r1-distill-qwen` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` | `tokenizer_config.json` (3071 B, sha256 `8ac8c85fb242563c2260baec0909debd69d718af6a0b3d90e6cab62b4d341cd5`) | MIT |
| `qwen2.5` | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | `tokenizer_config.json` (7305 B, sha256 `5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583`) | Apache-2.0 |
| `smollm3` | `HuggingFaceTB/SmolLM3-3B` | `a07cc9a04f16550a088caea529712d1d335b0ac1` | `chat_template.jinja` (5598 B); `tokenizer_config.json` (50387 B; bytes 48000 to the end read) | Apache-2.0 |
| `phi4-mini` | `microsoft/Phi-4-mini-instruct` | `5a149550068a1eb93398160d8953f5f56c3603e9` | `tokenizer_config.json` (2932 B, sha256 `9c9b6bc0c94d95f69f826c41069a3e8b387ac3ced89601d201886e99240ac9db`) | MIT |
| `glm4-0414` | `zai-org/GLM-4-9B-0414` | `645b8482494e31b6b752272bf7f7f273ef0f3caf` | `chat_template.jinja` (1015 B); `tokenizer_config.json` (4170 B, sha256 `e316474d867ad1f91165a12d90c73b27a7e0f9a1c9293831cca66c792bfe9a36`) | MIT |
| `gpt-oss` | `openai/gpt-oss-20b` | `6cee5e81ee83917806bbde320786a8fb61efebee` | `chat_template.jinja` (16738 B); `tokenizer_config.json` (4200 B) | Apache-2.0 |

## Stored files

| File | sha256 | Content |
|---|---|---|
| `qwen3/tokenizer_config.json` | `d2a9575e80943c677eda0e88c8d2ba3f9ca401fb5cde2f6b2200ab94274618de` | trimmed config |
| `llama3.2/tokenizer_config.json` | `dfa5933dc6fea3d50f65a47bf99b406c8757354f7e34d28beefdc808b900156c` | trimmed config |
| `gemma3/tokenizer_config.json` | `95fc50a5646b501c33db428728d428f872ae57ae9a67cc979b8ebc3a4558e45d` | trimmed config |
| `mistral-v0.3/tokenizer_config.json` | `fe19883ec46e9e313a30f0080f63d687f05e42b78c364a9ca7a1ef789a85da9d` | trimmed config |
| `deepseek-r1-distill-qwen/tokenizer_config.json` | `ffb12c883b67bd54ed30926d36b90502332df7235bd5a32c2d5cde1491b70c2e` | trimmed config; special tokens kept in their `{"__type": "AddedToken", "content": ...}` form |
| `qwen2.5/tokenizer_config.json` | `7beeda907c7be18da85df2139a6570ab228d83a658098ca151277ce5570bfe83` | trimmed config |
| `smollm3/chat_template.jinja` | `3e0e275a527f249670056e0078e474eadbdd007c134b4fdfae5a7b3089702814` | verbatim |
| `smollm3/tokenizer_config.json` | `2bb422a5f7f8f54a161a1ad8933616adc274ca018570fc66eb2748bece845615` | special tokens only |
| `phi4-mini/tokenizer_config.json` | `a154363cec4802a3985f62f3386b332ca274cf097780499ff31bf7d2bab3826e` | trimmed config |
| `glm4-0414/chat_template.jinja` | `db700f25fa300e53634c6fc78dee86b7fbd6d27e624edb855b18a4078c83a822` | verbatim |
| `glm4-0414/tokenizer_config.json` | `e4933bf6b5bc5949991803c5abf11138c3fd662d0c9be17d4a57736c00d8bb7a` | trimmed config, inline template kept |
| `gpt-oss/chat_template.jinja` | `a4c9919cbbd4acdd51ccffe22da049264b1b73e59055fa58811a99efbd7c8146` | verbatim (the harmony template) |
| `gpt-oss/tokenizer_config.json` | `bf84d3a9d87b03340fa45f62c24b27f1263aa5af02511b4caaf2c6911c84bfec` | special tokens only |

The templates remain under their upstream licenses, listed above.
`.gitattributes` exempts the verbatim `.jinja` files from git's whitespace
checks, because SmolLM3's template contains trailing spaces as published.
