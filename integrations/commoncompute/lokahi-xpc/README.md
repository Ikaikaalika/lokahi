# Lōkahi XPC service (Common Compute)

The Swift side of the Common Compute boundary. The XPC service links the
native engine's C ABI and exports `LokahiNativeService`, which implements the
frozen `LokahiComputeServiceV1` protocol
([`../lokahi-wire-v1/swift/LokahiXPCV1.swift`](../lokahi-wire-v1/swift/LokahiXPCV1.swift)).
Requests, controls and envelopes cross as bytes. The engine owns the
protocol:

- it decodes `start` and `control` documents strictly;
- it validates every event and receipt before sending it;
- it sends exactly one receipt per admitted request.

The host keeps validating everything it relays, as
[`INTEGRATION.md`](../lokahi-wire-v1/INTEGRATION.md) requires.

| Piece | Where |
|---|---|
| Wire v1 reference types (unmodified, one source of truth) | `../lokahi-wire-v1/swift` → `LokahiWireV1` |
| C ABI module | `CLokahi/module.modulemap` → `native/engine/include/lokahi/lokahi.h` |
| `LokahiNativeService`, `LokahiXPCInterfaceV1`, `LokahiServiceListenerDelegate` | `Sources/LokahiXPC` |
| Gate B tests across a real XPC connection | `Tests/LokahiXPCTests` |

## Using it in the XPC service

```swift
import LokahiXPC

let service = try LokahiNativeService(configuration: configJSON)  // loads the pinned model once
let delegate = LokahiServiceListenerDelegate(service: service)
let listener = NSXPCListener.service()
listener.delegate = delegate
listener.resume()
```

On the host side, set the connection's `remoteObjectInterface` to
`LokahiXPCInterfaceV1.service()`, so the event sink argument travels by
proxy.

`configJSON` lists the models the service may run, by the exact identity a
request must name: `id`, `revision`, `artifact_digest`, and optionally
`adapter_id`. It also gives each model's snapshot directory. See
`native/engine/include/lokahi/lokahi.h`. The service compares the digest it
is given; verifying the artifact against it is still the host's job.

## Behaviour

| Situation | What the host receives |
|---|---|
| Undecodable `start` or `control`, or a `request_id` still in flight | The XPC reply carries an error; no envelope |
| Unknown model, revision or digest; another runtime revision | `rejected` receipt, `unsupported-model-revision` / `runtime-revision-mismatch`, retryable |
| Deadline already passed, prompt plus output over the context, chat template error | `rejected` receipt, `deadline-exceeded` / `context-exceeded` / `template-error`, not retryable |
| Weights over the request's memory ceiling; queue full; draining | `rejected` receipt, `memory-limit` / `overloaded` / `service-draining`, retryable |
| Admitted | `accepted` (sequence 0), then `queue_state` if waiting, then coalesced `text_delta` events (one per `text_flush_ms`), then one receipt |
| Stop string or EOS / `max_output_tokens` reached | `completed`, `stop` / `length` |
| Cancel while queued / while running | `cancelled`, stage `not_started` / `running` or `partial_output`. Running requests stop at the next token |
| Deadline passes while queued or running | `failed`, `deadline-exceeded` |
| Cancel for an unknown or finished request | Nothing (idempotent) |
| `drain` | New requests are rejected; admitted ones finish |

## Testing

```sh
cmake -S native/engine -B build/engine && cmake --build build/engine
PYTHONPATH=src python3 -m lokahi.fixtures.chat_model build/xpc-model
# write build/xpc-config.json listing build/xpc-model (see .github/workflows/ci.yml)
LOKAHI_XPC_TEST_CONFIG=$PWD/build/xpc-config.json swift test --package-path integrations/commoncompute \
  -Xlinker -L$PWD/build/engine -Xlinker -rpath -Xlinker $PWD/build/engine
```

The hosted macOS CI job runs these tests with the CPU backend. The same
behaviour is also tested without Swift, through `lokahi serve` and the C ABI
(`tests/test_service.py`). Every document there is checked by the kit's
reference verifier.
