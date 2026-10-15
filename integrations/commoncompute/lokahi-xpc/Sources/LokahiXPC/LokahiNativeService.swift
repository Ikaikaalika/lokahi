import CLokahi
import Foundation
import LokahiWireV1

/// A failure the native engine reported through `lokahi_last_error`.
public struct LokahiNativeError: Error, CustomStringConvertible, Sendable {
    public let description: String

    static func last(_ call: String) -> LokahiNativeError {
        LokahiNativeError(description: "\(call): \(String(cString: lokahi_last_error()))")
    }
}

/// `LokahiComputeServiceV1` backed by the native engine's wire v1 service.
///
/// Requests and controls pass through as bytes. The engine validates every
/// document it receives and every document it sends back, and sends exactly
/// one receipt per admitted request. Each request's event sink is kept alive until its
/// receipt has been delivered. An XPC service exports one instance per
/// process, through `LokahiServiceListenerDelegate`.
public final class LokahiNativeService: NSObject, LokahiComputeServiceV1 {
    private let handle: OpaquePointer

    /// `configuration` is the engine's service config JSON, described in
    /// `native/engine/include/lokahi/lokahi.h`. Loading takes as long as
    /// loading every listed model.
    public init(configuration: Data) throws {
        var service: OpaquePointer?
        let status = configuration.withUnsafeBytes { bytes in
            lokahi_service_create(bytes.baseAddress?.assumingMemoryBound(to: CChar.self), bytes.count, &service)
        }
        guard status == 0, let service else { throw LokahiNativeError.last("lokahi_service_create") }
        handle = service
        super.init()
    }

    /// Cancels outstanding requests and delivers their receipts first.
    deinit { lokahi_service_free(handle) }

    public func startLokahi(_ request: Data, eventSink: LokahiEventSinkV1, reply: @escaping (Error?) -> Void) {
        let relay = Unmanaged.passRetained(EnvelopeRelay(sink: eventSink)).toOpaque()
        let status = request.withUnsafeBytes { bytes in
            lokahi_service_start(handle, bytes.baseAddress?.assumingMemoryBound(to: CChar.self), bytes.count,
                                 relayEnvelope, relay)
        }
        if status != 0 {
            // Nothing was admitted, so no envelope will reference the relay.
            Unmanaged<EnvelopeRelay>.fromOpaque(relay).release()
            reply(LokahiNativeError.last("startLokahi"))
            return
        }
        reply(nil)
    }

    public func controlLokahi(_ control: Data, reply: @escaping (Error?) -> Void) {
        let status = control.withUnsafeBytes { bytes in
            lokahi_service_control(handle, bytes.baseAddress?.assumingMemoryBound(to: CChar.self), bytes.count)
        }
        reply(status == 0 ? nil : LokahiNativeError.last("controlLokahi"))
    }
}

/// Forwards one request's envelopes to its sink; released after the receipt.
private final class EnvelopeRelay {
    let sink: LokahiEventSinkV1

    init(sink: LokahiEventSinkV1) { self.sink = sink }
}

private let relayEnvelope: lokahi_envelope_fn = { context, envelope, length in
    guard let context, let envelope else { return }
    let relay = Unmanaged<EnvelopeRelay>.fromOpaque(context)
    let data = Data(bytes: envelope, count: length)
    relay.takeUnretainedValue().sink.receiveLokahiEnvelope(data)
    if let decoded = try? LokahiWireDecoderV1.decodeEnvelope(data), case .receipt = decoded {
        relay.release()
    }
}

/// The NSXPC interfaces for the service protocol. The event sink argument
/// travels by proxy, so the service calls back into the host's object.
public enum LokahiXPCInterfaceV1 {
    public static func service() -> NSXPCInterface {
        let interface = NSXPCInterface(with: LokahiComputeServiceV1.self)
        interface.setInterface(
            NSXPCInterface(with: LokahiEventSinkV1.self),
            for: #selector(LokahiComputeServiceV1.startLokahi(_:eventSink:reply:)),
            argumentIndex: 1,
            ofReply: false
        )
        return interface
    }
}

/// Exports one `LokahiNativeService` on every accepted connection. In the
/// XPC service's entry point, set it as `NSXPCListener.service()`'s
/// delegate.
public final class LokahiServiceListenerDelegate: NSObject, NSXPCListenerDelegate {
    private let service: LokahiNativeService

    public init(service: LokahiNativeService) {
        self.service = service
    }

    public func listener(_ listener: NSXPCListener, shouldAcceptNewConnection connection: NSXPCConnection) -> Bool {
        connection.exportedInterface = LokahiXPCInterfaceV1.service()
        connection.exportedObject = service
        connection.resume()
        return true
    }
}
