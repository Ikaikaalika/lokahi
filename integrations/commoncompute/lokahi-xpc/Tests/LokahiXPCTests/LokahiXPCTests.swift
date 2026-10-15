// Gate B evidence: the native wire service across a real XPC boundary.
//
// An anonymous NSXPCListener in this process exports LokahiNativeService, and
// the tests talk to it through NSXPCConnection proxies, so every request,
// control, reply and envelope is serialized by XPC. The host-side sink
// decodes each envelope with the frozen reference decoder and enforces the
// reference stream rules (LokahiStreamStateV1).
//
// LOKAHI_XPC_TEST_CONFIG names a service config for a generated fixture
// model (python -m lokahi.fixtures.chat_model DIR, then a config listing it);
// without it the tests skip.
import Foundation
import LokahiWireV1
import LokahiXPC
import XCTest

final class CollectingSink: NSObject, LokahiEventSinkV1 {
    private let lock = NSLock()
    private var states: [String: LokahiStreamStateV1] = [:]
    private var events: [String: [LokahiEventV1]] = [:]
    private var receipts: [String: LokahiReceiptV1] = [:]
    private var waiting: [String: XCTestExpectation] = [:]
    private(set) var violations: [String] = []

    func expectReceipt(_ requestID: String, _ test: XCTestCase) -> XCTestExpectation {
        lock.lock()
        defer { lock.unlock() }
        let expectation = test.expectation(description: "receipt for \(requestID)")
        if receipts[requestID] != nil { expectation.fulfill() } else { waiting[requestID] = expectation }
        return expectation
    }

    func waitForEvent(_ requestID: String, kind: LokahiEventKindV1, timeout: TimeInterval) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            lock.lock()
            let seen = events[requestID, default: []].contains { $0.kind == kind }
            lock.unlock()
            if seen { return true }
            Thread.sleep(forTimeInterval: 0.005)
        }
        return false
    }

    func receiveLokahiEnvelope(_ envelope: Data) {
        lock.lock()
        defer { lock.unlock() }
        do {
            switch try LokahiWireDecoderV1.decodeEnvelope(envelope) {
            case .event(let event):
                var state = states[event.requestID] ?? LokahiStreamStateV1(requestID: event.requestID)
                try state.accept(event)
                states[event.requestID] = state
                events[event.requestID, default: []].append(event)
            case .receipt(let receipt):
                var state = states[receipt.requestID] ?? LokahiStreamStateV1(requestID: receipt.requestID)
                try state.accept(receipt)
                states[receipt.requestID] = state
                receipts[receipt.requestID] = receipt
                waiting.removeValue(forKey: receipt.requestID)?.fulfill()
            default:
                violations.append("the service sent a request-side document")
            }
        } catch {
            violations.append("\(error)")
        }
    }

    func receipt(_ requestID: String) -> LokahiReceiptV1? {
        lock.lock()
        defer { lock.unlock() }
        return receipts[requestID]
    }

    func text(_ requestID: String) -> String {
        lock.lock()
        defer { lock.unlock() }
        return events[requestID, default: []].compactMap(\.textDelta).joined()
    }

    func kinds(_ requestID: String) -> [LokahiEventKindV1] {
        lock.lock()
        defer { lock.unlock() }
        return events[requestID, default: []].map(\.kind)
    }
}

final class LokahiXPCTests: XCTestCase {
    private var config: [String: Any] = [:]
    private var service: LokahiNativeService?
    private var delegate: LokahiServiceListenerDelegate?
    private var listener: NSXPCListener?

    override func setUpWithError() throws {
        guard let path = ProcessInfo.processInfo.environment["LOKAHI_XPC_TEST_CONFIG"] else {
            throw XCTSkip("set LOKAHI_XPC_TEST_CONFIG to a service config for a fixture model")
        }
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        config = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let service = try LokahiNativeService(configuration: data)
        let delegate = LokahiServiceListenerDelegate(service: service)
        let listener = NSXPCListener.anonymous()
        listener.delegate = delegate
        listener.resume()
        (self.service, self.delegate, self.listener) = (service, delegate, listener)
    }

    override func tearDown() {
        listener?.invalidate()
        (service, delegate, listener) = (nil, nil, nil)
    }

    private func connect() -> (NSXPCConnection, LokahiComputeServiceV1) {
        let connection = NSXPCConnection(listenerEndpoint: listener!.endpoint)
        connection.remoteObjectInterface = LokahiXPCInterfaceV1.service()
        connection.resume()
        let proxy = connection.remoteObjectProxyWithErrorHandler { error in
            XCTFail("XPC connection error: \(error)")
        } as! LokahiComputeServiceV1
        return (connection, proxy)
    }

    /// A start document for the configured fixture model, from the kit's golden fixture.
    private func startDocument(_ requestID: String, maxOutputTokens: Int, model: [String: Any]? = nil) throws -> Data {
        let kit = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent(
            "../../../lokahi-wire-v1/fixtures/valid/start-greedy.json")
        var doc = try JSONSerialization.jsonObject(with: Data(contentsOf: kit)) as! [String: Any]
        let configured = (config["models"] as! [[String: Any]])[0]
        var identity = doc["model"] as! [String: Any]
        for key in ["id", "revision", "artifact_digest", "adapter_id"] { identity[key] = configured[key] }
        for (key, value) in model ?? [:] { identity[key] = value }
        let now = Int64(Date().timeIntervalSince1970 * 1000)
        doc["request_id"] = requestID
        doc["task_id"] = "task-\(requestID)"
        doc["model"] = identity
        doc["runtime_revision"] = config["runtime_revision"]
        doc["created_at_unix_ms"] = now
        doc["deadline_unix_ms"] = now + 120_000
        var objective = doc["objective"] as! [String: Any]
        objective["context_tokens"] = 4096
        objective["max_output_tokens"] = maxOutputTokens
        doc["objective"] = objective
        return try JSONSerialization.data(withJSONObject: doc)
    }

    private func controlDocument(_ action: String, target: String?) throws -> Data {
        var doc: [String: Any] = [
            "schema_version": 1, "message_type": "control", "control_id": "control-\(UUID().uuidString)",
            "action": action, "issued_at_unix_ms": Int64(Date().timeIntervalSince1970 * 1000),
        ]
        if let target { doc["target_request_id"] = target }
        return try JSONSerialization.data(withJSONObject: doc)
    }

    @discardableResult
    private func call(_ body: (@escaping (Error?) -> Void) -> Void) -> Error? {
        let replied = expectation(description: "reply")
        var result: Error?
        body { error in
            result = error
            replied.fulfill()
        }
        wait(for: [replied], timeout: 30)
        return result
    }

    func testCompletesARequestOverXPC() throws {
        let (connection, proxy) = connect()
        defer { connection.invalidate() }
        let sink = CollectingSink()
        let done = sink.expectReceipt("req-xpc-complete", self)
        XCTAssertNil(call { proxy.startLokahi(try! self.startDocument("req-xpc-complete", maxOutputTokens: 16),
                                              eventSink: sink, reply: $0) })
        wait(for: [done], timeout: 120)
        let receipt = try XCTUnwrap(sink.receipt("req-xpc-complete"))
        XCTAssertEqual(receipt.status, .completed)
        XCTAssertEqual(receipt.outputText, sink.text("req-xpc-complete"))
        XCTAssertEqual(receipt.usage.totalTokens, receipt.usage.promptTokens + receipt.usage.completionTokens)
        XCTAssertGreaterThan(receipt.usage.completionTokens, 0)
        XCTAssertEqual(sink.kinds("req-xpc-complete").first, .accepted)
        XCTAssertEqual(sink.violations, [])
    }

    func testRejectsInvalidAndUnservableRequests() throws {
        let (connection, proxy) = connect()
        defer { connection.invalidate() }
        let sink = CollectingSink()
        // Undecodable documents fail the call itself: no receipt can be addressed.
        XCTAssertNotNil(call { proxy.startLokahi(Data("{}".utf8), eventSink: sink, reply: $0) })
        XCTAssertNotNil(call { proxy.controlLokahi(Data("[]".utf8), reply: $0) })
        let rejected = sink.expectReceipt("req-xpc-unknown", self)
        let unknown = try startDocument("req-xpc-unknown", maxOutputTokens: 8, model: ["revision": String(repeating: "c", count: 40)])
        XCTAssertNil(call { proxy.startLokahi(unknown, eventSink: sink, reply: $0) })
        wait(for: [rejected], timeout: 30)
        let receipt = try XCTUnwrap(sink.receipt("req-xpc-unknown"))
        XCTAssertEqual(receipt.status, .rejected)
        XCTAssertEqual(receipt.terminalSequence, 0)
        XCTAssertEqual(receipt.failure?.stage, .notStarted)
        XCTAssertEqual(sink.violations, [])
    }

    func testCancelsRunningAndQueuedRequests() throws {
        let (connection, proxy) = connect()
        defer { connection.invalidate() }
        let sink = CollectingSink()
        let running = sink.expectReceipt("req-xpc-running", self)
        let queued = sink.expectReceipt("req-xpc-queued", self)
        XCTAssertNil(call { proxy.startLokahi(try! self.startDocument("req-xpc-running", maxOutputTokens: 4000),
                                              eventSink: sink, reply: $0) })
        XCTAssertNil(call { proxy.startLokahi(try! self.startDocument("req-xpc-queued", maxOutputTokens: 8),
                                              eventSink: sink, reply: $0) })
        XCTAssertTrue(sink.waitForEvent("req-xpc-running", kind: .textDelta, timeout: 60))
        XCTAssertNil(call { proxy.controlLokahi(try! self.controlDocument("cancel", target: "req-xpc-queued"), reply: $0) })
        XCTAssertNil(call { proxy.controlLokahi(try! self.controlDocument("cancel", target: "req-xpc-running"), reply: $0) })
        wait(for: [running, queued], timeout: 120)
        XCTAssertEqual(sink.receipt("req-xpc-queued")?.status, .cancelled)
        XCTAssertEqual(sink.receipt("req-xpc-queued")?.failure?.stage, .notStarted)
        let receipt = try XCTUnwrap(sink.receipt("req-xpc-running"))
        XCTAssertEqual(receipt.status, .cancelled)
        XCTAssertLessThan(receipt.usage.completionTokens, 4000)
        // Cancelling a finished request does nothing.
        XCTAssertNil(call { proxy.controlLokahi(try! self.controlDocument("cancel", target: "req-xpc-running"), reply: $0) })
        XCTAssertEqual(sink.violations, [])
    }

    func testServesAgainAfterTheClientReconnects() throws {
        let (first, firstProxy) = connect()
        let sink = CollectingSink()
        let one = sink.expectReceipt("req-xpc-before", self)
        XCTAssertNil(call { firstProxy.startLokahi(try! self.startDocument("req-xpc-before", maxOutputTokens: 4),
                                                   eventSink: sink, reply: $0) })
        wait(for: [one], timeout: 60)
        first.invalidate()
        let (second, secondProxy) = connect()
        defer { second.invalidate() }
        let two = sink.expectReceipt("req-xpc-after", self)
        XCTAssertNil(call { secondProxy.startLokahi(try! self.startDocument("req-xpc-after", maxOutputTokens: 4),
                                                    eventSink: sink, reply: $0) })
        wait(for: [two], timeout: 60)
        XCTAssertEqual(sink.receipt("req-xpc-after")?.status, .completed)
        // Same prompt, greedy: the new connection is served identically.
        XCTAssertEqual(sink.receipt("req-xpc-after")?.outputText, sink.receipt("req-xpc-before")?.outputText)
        XCTAssertEqual(sink.violations, [])
    }
}
