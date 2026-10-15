// swift-tools-version:5.9
// The Swift side of the Common Compute boundary: the frozen wire v1
// reference types (lokahi-wire-v1/swift, one source of truth, unmodified)
// and an XPC-exportable service backed by the native engine's C ABI.
//
// Build the engine first, then point the linker at it:
//   cmake -S native/engine -B build/engine && cmake --build build/engine
//   swift test --package-path integrations/commoncompute \
//     -Xlinker -L$PWD/build/engine -Xlinker -rpath -Xlinker $PWD/build/engine
import PackageDescription

let package = Package(
    name: "LokahiCommonCompute",
    platforms: [.macOS(.v13)],
    products: [
        .library(name: "LokahiWireV1", targets: ["LokahiWireV1"]),
        .library(name: "LokahiXPC", targets: ["LokahiXPC"]),
    ],
    targets: [
        .systemLibrary(name: "CLokahi", path: "lokahi-xpc/CLokahi"),
        .target(name: "LokahiWireV1", path: "lokahi-wire-v1/swift", exclude: ["VerifyFixtures.swift"]),
        .target(name: "LokahiXPC", dependencies: ["CLokahi", "LokahiWireV1"], path: "lokahi-xpc/Sources/LokahiXPC"),
        .testTarget(
            name: "LokahiXPCTests",
            dependencies: ["LokahiXPC", "LokahiWireV1"],
            path: "lokahi-xpc/Tests/LokahiXPCTests"
        ),
    ]
)
