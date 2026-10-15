// Helpers shared by the Metal host files (Objective-C++ only).
#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <string>

namespace lokahi {

std::string ns_error(NSError* error);

// The engine's kernels (decoder_kernels.metal), compiled from source.
id<MTLLibrary> compile_kernels(id<MTLDevice> device);

}  // namespace lokahi
