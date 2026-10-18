// Helpers shared by the Metal host files (Objective-C++ only).
#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <string>

namespace lokahi {

std::string ns_error(NSError* error);

// The engine's kernels (decoder_kernels.metal), compiled from source.
id<MTLLibrary> compile_kernels(id<MTLDevice> device);

// Fails unless this GPU's simdgroup matrices hold the elements the kernels
// load and store per lane (qmm, dense_mm, attn_flash, gdn_chunk, ...): lane
// l, two adjacent elements of row (l / 4 & 4) + (l / 2 % 4) from column
// (l / 4 & 2) * 2 + (l % 2) * 2, for loaded matrices and for products alike.
// One dispatch of simdgroup_layout_probe.
void check_fragment_layout(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library);

}  // namespace lokahi
