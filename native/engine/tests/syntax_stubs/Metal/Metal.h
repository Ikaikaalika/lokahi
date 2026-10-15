// Syntax-check stub for Linux CI only (see Foundation.h stub).
#pragma once
#import <Foundation/Foundation.h>
typedef struct { NSUInteger width, height, depth; } MTLSize;
static inline MTLSize MTLSizeMake(NSUInteger w, NSUInteger h, NSUInteger d) { MTLSize s = {w, h, d}; return s; }
typedef NSUInteger MTLResourceOptions;
enum { MTLResourceStorageModeShared = 0, MTLResourceStorageModePrivate = 32, MTLResourceHazardTrackingModeUntracked = 256 };
typedef NSUInteger MTLDataType;
enum { MTLDataTypeInt = 29, MTLDataTypeBool = 53 };
typedef NSUInteger MTLLanguageVersion;
enum { MTLLanguageVersion3_0 = 196608 };
typedef NSUInteger MTLCommandBufferStatus;
enum { MTLCommandBufferStatusCompleted = 4 };
@protocol MTLBuffer
- (void*)contents;
@end
@protocol MTLFunction
@end
@protocol MTLComputePipelineState
- (NSUInteger)maxTotalThreadsPerThreadgroup;
@end
@protocol MTLComputeCommandEncoder
- (void)setComputePipelineState:(id<MTLComputePipelineState>)state;
- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index;
- (void)setBytes:(const void*)bytes length:(NSUInteger)length atIndex:(NSUInteger)index;
- (void)dispatchThreadgroups:(MTLSize)groups threadsPerThreadgroup:(MTLSize)threads;
- (void)dispatchThreads:(MTLSize)threads threadsPerThreadgroup:(MTLSize)group;
- (void)endEncoding;
@end
@protocol MTLCommandBuffer
- (id<MTLComputeCommandEncoder>)computeCommandEncoder;
- (void)commit;
- (void)waitUntilCompleted;
- (MTLCommandBufferStatus)status;
- (NSError*)error;
- (double)GPUStartTime;  // CFTimeInterval
- (double)GPUEndTime;
@end
@protocol MTLCommandQueue
- (id<MTLCommandBuffer>)commandBuffer;
@end
@interface MTLFunctionConstantValues : NSObject
- (void)setConstantValue:(const void*)value type:(MTLDataType)type atIndex:(NSUInteger)index;
@end
@interface MTLCompileOptions : NSObject
@property MTLLanguageVersion languageVersion;
@end
@protocol MTLLibrary
- (id<MTLFunction>)newFunctionWithName:(NSString*)name;
- (id<MTLFunction>)newFunctionWithName:(NSString*)name constantValues:(MTLFunctionConstantValues*)values error:(NSError**)error;
@end
@protocol MTLDevice
- (id<MTLCommandQueue>)newCommandQueue;
- (id<MTLLibrary>)newLibraryWithSource:(NSString*)source options:(MTLCompileOptions*)options error:(NSError**)error;
- (id<MTLComputePipelineState>)newComputePipelineStateWithFunction:(id<MTLFunction>)function error:(NSError**)error;
- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options;
- (id<MTLBuffer>)newBufferWithBytes:(const void*)pointer length:(NSUInteger)length options:(MTLResourceOptions)options;
- (NSUInteger)maxBufferLength;
- (NSString*)name;
@end
id<MTLDevice> MTLCreateSystemDefaultDevice(void);
