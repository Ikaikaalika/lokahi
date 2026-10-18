#include "metal/gpu_cores.h"

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>

#include <algorithm>

namespace lokahi {

int gpu_core_count() {
  int cores = 8;  // an M1's, when the registry does not say
  io_service_t service = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"));
  if (service != IO_OBJECT_NULL) {
    CFTypeRef value = IORegistryEntryCreateCFProperty(service, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
    if (value != nullptr) {
      if (CFGetTypeID(value) == CFNumberGetTypeID()) {
        CFNumberGetValue(static_cast<CFNumberRef>(value), kCFNumberIntType, &cores);
      }
      CFRelease(value);
    }
    IOObjectRelease(service);
  }
  return std::max(1, cores);
}

}  // namespace lokahi
