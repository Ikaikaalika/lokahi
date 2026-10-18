// The GPU's core count, for sizing grids to the Mac they run on.
#pragma once

namespace lokahi {

// AGXAccelerator's "gpu-core-count" from the IO registry (8 on an M1, up to
// 80 on an M3 Ultra), or 8 when the registry does not say.
int gpu_core_count();

}  // namespace lokahi
