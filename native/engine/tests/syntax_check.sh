#!/bin/sh
# Parse the Objective-C++ Metal host and the Metal kernels with clang on any
# platform, using the stub headers in tests/syntax_stubs. This catches typos
# and C++ type errors early; it is not a substitute for xcrun metal or for
# running the Metal backend on Apple Silicon.
set -eu
ENGINE=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
WORK=${TMPDIR:-/tmp}/lokahi-syntax-check.$$
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT
CXX=${CXX:-clang++}
cmake -DINPUT="$ENGINE/src/metal/decoder_kernels.metal" -DOUTPUT="$WORK/decoder_kernels_source.h" \
  -DSYMBOL=kDecoderKernelSource -P "$ENGINE/cmake/embed_text.cmake"
"$CXX" -std=c++20 -fsyntax-only -x objective-c++ -fobjc-arc -fobjc-runtime=macosx-13.0 \
  -Wall -Wextra -Werror -Wno-unused-parameter \
  -I"$ENGINE/tests/syntax_stubs" -I"$ENGINE/src" -I"$ENGINE/include" -I"$WORK" \
  "$ENGINE/src/metal/metal_decoder.mm" "$ENGINE/src/metal/metal_vision.mm"
"$CXX" -std=c++17 -fsyntax-only -x c++ -Wall -Werror -Wno-unknown-attributes -Wno-unused-parameter \
  '-DLOKAHI_INSTANCE_ATTRS(name)=' -I"$ENGINE/tests/syntax_stubs/msl" \
  "$ENGINE/src/metal/decoder_kernels.metal"
echo "metal syntax check: ok"
