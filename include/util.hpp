#ifndef _UTIL_H
#define _UTIL_H

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include <vector>
#include <x86intrin.h>

#define SIMD_STATE 4 // 0:none, 2:scalar2x, 4:simd4x
#define SIMD_MODE 1  // 0:naive 1: filter
typedef int PackBase;
#ifdef SI64
typedef long long PackState;
#else
typedef int PackState;
#endif
const int PACK_WIDTH = sizeof(PackState) * 8;
const int PACK_SHIFT = __builtin_ctzll(PACK_WIDTH);
const int PACK_MASK = PACK_WIDTH - 1;

const size_t PARA_DEG_M128 = sizeof(__m128i) / sizeof(PackState);
const size_t PARA_DEG_M256 = sizeof(__m256i) / sizeof(PackState);

const size_t PACK_NODE_POOL_SIZE = 1024000000;

const int CACHE_LINE_SIZE = sysconf(_SC_LEVEL1_DCACHE_LINESIZE); // in byte.

void align_malloc(void **memptr, size_t alignment, size_t size);

#endif