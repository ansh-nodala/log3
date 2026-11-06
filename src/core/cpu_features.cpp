// src/core/cpu_features.cpp
//
// Platform-specific CPU feature detection via cpuid (x86/x64) and auxval / sysctl
// (ARM/ARM64 on Linux, Android, and macOS).

#include "log3/core/cpu_features.h"

#include <cstring>

#if L3_ARCH_X86_FAMILY
#if L3_MSVC_ABI
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

#if L3_OS_LINUX || L3_OS_ANDROID
#include <sys/auxv.h>
#include <unistd.h>
#ifndef HWCAP_ASIMD
#define HWCAP_ASIMD (1 << 3)
#endif
#ifndef HWCAP_FPHP
#define HWCAP_FPHP (1 << 9)
#endif
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif
#ifdef __aarch64__
#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1 << 13)
#endif
#endif
#endif

#if L3_OS_APPLE
#include <sys/sysctl.h>
#endif

namespace l3 {

namespace {

#if L3_ARCH_X86_FAMILY
void detect_x86_features(CpuFeatures& f) noexcept {
  int cpuInfo[4] = {0, 0, 0, 0};

  // Get max supported standard leaf
#if L3_MSVC_ABI
  ::__cpuid(cpuInfo, 0);
#else
  __cpuid(0, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
#endif
  const int max_std_leaf = cpuInfo[0];

  if (max_std_leaf >= 1) {
#if L3_MSVC_ABI
    ::__cpuid(cpuInfo, 1);
#else
    __cpuid(1, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
#endif
    const u32 ecx = static_cast<u32>(cpuInfo[2]);
    const u32 edx = static_cast<u32>(cpuInfo[3]);

    f.has_sse2 = (edx & (1u << 26)) != 0;
    f.has_sse41 = (ecx & (1u << 19)) != 0;
    f.has_sse42 = (ecx & (1u << 20)) != 0;
    f.has_f16c = (ecx & (1u << 29)) != 0;
    f.has_fma = (ecx & (1u << 12)) != 0;
    f.has_avx = (ecx & (1u << 28)) != 0;
  }

  if (max_std_leaf >= 7) {
#if L3_MSVC_ABI
    ::__cpuidex(cpuInfo, 7, 0);
#else
    __cpuid_count(7, 0, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
#endif
    const u32 ebx = static_cast<u32>(cpuInfo[1]);
    const u32 ecx = static_cast<u32>(cpuInfo[2]);

    f.has_avx2 = f.has_avx && ((ebx & (1u << 5)) != 0);
    f.has_avx512f = (ebx & (1u << 16)) != 0;
    f.has_avx512dq = (ebx & (1u << 17)) != 0;
    f.has_avx512bw = (ebx & (1u << 30)) != 0;
    f.has_avx512_vnni = f.has_avx512f && ((ecx & (1u << 11)) != 0);
  }
}
#endif

#if L3_ARCH_ARM_FAMILY
void detect_arm_features(CpuFeatures& f) noexcept {
#if L3_OS_LINUX || L3_OS_ANDROID
  const unsigned long hwcap = getauxval(AT_HWCAP);
  f.has_neon = (hwcap & HWCAP_ASIMD) != 0;
  f.has_neon_fp16 = (hwcap & HWCAP_FPHP) != 0;
  f.has_neon_dotprod = (hwcap & HWCAP_ASIMDDP) != 0;

#ifdef __aarch64__
  const unsigned long hwcap2 = getauxval(AT_HWCAP2);
  f.has_neon_i8mm = (hwcap2 & HWCAP2_I8MM) != 0;
#endif

#elif L3_OS_APPLE
  // Apple Silicon always has NEON, FP16, and DotProd (M1+).
  f.has_neon = true;
  f.has_neon_fp16 = true;
  f.has_neon_dotprod = true;
  
  int val = 0;
  usize size = sizeof(val);
  if (sysctlbyname("hw.optional.arm.FEAT_I8MM", &val, &size, nullptr, 0) == 0) {
    f.has_neon_i8mm = (val != 0);
  }
#else
  f.has_neon = L3_HAS_NEON;
#endif
}
#endif

void detect_cache_and_cores(CpuFeatures& f) noexcept {
  f.cache_line_size = kCacheLineSize;

#if L3_OS_WINDOWS
  const long nprocs = ::sysconf(_SC_NPROCESSORS_ONLN);
  f.logical_cores = nprocs > 0 ? static_cast<int>(nprocs) : 4;
  f.physical_cores = f.logical_cores;
#elif L3_OS_APPLE
  int cpu_count = 1;
  usize size = sizeof(cpu_count);
  if (sysctlbyname("hw.logicalcpu", &cpu_count, &size, nullptr, 0) == 0) {
    f.logical_cores = cpu_count;
  }
  if (sysctlbyname("hw.physicalcpu", &cpu_count, &size, nullptr, 0) == 0) {
    f.physical_cores = cpu_count;
  }
#else
  // Linux / Android fallback
  f.logical_cores = 4;
  f.physical_cores = 4;
#endif
}

}  // namespace

const CpuFeatures& cpu_features() noexcept {
  static CpuFeatures s_features = []() {
    CpuFeatures f{};
    std::memset(&f, 0, sizeof(CpuFeatures));
    f.cache_line_size = kCacheLineSize;

#if L3_ARCH_X86_FAMILY
    detect_x86_features(f);
#elif L3_ARCH_ARM_FAMILY
    detect_arm_features(f);
#endif

    detect_cache_and_cores(f);
    return f;
  }();

  return s_features;
}

}  // namespace l3
