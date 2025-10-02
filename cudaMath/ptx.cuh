#ifndef _PTX_H
#define _PTX_H

#include <cuda_runtime.h>

#define madc_hi(dest, a, x, b) asm volatile("madc.hi.u32 %0, %1, %2, %3;\n\t" : "=r"(dest) : "r"(a), "r"(x), "r"(b))
#define madc_hi_cc(dest, a, x, b) asm volatile("madc.hi.cc.u32 %0, %1, %2, %3;\n\t" : "=r"(dest) : "r"(a), "r"(x), "r"(b))
#define mad_hi_cc(dest, a, x, b) asm volatile("mad.hi.cc.u32 %0, %1, %2, %3;\n\t" : "=r"(dest) : "r"(a), "r"(x), "r"(b))

#define mad_lo_cc(dest, a, x, b) asm volatile("mad.lo.cc.u32 %0, %1, %2, %3;\n\t" : "=r"(dest) : "r"(a), "r"(x), "r"(b))
#define madc_lo(dest, a, x, b) asm volatile("madc.lo.u32 %0, %1, %2, %3;\n\t" : "=r"(dest) : "r"(a), "r"(x), "r"(b))
#define madc_lo_cc(dest, a, x, b) asm volatile("madc.lo.cc.u32 %0, %1, %2, %3;\n\t" : "=r"(dest) : "r"(a), "r"(x), "r"(b))

#define addc(dest, a, b) asm volatile("addc.u32 %0, %1, %2;\n\t" : "=r"(dest) : "r"(a), "r"(b))
#define add_cc(dest, a, b) asm volatile("add.cc.u32 %0, %1, %2;\n\t" : "=r"(dest) : "r"(a), "r"(b))
#define addc_cc(dest, a, b) asm volatile("addc.cc.u32 %0, %1, %2;\n\t" : "=r"(dest) : "r"(a), "r"(b))

#define sub_cc(dest, a, b) asm volatile("sub.cc.u32 %0, %1, %2;\n\t" : "=r"(dest) : "r"(a), "r"(b))
#define subc_cc(dest, a, b) asm volatile("subc.cc.u32 %0, %1, %2;\n\t" : "=r"(dest) : "r"(a), "r"(b))
#define subc(dest, a, b) asm volatile("subc.u32 %0, %1, %2;\n\t" : "=r"(dest) : "r"(a), "r"(b))

#define set_eq(dest, a, b) asm volatile("set.eq.u32.u32 %0, %1, %2;\n\t" : "=r"(dest) : "r"(a), "r"(b))

#define lsbpos(x) (__ffs((x)))

__device__ __forceinline__ unsigned int endian(unsigned int x)
{
	return (x << 24) | ((x << 8) & 0x00ff0000) | ((x >> 8) & 0x0000ff00) | (x >> 24);
}

// SHA256 optimized PTX macros
#define sha_rotr(dest, x, n) asm volatile("shf.r.wrap.b32 %0, %1, %1, %2;\n\t" : "=r"(dest) : "r"(x), "r"(n))
#define sha_maj(dest, a, b, c) asm volatile("lop3.b32 %0, %1, %2, %3, 0xe8;\n\t" : "=r"(dest) : "r"(a), "r"(b), "r"(c))
#define sha_ch(dest, e, f, g) asm volatile("lop3.b32 %0, %1, %2, %3, 0xca;\n\t" : "=r"(dest) : "r"(e), "r"(f), "r"(g))
#define sha_xor(dest, a, b) asm volatile("xor.b32 %0, %1, %2;\n\t" : "=r"(dest) : "r"(a), "r"(b))
#define sha_add(dest, a, b) asm volatile("add.u32 %0, %1, %2;\n\t" : "=r"(dest) : "r"(a), "r"(b))
#define sha_shr(dest, x, n) asm volatile("shr.b32 %0, %1, %2;\n\t" : "=r"(dest) : "r"(x), "r"(n))

#endif
