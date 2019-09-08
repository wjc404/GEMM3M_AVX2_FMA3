# GEMM3M_AVX2_FMA3
cgemm3m and zgemm3m subroutines for large matrices, using AVX2 and FMA3 instructions, outperform MKL 2019 update 4.

Interface: FORTRAN, 32-bit integer.

Tuned parameters on i9-9900K (see Makefile):

    ZGEMM3M: BlkDimK=256, BlkDimN=128, B_PR_ELEM=64, A_PR_BYTE=256.
    
    CGEMM3M: BlkDimK=256, BlkDimN=256, B_PR_ELEM=64, A_PR_BYTE=256.
