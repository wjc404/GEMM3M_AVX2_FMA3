#define A0	%rdi //ablk pointer
#define B0	%rsi //bblk pointer
#define CL      %r14 //cload pointer
#define CS      %r15 //cstore pointer
#define LDC     %rcx //ldc * sizeof(float)
#define AL      %rax //aload pointer
#define CIP  -8(%rsp)//cstartpos
//BlkDimK=BlkDimN=256
//A_PR_BYTE=256
//B_PR_ELEM=96
//in gemm_driver.c: BlkDimM=192/SIZE/2
#ifdef DOUBLE
 #define VEC_BROAD vbroadcastsd
 #define VEC_FMA231 vfmadd231pd
 #define VEC_ADD vaddpd
 #define VEC_ADDSUB vaddsubpd
 #define MASKMOV vmaskmovpd
 #define SIZE 8
#else
 #define VEC_BROAD vbroadcastss
 #define VEC_FMA231 vfmadd231ps
 #define VEC_ADD vaddps
 #define VEC_ADDSUB vaddsubps
 #define MASKMOV vmaskmovps
 #define SIZE 4
#endif

#define NEXT_A_PREF_STEP (96*BlkDimK/BlkDimN) //in bytes

.macro UNPACKC1 src,dest
#ifdef DOUBLE
    vunpcklpd \src,\src,\dest
#else
    vpermilps $160,\src,\dest
#endif
.endm

.macro UNPACKC2 src,dest
#ifdef DOUBLE
    vunpckhpd \src,\src,\dest
#else
    vpermilps $245,\src,\dest
#endif
.endm

.macro UNPACKZRC1 src,dest
    vpxor \dest,\dest,\dest
#ifdef DOUBLE
    vunpcklpd \src,\dest,\dest
#else
    vblendps $170,\dest,\src,\dest
    vpermilps $177,\dest,\dest
#endif
.endm

.macro UNPACKZRC2 src,dest
    vpxor \dest,\dest,\dest
#ifdef DOUBLE
    vblendpd $10,\src,\dest,\dest
#else
    vblendps $170,\src,\dest,\dest
#endif
.endm

.macro KERNEL_1 Aoff,Boff
    vmovaps \Aoff(A0),%ymm1
    vmovaps \Aoff+32(A0),%ymm2
    vmovaps \Aoff+64(A0),%ymm3
    VEC_BROAD \Boff(B0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm4
    VEC_FMA231 %ymm0,%ymm2,%ymm5
    VEC_FMA231 %ymm0,%ymm3,%ymm6
    VEC_BROAD \Boff+SIZE(B0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm7
    VEC_FMA231 %ymm0,%ymm2,%ymm8
    VEC_FMA231 %ymm0,%ymm3,%ymm9
    VEC_BROAD \Boff+2*SIZE(B0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm10
    VEC_FMA231 %ymm0,%ymm2,%ymm11
    VEC_FMA231 %ymm0,%ymm3,%ymm12
    VEC_BROAD \Boff+3*SIZE(B0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm13
    VEC_FMA231 %ymm0,%ymm2,%ymm14
    VEC_FMA231 %ymm0,%ymm3,%ymm15
.endm

.macro KERNEL_f Aoff,Boff,delta,deltb
    vmovaps \Aoff(A0),%ymm1
    vmovaps \Aoff+32(A0),%ymm2
    vmovaps \Aoff+64(A0),%ymm3
    addq $\delta,A0
    VEC_BROAD \Boff(B0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm4
    VEC_FMA231 %ymm0,%ymm2,%ymm5
    VEC_FMA231 %ymm0,%ymm3,%ymm6
    VEC_BROAD \Boff+SIZE(B0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm7
    VEC_FMA231 %ymm0,%ymm2,%ymm8
    VEC_FMA231 %ymm0,%ymm3,%ymm9
    VEC_BROAD \Boff+2*SIZE(B0),%ymm0
    VEC_FMA231 %ymm0,%ymm1,%ymm10
    VEC_FMA231 %ymm0,%ymm2,%ymm11
    VEC_FMA231 %ymm0,%ymm3,%ymm12
    VEC_BROAD \Boff+3*SIZE(B0),%ymm0
    addq $\deltb,B0
    VEC_FMA231 %ymm0,%ymm1,%ymm13
    VEC_FMA231 %ymm0,%ymm2,%ymm14
    VEC_FMA231 %ymm0,%ymm3,%ymm15
.endm

.macro KERNEL_4 next_ablk_pref
    prefetcht0 A_PR_BYTE(A0)
    prefetcht0 A_PR_BYTE+64(A0)
    prefetcht1 (\next_ablk_pref)
    KERNEL_1 0,0
    prefetcht0 A_PR_BYTE+128(A0)
    prefetcht0 B_PR_ELEM*SIZE(B0)
    prefetcht1 64(\next_ablk_pref)
    incq %r11
    KERNEL_1 96,4*SIZE
    prefetcht0 A_PR_BYTE+192(A0)
    prefetcht0 A_PR_BYTE+256(A0)
    prefetcht1 128(\next_ablk_pref)
    KERNEL_1 192,8*SIZE
    prefetcht0 A_PR_BYTE+320(A0)
#ifdef DOUBLE
    prefetcht0 (B_PR_ELEM+8)*SIZE(B0)
#endif
    prefetcht1 192(\next_ablk_pref)
    addq $256,\next_ablk_pref
    KERNEL_f 288,12*SIZE,384,16*SIZE
.endm

.macro KERNEL_8
    prefetcht0 A_PR_BYTE(A0)
    prefetcht0 A_PR_BYTE+64(A0)
    KERNEL_1 0,0
    prefetcht0 B_PR_ELEM*SIZE(B0)
    prefetcht0 A_PR_BYTE+128(A0)
    KERNEL_1 96,4*SIZE
    prefetcht0 A_PR_BYTE+192(A0)
    prefetcht0 A_PR_BYTE+256(A0)
    KERNEL_1 192,8*SIZE
    prefetcht0 A_PR_BYTE+320(A0)
#ifdef DOUBLE
    prefetcht0 (B_PR_ELEM+8)*SIZE(B0)
#endif
    KERNEL_1 288,12*SIZE
    incq %r11
    prefetcht0 A_PR_BYTE+384(A0)
    prefetcht0 A_PR_BYTE+448(A0)
    KERNEL_1 384,16*SIZE
    prefetcht0 (B_PR_ELEM+16)*SIZE(B0)
    prefetcht0 A_PR_BYTE+512(A0)
    KERNEL_1 480,20*SIZE
    prefetcht0 A_PR_BYTE+576(A0)
    prefetcht0 A_PR_BYTE+640(A0)
    KERNEL_1 576,24*SIZE
    prefetcht0 A_PR_BYTE+704(A0)
#ifdef DOUBLE
    prefetcht0 (B_PR_ELEM+24)*SIZE(B0)
#endif
    KERNEL_f 672,28*SIZE,768,32*SIZE
.endm

.macro SHIFTYMM
    vmovaps %ymm7,%ymm4
    vmovaps %ymm8,%ymm5
    vmovaps %ymm9,%ymm6
    vmovaps %ymm10,%ymm7
    vmovaps %ymm11,%ymm8
    vmovaps %ymm12,%ymm9
    vmovaps %ymm13,%ymm10
    vmovaps %ymm14,%ymm11
    vmovaps %ymm15,%ymm12
.endm

.macro PREFC_1col src
    prefetcht0 (\src)
    prefetcht0 64(\src)
    prefetcht0 128(\src)
    prefetcht0 191(\src)
.endm

.macro PREFT1_C_1col src
    prefetcht1 (\src)
    prefetcht1 64(\src)
    prefetcht1 128(\src)
    prefetcht1 191(\src)
.endm

.macro CLEAR r1,r2,r3
    vpxor \r1,\r1,\r1
    vpxor \r2,\r2,\r2
    vpxor \r3,\r3,\r3
.endm

.macro UPDATECBLK_1col
    SHIFTYMM
    CLEAR %ymm13,%ymm14,%ymm15
.endm

//SP vector c1(low->high):-C1_1,-C2_1,-C1_2,-C2_2,-C1_3,-C2_3,-C1_4,-C2_4;
//SP output vector 1(low->high):C1_1,-C1_1,C1_2,-C1_2,C1_3,-C1_3,C1_4,-C1_4;
//SP output vector 2(low->high):C2_1,-C2_1,C2_2,-C2_2,C2_3,-C2_3,C2_4,-C2_4;
.macro STORECBLK_1vec_mArBr reg,mem
    UNPACKC1 \reg,%ymm0
    UNPACKC2 \reg,%ymm1
    vmovups \mem,%ymm2
    VEC_ADDSUB %ymm0,%ymm2,%ymm0
    vmovups %ymm0,\mem
    vmovups 32+\mem,%ymm3
    VEC_ADDSUB %ymm1,%ymm3,%ymm1
    vmovups %ymm1,32+\mem
.endm

.macro STORECBLK_1col_mArBr
    STORECBLK_1vec_mArBr %ymm4,(CS)
    STORECBLK_1vec_mArBr %ymm5,64(CS)
    STORECBLK_1vec_mArBr %ymm6,128(CS)
    addq LDC,CS
.endm

//SP vector c1(low->high):-C1_1,-C2_1,-C1_2,-C2_2,-C1_3,-C2_3,-C1_4,-C2_4;
//SP output vector 1(low->high):-C1_1,-C1_1,-C1_2,-C1_2,-C1_3,-C1_3,-C1_4,-C1_4;
//SP output vector 2(low->high):-C2_1,-C2_1,-C2_2,-C2_2,-C2_3,-C2_3,-C2_4,-C2_4;
.macro STORECBLK_1vec_mAiBi reg,mem
    UNPACKC1 \reg,%ymm0
    UNPACKC2 \reg,%ymm1
    VEC_ADD \mem,%ymm0,%ymm0
    vmovups %ymm0,\mem
    VEC_ADD 32+\mem,%ymm1,%ymm1
    vmovups %ymm1,32+\mem
.endm

.macro STORECBLK_1col_mAiBi
    STORECBLK_1vec_mAiBi %ymm4,(CS)
    STORECBLK_1vec_mAiBi %ymm5,64(CS)
    STORECBLK_1vec_mAiBi %ymm6,128(CS)
    addq LDC,CS
.endm

//SP vector c1(low->high):C1_1,C2_1,C1_2,C2_2,C1_3,C2_3,C1_4,C2_4;
//SP output vector 1(low->high):0,C1_1,0,C1_2,0,C1_3,0,C1_4;
//SP output vector 2(low->high):0,C2_1,0,C2_2,0,C2_3,0,C2_4;
.macro STORECBLK_1vec_AsBs reg,mem
    UNPACKZRC1 \reg,%ymm1
    UNPACKZRC2 \reg,%ymm2
    VEC_ADD \mem,%ymm1,%ymm1
    vmovups %ymm1,\mem
    VEC_ADD 32+\mem,%ymm2,%ymm2
    vmovups %ymm2,32+\mem
.endm

.macro STORECBLK_1col_AsBs
    STORECBLK_1vec_AsBs %ymm4,(CS)
    STORECBLK_1vec_AsBs %ymm5,64(CS)
    STORECBLK_1vec_AsBs %ymm6,128(CS)
    addq LDC,CS
.endm

.macro STORECBLK_1vec_mArBr_irregm reg,mem,mask
    UNPACKC1 \reg,%ymm0
    UNPACKC2 \reg,%ymm1
    vmovups \mask,%ymm3
    MASKMOV \mem,%ymm3,%ymm2
    VEC_ADDSUB %ymm0,%ymm2,%ymm2
    MASKMOV %ymm2,%ymm3,\mem
    vmovups 32+\mask,%ymm3
    MASKMOV 32+\mem,%ymm3,%ymm2
    VEC_ADDSUB %ymm1,%ymm2,%ymm2
    MASKMOV %ymm2,%ymm3,32+\mem
.endm

.macro STORECBLK_1col_mArBr_irregm mask
    STORECBLK_1vec_mArBr_irregm %ymm4,(CS),(\mask)
    STORECBLK_1vec_mArBr_irregm %ymm5,64(CS),64(\mask)
    STORECBLK_1vec_mArBr_irregm %ymm6,128(CS),128(\mask)
    addq LDC,CS
.endm

.macro STORECBLK_1vec_mAiBi_irregm reg,mem,mask
    UNPACKC1 \reg,%ymm0
    UNPACKC2 \reg,%ymm1
    vmovups \mask,%ymm3
    MASKMOV \mem,%ymm3,%ymm2
    VEC_ADD %ymm2,%ymm0,%ymm2
    MASKMOV %ymm2,%ymm3,\mem
    vmovups 32+\mask,%ymm3
    MASKMOV 32+\mem,%ymm3,%ymm2
    VEC_ADD %ymm2,%ymm1,%ymm1
    MASKMOV %ymm1,%ymm3,32+\mem
.endm

.macro STORECBLK_1col_mAiBi_irregm mask
    STORECBLK_1vec_mAiBi_irregm %ymm4,(CS),(\mask)
    STORECBLK_1vec_mAiBi_irregm %ymm5,64(CS),64(\mask)
    STORECBLK_1vec_mAiBi_irregm %ymm6,128(CS),128(\mask)
    addq LDC,CS
.endm

.macro STORECBLK_1vec_AsBs_irregm reg,mem,mask
    UNPACKZRC1 \reg,%ymm0
    UNPACKZRC2 \reg,%ymm1
    vmovups \mask,%ymm3
    MASKMOV \mem,%ymm3,%ymm2
    VEC_ADD %ymm2,%ymm0,%ymm2
    MASKMOV %ymm2,%ymm3,\mem
    vmovups 32+\mask,%ymm3
    MASKMOV 32+\mem,%ymm3,%ymm2
    VEC_ADD %ymm2,%ymm1,%ymm2
    MASKMOV %ymm2,%ymm3,32+\mem
.endm

.macro STORECBLK_1col_AsBs_irregm mask
    STORECBLK_1vec_AsBs_irregm %ymm4,(CS),(\mask)
    STORECBLK_1vec_AsBs_irregm %ymm5,64(CS),64(\mask)
    STORECBLK_1vec_AsBs_irregm %ymm6,128(CS),128(\mask)
    addq LDC,CS
.endm

.macro INIT_C_3col
    CLEAR %ymm7,%ymm8,%ymm9
    CLEAR %ymm10,%ymm11,%ymm12
    CLEAR %ymm13,%ymm14,%ymm15
.endm

.macro FIN_C_3col_mArBr
    STORECBLK_1vec_mArBr %ymm7,(CS)
    STORECBLK_1vec_mArBr %ymm8,64(CS)
    STORECBLK_1vec_mArBr %ymm9,128(CS)
    addq LDC,CS
    STORECBLK_1vec_mArBr %ymm10,(CS)
    STORECBLK_1vec_mArBr %ymm11,64(CS)
    STORECBLK_1vec_mArBr %ymm12,128(CS)
    addq LDC,CS
    STORECBLK_1vec_mArBr %ymm13,(CS)
    STORECBLK_1vec_mArBr %ymm14,64(CS)
    STORECBLK_1vec_mArBr %ymm15,128(CS)
.endm

.macro FIN_C_3col_mAiBi
    STORECBLK_1vec_mAiBi %ymm7,(CS)
    STORECBLK_1vec_mAiBi %ymm8,64(CS)
    STORECBLK_1vec_mAiBi %ymm9,128(CS)
    addq LDC,CS
    STORECBLK_1vec_mAiBi %ymm10,(CS)
    STORECBLK_1vec_mAiBi %ymm11,64(CS)
    STORECBLK_1vec_mAiBi %ymm12,128(CS)
    addq LDC,CS
    STORECBLK_1vec_mAiBi %ymm13,(CS)
    STORECBLK_1vec_mAiBi %ymm14,64(CS)
    STORECBLK_1vec_mAiBi %ymm15,128(CS)
.endm

.macro FIN_C_3col_AsBs
    STORECBLK_1vec_AsBs %ymm7,(CS)
    STORECBLK_1vec_AsBs %ymm8,64(CS)
    STORECBLK_1vec_AsBs %ymm9,128(CS)
    addq LDC,CS
    STORECBLK_1vec_AsBs %ymm10,(CS)
    STORECBLK_1vec_AsBs %ymm11,64(CS)
    STORECBLK_1vec_AsBs %ymm12,128(CS)
    addq LDC,CS
    STORECBLK_1vec_AsBs %ymm13,(CS)
    STORECBLK_1vec_AsBs %ymm14,64(CS)
    STORECBLK_1vec_AsBs %ymm15,128(CS)
.endm

.macro FIN_C_3col_mArBr_irregm mask
    STORECBLK_1vec_mArBr_irregm %ymm7,(CS),(\mask)
    STORECBLK_1vec_mArBr_irregm %ymm8,64(CS),64(\mask)
    STORECBLK_1vec_mArBr_irregm %ymm9,128(CS),128(\mask)
    addq LDC,CS
    STORECBLK_1vec_mArBr_irregm %ymm10,(CS),(\mask)
    STORECBLK_1vec_mArBr_irregm %ymm11,64(CS),64(\mask)
    STORECBLK_1vec_mArBr_irregm %ymm12,128(CS),128(\mask)
    addq LDC,CS
    STORECBLK_1vec_mArBr_irregm %ymm13,(CS),(\mask)
    STORECBLK_1vec_mArBr_irregm %ymm14,64(CS),64(\mask)
    STORECBLK_1vec_mArBr_irregm %ymm15,128(CS),128(\mask)
.endm

.macro FIN_C_3col_mAiBi_irregm mask
    STORECBLK_1vec_mAiBi_irregm %ymm7,(CS),(\mask)
    STORECBLK_1vec_mAiBi_irregm %ymm8,64(CS),64(\mask)
    STORECBLK_1vec_mAiBi_irregm %ymm9,128(CS),128(\mask)
    addq LDC,CS
    STORECBLK_1vec_mAiBi_irregm %ymm10,(CS),(\mask)
    STORECBLK_1vec_mAiBi_irregm %ymm11,64(CS),64(\mask)
    STORECBLK_1vec_mAiBi_irregm %ymm12,128(CS),128(\mask)
    addq LDC,CS
    STORECBLK_1vec_mAiBi_irregm %ymm13,(CS),(\mask)
    STORECBLK_1vec_mAiBi_irregm %ymm14,64(CS),64(\mask)
    STORECBLK_1vec_mAiBi_irregm %ymm15,128(CS),128(\mask)
.endm

.macro FIN_C_3col_AsBs_irregm mask
    STORECBLK_1vec_AsBs_irregm %ymm7,(CS),(\mask)
    STORECBLK_1vec_AsBs_irregm %ymm8,64(CS),64(\mask)
    STORECBLK_1vec_AsBs_irregm %ymm9,128(CS),128(\mask)
    addq LDC,CS
    STORECBLK_1vec_AsBs_irregm %ymm10,(CS),(\mask)
    STORECBLK_1vec_AsBs_irregm %ymm11,64(CS),64(\mask)
    STORECBLK_1vec_AsBs_irregm %ymm12,128(CS),128(\mask)
    addq LDC,CS
    STORECBLK_1vec_AsBs_irregm %ymm13,(CS),(\mask)
    STORECBLK_1vec_AsBs_irregm %ymm14,64(CS),64(\mask)
    STORECBLK_1vec_AsBs_irregm %ymm15,128(CS),128(\mask)
.endm

.macro SETMASKm//use stack to store mask integer array
#ifdef DOUBLE
    xorq %rax,%rax
    subq %r8,%rax
    addq $11,%rax
    movq %rax,-8(%rsp)
    movq %rax,-16(%rsp)
    decq %rax
    movq %rax,-24(%rsp)
    movq %rax,-32(%rsp)
    decq %rax
    movq %rax,-40(%rsp)
    movq %rax,-48(%rsp)
    decq %rax
    movq %rax,-56(%rsp)
    movq %rax,-64(%rsp)
    decq %rax
    movq %rax,-72(%rsp)
    movq %rax,-80(%rsp)
    decq %rax
    movq %rax,-88(%rsp)
    movq %rax,-96(%rsp)
    decq %rax
    movq %rax,-104(%rsp)
    movq %rax,-112(%rsp)
    decq %rax
    movq %rax,-120(%rsp)
    movq %rax,-128(%rsp)
    decq %rax
    movq %rax,-136(%rsp)
    movq %rax,-144(%rsp)
    decq %rax
    movq %rax,-152(%rsp)
    movq %rax,-160(%rsp)
    decq %rax
    movq %rax,-168(%rsp)
    movq %rax,-176(%rsp)
    decq %rax
    movq %rax,-184(%rsp)
    movq %rax,-192(%rsp)
    leaq -192(%rsp),%rax
#else
    xorl %eax,%eax
    subl %r8d,%eax
    addl $23,%eax
    movl %eax,-4(%rsp)
    movl %eax,-8(%rsp)
    decl %eax
    movl %eax,-12(%rsp)
    movl %eax,-16(%rsp)
    decl %eax
    movl %eax,-20(%rsp)
    movl %eax,-24(%rsp)
    decl %eax
    movl %eax,-28(%rsp)
    movl %eax,-32(%rsp)
    decl %eax
    movl %eax,-36(%rsp)
    movl %eax,-40(%rsp)
    decl %eax
    movl %eax,-44(%rsp)
    movl %eax,-48(%rsp)
    decl %eax
    movl %eax,-52(%rsp)
    movl %eax,-56(%rsp)
    decl %eax
    movl %eax,-60(%rsp)
    movl %eax,-64(%rsp)
    decl %eax
    movl %eax,-68(%rsp)
    movl %eax,-72(%rsp)
    decl %eax
    movl %eax,-76(%rsp)
    movl %eax,-80(%rsp)
    decl %eax
    movl %eax,-84(%rsp)
    movl %eax,-88(%rsp)
    decl %eax
    movl %eax,-92(%rsp)
    movl %eax,-96(%rsp)
    decl %eax
    movl %eax,-100(%rsp)
    movl %eax,-104(%rsp)
    decl %eax
    movl %eax,-108(%rsp)
    movl %eax,-112(%rsp)
    decl %eax
    movl %eax,-116(%rsp)
    movl %eax,-120(%rsp)
    decl %eax
    movl %eax,-124(%rsp)
    movl %eax,-128(%rsp)
    decl %eax
    movl %eax,-132(%rsp)
    movl %eax,-136(%rsp)
    decl %eax
    movl %eax,-140(%rsp)
    movl %eax,-144(%rsp)
    decl %eax
    movl %eax,-148(%rsp)
    movl %eax,-152(%rsp)
    decl %eax
    movl %eax,-156(%rsp)
    movl %eax,-160(%rsp)
    decl %eax
    movl %eax,-164(%rsp)
    movl %eax,-168(%rsp)
    decl %eax
    movl %eax,-172(%rsp)
    movl %eax,-176(%rsp)
    decl %eax
    movl %eax,-180(%rsp)
    movl %eax,-184(%rsp)
    decl %eax
    movl %eax,-188(%rsp)
    movl %eax,-192(%rsp)
    leaq -192(%rsp),%rax
#endif
.endm

.macro SET_LDC
#ifdef DOUBLE
    salq $4,LDC
#else
    salq $3,LDC
#endif
.endm

.macro PREF_ABLK_HEAD
    prefetcht0 (A0)
# if A_PR_BYTE > 64
    prefetcht0 64(A0)
# endif
# if A_PR_BYTE > 128
    prefetcht0 128(A0)
# endif
# if A_PR_BYTE > 192
    prefetcht0 192(A0)
# endif
# if A_PR_BYTE > 256
    prefetcht0 256(A0)
# endif
# if A_PR_BYTE > 320
    prefetcht0 320(A0)
# endif
.endm

.macro PREF_NEXT_ABLK_SEGMENT pointer
    prefetcht1 (\pointer)
# if NEXT_A_PREF_STEP > 64
    prefetcht1 64(\pointer)
# endif
# if NEXT_A_PREF_STEP > 128
    prefetcht1 128(\pointer)
# endif
# if NEXT_A_PREF_STEP > 192
    prefetcht1 192(\pointer)
# endif
.endm

.section .text
//enter the function gemmblkregccc, rdi=abufferctpos, rsi=bblk, rdx=cstartpos, ecx=ldc
.globl gemmblkregccc
.type gemmblkregccc,@function
gemmblkregccc:
//parameter setup
    push %r15
    push %r14
    push %r12
    movq %rdx,CIP
    movslq %ecx,LDC
    SET_LDC

/*first do -A(r)B(r)*/
    movq A0,AL //A0=%rdi
    addq $192*BlkDimK,AL //point to (prefetch) next ablk zone of abuffer, start from the tail part
    movq CIP,CS

    INIT_C_3col
    movq $(65536-96*BlkDimK)*281474976710656,%r10 //64-bit offset vector(4 elements): (65536-SIZE_OF_BUFFER_A)*281474976710656
    xorq %r12,%r12
.Louter_gemmblkregccc_mArBr:
    UPDATECBLK_1col
    movswq %r10w,%r9
    PREFC_1col CS
    prefetcht2 255(CS)
    subq $NEXT_A_PREF_STEP,AL
    PREF_NEXT_ABLK_SEGMENT AL
    xorq %r11,%r11
.Linner_gemmblkregccc_mArBr:
    KERNEL_8
    cmpq $BlkDimK/32,%r11
    jb .Linner_gemmblkregccc_mArBr

    addq %r9,A0
    PREF_ABLK_HEAD
    incq %r12
    STORECBLK_1col_mArBr
    rorq $16,%r10
    cmpq $BlkDimN-4,%r12
    jb .Louter_gemmblkregccc_mArBr

    movq AL,%r9
    subq $NEXT_A_PREF_STEP*4,%r9
    movq CIP,CL
.Louter_gemmblkregccc_mArBr_last:
    UPDATECBLK_1col
    PREFC_1col CS
    prefetcht2 255(CS)
    PREFT1_C_1col CL
    addq LDC,CL
    xorq %r11,%r11
.Linner_gemmblkregccc_mArBr_last:
    KERNEL_4 %r9
    cmpq $BlkDimK/16,%r11
    jb .Linner_gemmblkregccc_mArBr_last

    incq %r12
    STORECBLK_1col_mArBr
    cmpq $BlkDimN,%r12
    jb .Louter_gemmblkregccc_mArBr_last

    movq CIP,CS
    FIN_C_3col_mArBr

/*then do -A(i)B(i)*/
    movq A0,AL
    addq $192*BlkDimK,AL //point to (prefetch) next ablk zone of abuffer, start from the tail part
    movq CIP,CS

    INIT_C_3col
    xorq %r12,%r12
.Louter_gemmblkregccc_mAiBi:
    UPDATECBLK_1col
    movswq %r10w,%r9
    PREFC_1col CS
    prefetcht2 319(CS)
    subq $NEXT_A_PREF_STEP,AL
    PREF_NEXT_ABLK_SEGMENT AL
    xorq %r11,%r11
.Linner_gemmblkregccc_mAiBi:
    KERNEL_8
    cmpq $BlkDimK/32,%r11
    jb .Linner_gemmblkregccc_mAiBi

    addq %r9,A0
    PREF_ABLK_HEAD
    incq %r12
    STORECBLK_1col_mAiBi
    rorq $16,%r10
    cmpq $BlkDimN-4,%r12
    jb .Louter_gemmblkregccc_mAiBi

    movq AL,%r9
    subq $NEXT_A_PREF_STEP*4,%r9
    movq CIP,CL
.Louter_gemmblkregccc_mAiBi_last:
    UPDATECBLK_1col
    PREFC_1col CS
    prefetcht2 319(CS)
    PREFT1_C_1col CL
    addq LDC,CL
    xorq %r11,%r11
.Linner_gemmblkregccc_mAiBi_last:
    KERNEL_4 %r9
    cmpq $BlkDimK/16,%r11
    jb .Linner_gemmblkregccc_mAiBi_last

    incq %r12
    STORECBLK_1col_mAiBi
    cmpq $BlkDimN,%r12
    jb .Louter_gemmblkregccc_mAiBi_last

    movq CIP,CS
    FIN_C_3col_mAiBi

/*finally do A(i+r)B(i+r)*/
    movq A0,AL
    addq $192*BlkDimK,AL //point to (prefetch) next ablk zone of abuffer, start from the tail part
    movq CIP,CS

    INIT_C_3col
    xorq %r12,%r12
.Louter_gemmblkregccc_AsBs:
    UPDATECBLK_1col
    movswq %r10w,%r9
    PREFC_1col CS
    prefetcht2 383(CS)
    subq $NEXT_A_PREF_STEP,AL
    PREF_NEXT_ABLK_SEGMENT AL
    xorq %r11,%r11
.Linner_gemmblkregccc_AsBs:
    KERNEL_8
    cmpq $BlkDimK/32,%r11
    jb .Linner_gemmblkregccc_AsBs

    addq %r9,A0
    PREF_ABLK_HEAD
    incq %r12
    STORECBLK_1col_AsBs
    rorq $16,%r10
    cmpq $BlkDimN-4,%r12
    jb .Louter_gemmblkregccc_AsBs

    movq AL,%r9
    subq $NEXT_A_PREF_STEP*4,%r9
    movq CIP,CL
.Louter_gemmblkregccc_AsBs_last:
    UPDATECBLK_1col
    PREFC_1col CS
    prefetcht2 383(CS)
    PREFT1_C_1col CL
    addq LDC,CL
    xorq %r11,%r11
.Linner_gemmblkregccc_AsBs_last:
    KERNEL_4 %r9
    cmpq $BlkDimK/16,%r11
    jb .Linner_gemmblkregccc_AsBs_last

    incq %r12
    STORECBLK_1col_AsBs
    cmpq $BlkDimN,%r12
    jb .Louter_gemmblkregccc_AsBs_last

    movq CIP,CS
    FIN_C_3col_AsBs

/*clean up and return*/
    vzeroupper
    pop %r12
    pop %r14
    pop %r15
    retq

//enter the function gemmblktailccc, rdi=ablk, rsi=bblk, rdx=cstartpos, ecx=ldc, r8d=mdim
.globl gemmblktailccc
.type gemmblktailccc,@function
gemmblktailccc:

/*parameter setup*/
    push %r15
    push %r14
    push %r12
    push %rdx //cstartpos
    movslq %ecx,LDC
    SET_LDC
    movslq %r8d,%r8 //mdim
    SETMASKm //generate mask integers. Now %rax point to the base element of mask integers.
    add $8,%rsp //recover rsp so "CIP" can work normally

/*first do -A(r)B(r)*/
    movq $(96*BlkDimK)*281474976710656,%r10
    movq CIP,CS
    INIT_C_3col
    xorq %r12,%r12
.Louter_tail_mArBr:
    UPDATECBLK_1col
    PREFC_1col CS
    xorq %r11,%r11
.Linner_tail_mArBr:
    KERNEL_8
    cmpq $BlkDimK/32,%r11
    jb .Linner_tail_mArBr

    STORECBLK_1col_mArBr_irregm %rax
    incq %r12
    movswq %r10w,%r9
    subq %r9,A0
    ror $16,%r10
    cmpq $BlkDimN,%r12
    jb .Louter_tail_mArBr

    movq CIP,CS
    FIN_C_3col_mArBr_irregm %rax

/*then do -A(i)B(i)*/
    addq $96*BlkDimK,A0
    movq CIP,CS
    INIT_C_3col
    xorq %r12,%r12
.Louter_tail_mAiBi:
    UPDATECBLK_1col
    PREFC_1col CS
    xorq %r11,%r11
.Linner_tail_mAiBi:
    KERNEL_8
    cmpq $BlkDimK/32,%r11
    jb .Linner_tail_mAiBi

    STORECBLK_1col_mAiBi_irregm %rax
    incq %r12
    movswq %r10w,%r9
    subq %r9,A0
    ror $16,%r10
    cmpq $BlkDimN,%r12
    jb .Louter_tail_mAiBi

    movq CIP,CS
    FIN_C_3col_mAiBi_irregm %rax

/*finally do A(i+r)B(i+r)*/
    addq $96*BlkDimK,A0
    movq CIP,CS
    INIT_C_3col
    xorq %r12,%r12
.Louter_tail_AsBs:
    UPDATECBLK_1col
    PREFC_1col CS
    xorq %r11,%r11
.Linner_tail_AsBs:
    KERNEL_8
    cmpq $BlkDimK/32,%r11
    jb .Linner_tail_AsBs

    STORECBLK_1col_AsBs_irregm %rax
    incq %r12
    movswq %r10w,%r9
    subq %r9,A0
    ror $16,%r10
    cmpq $BlkDimN,%r12
    jb .Louter_tail_AsBs

    movq CIP,CS
    FIN_C_3col_AsBs_irregm %rax

/*clean up and return*/
    vzeroupper
    pop %r12
    pop %r14
    pop %r15
    retq

//enter the function timedelay
.globl timedelay
.type timedelay,@function
timedelay:
    xorq %r11,%r11
.Ltimedelay:
    incq %r11
    vhaddpd %ymm0,%ymm0,%ymm0
    cmpq $2000,%r11
    jb .Ltimedelay

    vzeroupper
    retq
