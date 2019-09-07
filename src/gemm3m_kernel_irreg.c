#ifdef DOUBLE
 #define IRREG_SIZE 8
 #define IRREG_VEC_TYPE __m256d
 #define IRREG_VEC_ZERO _mm256_setzero_pd
 #define IRREG_VEC_LOADA _mm256_load_pd
 #define IRREG_VEC_LOADU _mm256_loadu_pd
 #define IRREG_VEC_MASKLOAD _mm256_maskload_pd
 #define IRREG_VEC_STOREU _mm256_storeu_pd
 #define IRREG_VEC_MASKSTORE _mm256_maskstore_pd
 #define IRREG_VEC_BROAD _mm256_broadcast_sd
 #define IRREG_VEC_FMADD _mm256_fmadd_pd
 #define IRREG_VEC_ADD _mm256_add_pd
 #define IRREG_VEC_ADDSUB _mm256_addsub_pd
 #define IRREG_VEC_UNPACKC1(y1) _mm256_unpacklo_pd(y1,y1)
 #define IRREG_VEC_UNPACKC2(y1) _mm256_unpackhi_pd(y1,y1)
 #define IRREG_VEC_UNPACKZRC1(y1) _mm256_unpacklo_pd(_mm256_setzero_pd(),y1)
 #define IRREG_VEC_UNPACKZRC2(y1) _mm256_blend_pd(_mm256_setzero_pd(),y1,10)
#else
 #define IRREG_SIZE 4
 #define IRREG_VEC_TYPE __m256
 #define IRREG_VEC_ZERO _mm256_setzero_ps
 #define IRREG_VEC_LOADA _mm256_load_ps
 #define IRREG_VEC_LOADU _mm256_loadu_ps
 #define IRREG_VEC_MASKLOAD _mm256_maskload_ps
 #define IRREG_VEC_STOREU _mm256_storeu_ps
 #define IRREG_VEC_MASKSTORE _mm256_maskstore_ps
 #define IRREG_VEC_BROAD _mm256_broadcast_ss
 #define IRREG_VEC_FMADD _mm256_fmadd_ps
 #define IRREG_VEC_ADD _mm256_add_ps
 #define IRREG_VEC_ADDSUB _mm256_addsub_ps
 #define IRREG_VEC_UNPACKC1(y1) _mm256_permute_ps(y1,160)
 #define IRREG_VEC_UNPACKC2(y1) _mm256_permute_ps(y1,245)
 #define IRREG_VEC_UNPACKZRC1(y1) _mm256_blend_ps(_mm256_setzero_ps(),_mm256_permute_ps(y1,177),170)
 #define IRREG_VEC_UNPACKZRC2(y1) _mm256_blend_ps(_mm256_setzero_ps(),y1,170)
#endif
#define INIT_1col(pref) {\
   c1=IRREG_VEC_ZERO();_mm_prefetch((char *)ctemp,_MM_HINT_T0);\
   c2=IRREG_VEC_ZERO();_mm_prefetch((char *)(ctemp+64/IRREG_SIZE),_MM_HINT_T0);\
   c3=IRREG_VEC_ZERO();_mm_prefetch((char *)(ctemp+128/IRREG_SIZE),_MM_HINT_T0);\
   _mm_prefetch((char *)(ctemp+192/IRREG_SIZE-1),_MM_HINT_T0);\
   _mm_prefetch((char *)(ctemp+(192+pref)/IRREG_SIZE-1),_MM_HINT_T2);\
}//pref=64 for mArBr;128 for mAiBi;192 for AsBs.
#define INIT_4col(pref) {\
   c1=IRREG_VEC_ZERO();_mm_prefetch((char *)cpref,_MM_HINT_T0);\
   c2=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+64/IRREG_SIZE),_MM_HINT_T0);\
   c3=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+128/IRREG_SIZE),_MM_HINT_T0);\
   _mm_prefetch((char *)(cpref+192/IRREG_SIZE-1),_MM_HINT_T0);\
   _mm_prefetch((char *)(cpref+(192+pref)/IRREG_SIZE-1),_MM_HINT_T2);cpref+=ldc*2;\
   c4=IRREG_VEC_ZERO();_mm_prefetch((char *)cpref,_MM_HINT_T0);\
   c5=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+64/IRREG_SIZE),_MM_HINT_T0);\
   c6=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+128/IRREG_SIZE),_MM_HINT_T0);\
   _mm_prefetch((char *)(cpref+192/IRREG_SIZE-1),_MM_HINT_T0);\
   _mm_prefetch((char *)(cpref+(192+pref)/IRREG_SIZE-1),_MM_HINT_T2);cpref+=ldc*2;\
   c7=IRREG_VEC_ZERO();_mm_prefetch((char *)cpref,_MM_HINT_T0);\
   c8=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+64/IRREG_SIZE),_MM_HINT_T0);\
   c9=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+128/IRREG_SIZE),_MM_HINT_T0);\
   _mm_prefetch((char *)(cpref+192/IRREG_SIZE-1),_MM_HINT_T0);\
   _mm_prefetch((char *)(cpref+(192+pref)/IRREG_SIZE-1),_MM_HINT_T2);cpref+=ldc*2;\
   c10=IRREG_VEC_ZERO();_mm_prefetch((char *)cpref,_MM_HINT_T0);\
   c11=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+64/IRREG_SIZE),_MM_HINT_T0);\
   c12=IRREG_VEC_ZERO();_mm_prefetch((char *)(cpref+128/IRREG_SIZE),_MM_HINT_T0);\
   _mm_prefetch((char *)(cpref+192/IRREG_SIZE-1),_MM_HINT_T0);\
   _mm_prefetch((char *)(cpref+(192+pref)/IRREG_SIZE-1),_MM_HINT_T2);\
}//pref=64 for mArBr;128 for mAiBi;192 for AsBs.
#define KERNELkr {\
   a1=IRREG_VEC_LOADA(atemp);atemp+=32/IRREG_SIZE;\
   a2=IRREG_VEC_LOADA(atemp);atemp+=32/IRREG_SIZE;\
   a3=IRREG_VEC_LOADA(atemp);atemp+=32/IRREG_SIZE;\
   b1=IRREG_VEC_BROAD(btemp);btemp++;\
   c1=IRREG_VEC_FMADD(a1,b1,c1);c2=IRREG_VEC_FMADD(a2,b1,c2);c3=IRREG_VEC_FMADD(a3,b1,c3);\
}
#define KERNELk1 {\
   KERNELkr\
   b1=IRREG_VEC_BROAD(btemp);btemp++;\
   c4=IRREG_VEC_FMADD(a1,b1,c4);c5=IRREG_VEC_FMADD(a2,b1,c5);c6=IRREG_VEC_FMADD(a3,b1,c6);\
   b1=IRREG_VEC_BROAD(btemp);btemp++;\
   c7=IRREG_VEC_FMADD(a1,b1,c7);c8=IRREG_VEC_FMADD(a2,b1,c8);c9=IRREG_VEC_FMADD(a3,b1,c9);\
   b1=IRREG_VEC_BROAD(btemp);btemp++;\
   c10=IRREG_VEC_FMADD(a1,b1,c10);c11=IRREG_VEC_FMADD(a2,b1,c11);c12=IRREG_VEC_FMADD(a3,b1,c12);\
}
#define KERNELk2 {\
   _mm_prefetch((char *)(atemp+A_PR_BYTE/IRREG_SIZE),_MM_HINT_T0);\
   _mm_prefetch((char *)(btemp+B_PR_ELEM),_MM_HINT_T0);\
   KERNELk1\
   _mm_prefetch((char *)(atemp+(A_PR_BYTE-32)/IRREG_SIZE),_MM_HINT_T0);\
   _mm_prefetch((char *)(atemp+(A_PR_BYTE+32)/IRREG_SIZE),_MM_HINT_T0);\
   KERNELk1\
}
#define STORE_C_1vec_mArBr(mem,c1) {\
   a1=IRREG_VEC_UNPACKC1(c1);\
   a2=IRREG_VEC_UNPACKC2(c1);\
   a3=IRREG_VEC_LOADU(mem);\
   a3=IRREG_VEC_ADDSUB(a3,a1);\
   IRREG_VEC_STOREU(mem,a3);\
   a3=IRREG_VEC_LOADU(mem+32/IRREG_SIZE);\
   a3=IRREG_VEC_ADDSUB(a3,a2);\
   IRREG_VEC_STOREU(mem+32/IRREG_SIZE,a3);\
}
#define STORE_C_1col_mArBr(c1,c2,c3) {\
   STORE_C_1vec_mArBr(ctemp,c1);\
   STORE_C_1vec_mArBr(ctemp+64/IRREG_SIZE,c2);\
   STORE_C_1vec_mArBr(ctemp+128/IRREG_SIZE,c3);\
   ctemp+=ldc*2;\
}
#define STORE_C_1vec_mAiBi(mem,c1) {\
   a1=IRREG_VEC_UNPACKC1(c1);\
   a2=IRREG_VEC_UNPACKC2(c1);\
   a3=IRREG_VEC_LOADU(mem);\
   a3=IRREG_VEC_ADD(a3,a1);\
   IRREG_VEC_STOREU(mem,a3);\
   a3=IRREG_VEC_LOADU(mem+32/IRREG_SIZE);\
   a3=IRREG_VEC_ADD(a3,a2);\
   IRREG_VEC_STOREU(mem+32/IRREG_SIZE,a3);\
}
#define STORE_C_1col_mAiBi(c1,c2,c3) {\
   STORE_C_1vec_mAiBi(ctemp,c1);\
   STORE_C_1vec_mAiBi(ctemp+64/IRREG_SIZE,c2);\
   STORE_C_1vec_mAiBi(ctemp+128/IRREG_SIZE,c3);\
   ctemp+=ldc*2;\
}
#define STORE_C_1vec_AsBs(mem,c1) {\
   a1=IRREG_VEC_UNPACKZRC1(c1);\
   a2=IRREG_VEC_UNPACKZRC2(c1);\
   a3=IRREG_VEC_LOADU(mem);\
   a3=IRREG_VEC_ADD(a3,a1);\
   IRREG_VEC_STOREU(mem,a3);\
   a3=IRREG_VEC_LOADU(mem+32/IRREG_SIZE);\
   a3=IRREG_VEC_ADD(a3,a2);\
   IRREG_VEC_STOREU(mem+32/IRREG_SIZE,a3);\
}
#define STORE_C_1col_AsBs(c1,c2,c3) {\
   STORE_C_1vec_AsBs(ctemp,c1);\
   STORE_C_1vec_AsBs(ctemp+64/IRREG_SIZE,c2);\
   STORE_C_1vec_AsBs(ctemp+128/IRREG_SIZE,c3);\
   ctemp+=ldc*2;\
}
#define STOREIRREGM_C_1vec_mArBr(mem,c1,ml1,ml2) {\
   a1=IRREG_VEC_UNPACKC1(c1);\
   a2=IRREG_VEC_UNPACKC2(c1);\
   a3=IRREG_VEC_MASKLOAD(mem,ml1);\
   a3=IRREG_VEC_ADDSUB(a3,a1);\
   IRREG_VEC_MASKSTORE(mem,ml1,a3);\
   a3=IRREG_VEC_MASKLOAD(mem+32/IRREG_SIZE,ml2);\
   a3=IRREG_VEC_ADDSUB(a3,a2);\
   IRREG_VEC_MASKSTORE(mem+32/IRREG_SIZE,ml2,a3);\
}
#define STOREIRREGM_C_1col_mArBr(c1,c2,c3) {\
   STOREIRREGM_C_1vec_mArBr(ctemp,c1,ml1,ml2);\
   STOREIRREGM_C_1vec_mArBr(ctemp+64/IRREG_SIZE,c2,ml3,ml4);\
   STOREIRREGM_C_1vec_mArBr(ctemp+128/IRREG_SIZE,c3,ml5,ml6);\
   ctemp+=ldc*2;\
}
#define STOREIRREGM_C_1vec_mAiBi(mem,c1,ml1,ml2) {\
   a1=IRREG_VEC_UNPACKC1(c1);\
   a2=IRREG_VEC_UNPACKC2(c1);\
   a3=IRREG_VEC_MASKLOAD(mem,ml1);\
   a3=IRREG_VEC_ADD(a3,a1);\
   IRREG_VEC_MASKSTORE(mem,ml1,a3);\
   a3=IRREG_VEC_MASKLOAD(mem+32/IRREG_SIZE,ml2);\
   a3=IRREG_VEC_ADD(a3,a2);\
   IRREG_VEC_MASKSTORE(mem+32/IRREG_SIZE,ml2,a3);\
}
#define STOREIRREGM_C_1col_mAiBi(c1,c2,c3) {\
   STOREIRREGM_C_1vec_mAiBi(ctemp,c1,ml1,ml2);\
   STOREIRREGM_C_1vec_mAiBi(ctemp+64/IRREG_SIZE,c2,ml3,ml4);\
   STOREIRREGM_C_1vec_mAiBi(ctemp+128/IRREG_SIZE,c3,ml5,ml6);\
   ctemp+=ldc*2;\
}
#define STOREIRREGM_C_1vec_AsBs(mem,c1,ml1,ml2) {\
   a1=IRREG_VEC_UNPACKZRC1(c1);\
   a2=IRREG_VEC_UNPACKZRC2(c1);\
   a3=IRREG_VEC_MASKLOAD(mem,ml1);\
   a3=IRREG_VEC_ADD(a3,a1);\
   IRREG_VEC_MASKSTORE(mem,ml1,a3);\
   a3=IRREG_VEC_MASKLOAD(mem+32/IRREG_SIZE,ml2);\
   a3=IRREG_VEC_ADD(a3,a2);\
   IRREG_VEC_MASKSTORE(mem+32/IRREG_SIZE,ml2,a3);\
}
#define STOREIRREGM_C_1col_AsBs(c1,c2,c3) {\
   STOREIRREGM_C_1vec_AsBs(ctemp,c1,ml1,ml2);\
   STOREIRREGM_C_1vec_AsBs(ctemp+64/IRREG_SIZE,c2,ml3,ml4);\
   STOREIRREGM_C_1vec_AsBs(ctemp+128/IRREG_SIZE,c3,ml5,ml6);\
   ctemp+=ldc*2;\
}
static void gemmblkirregkccc(FLOAT * __restrict__ ablk,FLOAT * __restrict__ bblk,FLOAT * __restrict__ cstartpos,int ldc,int kdim){
  register IRREG_VEC_TYPE a1,a2,a3,b1,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12;FLOAT *atemp,*btemp,*ctemp,*cpref,*ahead;int ccol,acol;
  btemp=bblk;
/*first do mArBr*/
  ctemp=cstartpos;
  for(ccol=0;ccol<BlkDimN;ccol+=4){//loop over cblk-columns, calculate 4 columns of cblk in each iteration.
   cpref=ctemp;
   INIT_4col(64)
   atemp=ablk;
   for(acol=0;acol<kdim;acol++) KERNELk1//loop over ablk-columns, load 1 column of ablk in each micro-iteration.
   STORE_C_1col_mArBr(c1,c2,c3)
   STORE_C_1col_mArBr(c4,c5,c6)
   STORE_C_1col_mArBr(c7,c8,c9)
   STORE_C_1col_mArBr(c10,c11,c12)
  }
/*then do mAiBi*/
  ctemp=cstartpos;ahead=ablk+kdim*BlkDimM;
  for(ccol=0;ccol<BlkDimN;ccol+=4){
   cpref=ctemp;
   INIT_4col(128)
   atemp=ahead;
   for(acol=0;acol<kdim;acol++) KERNELk1
   STORE_C_1col_mAiBi(c1,c2,c3)
   STORE_C_1col_mAiBi(c4,c5,c6)
   STORE_C_1col_mAiBi(c7,c8,c9)
   STORE_C_1col_mAiBi(c10,c11,c12)
  }
/*finally do mAiBi*/
  ctemp=cstartpos;ahead=ablk+kdim*BlkDimM*2;
  for(ccol=0;ccol<BlkDimN;ccol+=4){
   cpref=ctemp;
   INIT_4col(192)
   atemp=ahead;
   for(acol=0;acol<kdim;acol++) KERNELk1
   STORE_C_1col_AsBs(c1,c2,c3)
   STORE_C_1col_AsBs(c4,c5,c6)
   STORE_C_1col_AsBs(c7,c8,c9)
   STORE_C_1col_AsBs(c10,c11,c12)
  }
}
static void gemmblkirregnccc(FLOAT * __restrict__ ablk,FLOAT * __restrict__ bblk,FLOAT * __restrict__ cstartpos,int ldc,int ndim){
  register IRREG_VEC_TYPE a1,a2,a3,b1,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12;
  FLOAT *atemp,*btemp,*ctemp,*cpref,*ahead;int ccol,acol;
  btemp=bblk;
/*first do mArBr*/
  ctemp=cstartpos;
  for(ccol=0;ccol<ndim-3;ccol+=4){//loop over cblk-columns, calculate 4 columns of cblk in each iteration.
   cpref=ctemp;
   INIT_4col(64)
   atemp=ablk;
   for(acol=0;acol<BlkDimK;acol+=8){//loop over ablk-columns, load 1 column of ablk in each micro-iteration.
    KERNELk2
    KERNELk2
    KERNELk2
    KERNELk2
   }
   STORE_C_1col_mArBr(c1,c2,c3)
   STORE_C_1col_mArBr(c4,c5,c6)
   STORE_C_1col_mArBr(c7,c8,c9)
   STORE_C_1col_mArBr(c10,c11,c12)
  }
  for(;ccol<ndim;ccol++){
   INIT_1col(64)
   atemp=ablk;
   for(acol=0;acol<BlkDimK;acol++) KERNELkr//loop over ablk-columns, load 1 column of ablk in each micro-iteration.
   STORE_C_1col_mArBr(c1,c2,c3)
  }
/*then do mAiBi*/
  ctemp=cstartpos;ahead=ablk+BlkDimM*BlkDimK;
  for(ccol=0;ccol<ndim-3;ccol+=4){
   cpref=ctemp;
   INIT_4col(128)
   atemp=ahead;
   for(acol=0;acol<BlkDimK;acol+=8){
    KERNELk2
    KERNELk2
    KERNELk2
    KERNELk2
   }
   STORE_C_1col_mAiBi(c1,c2,c3)
   STORE_C_1col_mAiBi(c4,c5,c6)
   STORE_C_1col_mAiBi(c7,c8,c9)
   STORE_C_1col_mAiBi(c10,c11,c12)
  }
  for(;ccol<ndim;ccol++){
   INIT_1col(128)
   atemp=ahead;
   for(acol=0;acol<BlkDimK;acol++) KERNELkr
   STORE_C_1col_mAiBi(c1,c2,c3)
  }
/*finally do AsBs*/
  ctemp=cstartpos;ahead=ablk+BlkDimM*BlkDimK*2;
  for(ccol=0;ccol<ndim-3;ccol+=4){
   cpref=ctemp;
   INIT_4col(192)
   atemp=ahead;
   for(acol=0;acol<BlkDimK;acol+=8){
    KERNELk2
    KERNELk2
    KERNELk2
    KERNELk2
   }
   STORE_C_1col_AsBs(c1,c2,c3)
   STORE_C_1col_AsBs(c4,c5,c6)
   STORE_C_1col_AsBs(c7,c8,c9)
   STORE_C_1col_AsBs(c10,c11,c12)
  }
  for(;ccol<ndim;ccol++){
   INIT_1col(192)
   atemp=ahead;
   for(acol=0;acol<BlkDimK;acol++) KERNELkr
   STORE_C_1col_AsBs(c1,c2,c3)
  }
}
static void gemmblkirregccc(FLOAT * __restrict__ ablk,FLOAT * __restrict__ bblk,FLOAT * __restrict__ cstartpos,int ldc,int mdim,int ndim,int kdim){
  register IRREG_VEC_TYPE a1,a2,a3,b1,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12;
  __m256i ml1,ml2,ml3,ml4,ml5,ml6;
  FLOAT *atemp,*btemp,*ctemp,*cpref,*ahead;int ccol,acol;
#ifdef DOUBLE
  ml1=_mm256_setr_epi32(0,-(mdim>0),0,-(mdim>0),0,-(mdim>1),0,-(mdim>1));
  ml2=_mm256_setr_epi32(0,-(mdim>2),0,-(mdim>2),0,-(mdim>3),0,-(mdim>3));
  ml3=_mm256_setr_epi32(0,-(mdim>4),0,-(mdim>4),0,-(mdim>5),0,-(mdim>5));
  ml4=_mm256_setr_epi32(0,-(mdim>6),0,-(mdim>6),0,-(mdim>7),0,-(mdim>7));
  ml5=_mm256_setr_epi32(0,-(mdim>8),0,-(mdim>8),0,-(mdim>9),0,-(mdim>9));
  ml6=_mm256_setr_epi32(0,-(mdim>10),0,-(mdim>10),0,-(mdim>11),0,-(mdim>11));
#else
  ml1=_mm256_setr_epi32(-(mdim>0),-(mdim>0),-(mdim>1),-(mdim>1),-(mdim>2),-(mdim>2),-(mdim>3),-(mdim>3));
  ml2=_mm256_setr_epi32(-(mdim>4),-(mdim>4),-(mdim>5),-(mdim>5),-(mdim>6),-(mdim>6),-(mdim>7),-(mdim>7));
  ml3=_mm256_setr_epi32(-(mdim>8),-(mdim>8),-(mdim>9),-(mdim>9),-(mdim>10),-(mdim>10),-(mdim>11),-(mdim>11));
  ml4=_mm256_setr_epi32(-(mdim>12),-(mdim>12),-(mdim>13),-(mdim>13),-(mdim>14),-(mdim>14),-(mdim>15),-(mdim>15));
  ml5=_mm256_setr_epi32(-(mdim>16),-(mdim>16),-(mdim>17),-(mdim>17),-(mdim>18),-(mdim>18),-(mdim>19),-(mdim>19));
  ml6=_mm256_setr_epi32(-(mdim>20),-(mdim>20),-(mdim>21),-(mdim>21),-(mdim>22),-(mdim>22),-(mdim>23),-(mdim>23));
#endif
  btemp=bblk;
/*first do mArBr*/
  ctemp=cstartpos;
  for(ccol=0;ccol<ndim-3;ccol+=4){//loop over cblk-columns, calculate 4 columns of cblk in each iteration.
   cpref=ctemp;
   INIT_4col(64)
   atemp=ablk;
   for(acol=0;acol<kdim;acol++) KERNELk1//loop over ablk-columns, load 1 column of ablk in each micro-iteration.
   STOREIRREGM_C_1col_mArBr(c1,c2,c3)
   STOREIRREGM_C_1col_mArBr(c4,c5,c6)
   STOREIRREGM_C_1col_mArBr(c7,c8,c9)
   STOREIRREGM_C_1col_mArBr(c10,c11,c12)
  }
  for(;ccol<ndim;ccol++){
   INIT_1col(64)
   atemp=ablk;
   for(acol=0;acol<kdim;acol++) KERNELkr//loop over ablk-columns, load 1 column of ablk in each micro-iteration.
   STOREIRREGM_C_1col_mArBr(c1,c2,c3)
  }
/*then do mAiBi*/
  ctemp=cstartpos;ahead=ablk+BlkDimM*kdim;
  for(ccol=0;ccol<ndim-3;ccol+=4){
   cpref=ctemp;
   INIT_4col(128)
   atemp=ahead;
   for(acol=0;acol<kdim;acol++) KERNELk1
   STOREIRREGM_C_1col_mAiBi(c1,c2,c3)
   STOREIRREGM_C_1col_mAiBi(c4,c5,c6)
   STOREIRREGM_C_1col_mAiBi(c7,c8,c9)
   STOREIRREGM_C_1col_mAiBi(c10,c11,c12)
  }
  for(;ccol<ndim;ccol++){
   INIT_1col(128)
   atemp=ahead;
   for(acol=0;acol<kdim;acol++) KERNELkr
   STOREIRREGM_C_1col_mAiBi(c1,c2,c3)
  }
/*finally do AsBs*/
  ctemp=cstartpos;ahead=ablk+BlkDimM*kdim*2;
  for(ccol=0;ccol<ndim-3;ccol+=4){
   cpref=ctemp;
   INIT_4col(192)
   atemp=ahead;
   for(acol=0;acol<kdim;acol++) KERNELk1
   STOREIRREGM_C_1col_AsBs(c1,c2,c3)
   STOREIRREGM_C_1col_AsBs(c4,c5,c6)
   STOREIRREGM_C_1col_AsBs(c7,c8,c9)
   STOREIRREGM_C_1col_AsBs(c10,c11,c12)
  }
  for(;ccol<ndim;ccol++){
   INIT_1col(192)
   atemp=ahead;
   for(acol=0;acol<kdim;acol++) KERNELkr
   STOREIRREGM_C_1col_AsBs(c1,c2,c3)
  }
}
