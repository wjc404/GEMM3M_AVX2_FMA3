#define ZERO_VALUE 1.0
#define Mpos_to_Rpos(arow) ((arow)%COMPLEX_PER_VEC*2+(arow)/COMPLEX_PER_VEC%2+(arow)/COMPLEX_PER_VEC/2*COMPLEX_PER_VEC*2)
#define VEC_PER_COLUMN 3
static void load_irreg_a_c(FLOAT * __restrict__ astartpos,FLOAT * __restrict__ ablk,int lda,int mdim,int kdim){//sparse lazy mode
  int acol,arow;FLOAT *aread,*awrite;
  awrite=ablk;
/* first extract real part */
  aread=astartpos;
  for(acol=0;acol<kdim;acol++){
    for(arow=0;arow<mdim;arow++){
      awrite[Mpos_to_Rpos(arow)]=aread[arow*2];
    }
    for(;arow<BlkDimM;arow++){
      awrite[Mpos_to_Rpos(arow)]=ZERO_VALUE;
    }
    aread+=lda*2;awrite+=BlkDimM;
  }
/* then extract imaginary part */
  aread=astartpos;
  for(acol=0;acol<kdim;acol++){
    for(arow=0;arow<mdim;arow++){
      awrite[Mpos_to_Rpos(arow)]=aread[arow*2+1];
    }
    for(;arow<BlkDimM;arow++){
      awrite[Mpos_to_Rpos(arow)]=ZERO_VALUE;
    }
    aread+=lda*2;awrite+=BlkDimM;
  }
/* finally store "real+imaginary" */
  aread=astartpos;
  for(acol=0;acol<kdim;acol++){
    for(arow=0;arow<mdim;arow++){
      awrite[Mpos_to_Rpos(arow)]=aread[arow*2]+aread[arow*2+1];
    }
    for(;arow<BlkDimM;arow++){
      awrite[Mpos_to_Rpos(arow)]=ZERO_VALUE;
    }
    aread+=lda*2;awrite+=BlkDimM;
  }
}
static void load_irreg_a_r(FLOAT * __restrict__ astartpos,FLOAT * __restrict__ ablk,int lda,int mdim,int kdim){//sparse lazy mode
  int acol,arow,arow_w;FLOAT *aread,*awrite;
/* first extract real part */
  aread=astartpos;awrite=ablk;
  for(arow=0;arow<mdim;arow++){
    arow_w=Mpos_to_Rpos(arow);
    for(acol=0;acol<kdim;acol++){
      awrite[arow_w+acol*BlkDimM]=aread[acol*2];
    }
    aread+=lda*2;
  }
  for(acol=0;acol<kdim;acol++){
    for(arow=mdim;arow<BlkDimM;arow++){
      awrite[Mpos_to_Rpos(arow)]=ZERO_VALUE;
    }
    awrite+=BlkDimM;
  }
/* then extract imaginary part */
  aread=astartpos;
  for(arow=0;arow<mdim;arow++){
    arow_w=Mpos_to_Rpos(arow);
    for(acol=0;acol<kdim;acol++){
      awrite[arow_w+acol*BlkDimM]=aread[acol*2+1];
    }
    aread+=lda*2;
  }
  for(acol=0;acol<kdim;acol++){
    for(arow=mdim;arow<BlkDimM;arow++){
      awrite[Mpos_to_Rpos(arow)]=ZERO_VALUE;
    }
    awrite+=BlkDimM;
  }
/* finally store "real+imaginary" */
  aread=astartpos;
  for(arow=0;arow<mdim;arow++){
    arow_w=Mpos_to_Rpos(arow);
    for(acol=0;acol<kdim;acol++){
      awrite[arow_w+acol*BlkDimM]=aread[acol*2]+aread[acol*2+1];
    }
    aread+=lda*2;
  }
  for(acol=0;acol<kdim;acol++){
    for(arow=mdim;arow<BlkDimM;arow++){
      awrite[Mpos_to_Rpos(arow)]=ZERO_VALUE;
    }
    awrite+=BlkDimM;
  }
}
static void load_irreg_a_h(FLOAT * __restrict__ astartpos,FLOAT * __restrict__ ablk,int lda,int mdim,int kdim){//sparse lazy mode
  int acol,arow,arow_w;FLOAT *aread,*awrite;
/* first extract real part */
  aread=astartpos;awrite=ablk;
  for(arow=0;arow<mdim;arow++){
    arow_w=Mpos_to_Rpos(arow);
    for(acol=0;acol<kdim;acol++){
      awrite[arow_w+acol*BlkDimM]=aread[acol*2];
    }
    aread+=lda*2;
  }
  for(acol=0;acol<kdim;acol++){
    for(arow=mdim;arow<BlkDimM;arow++){
      awrite[Mpos_to_Rpos(arow)]=ZERO_VALUE;
    }
    awrite+=BlkDimM;
  }
/* then extract imaginary part */
  aread=astartpos;
  for(arow=0;arow<mdim;arow++){
    arow_w=Mpos_to_Rpos(arow);
    for(acol=0;acol<kdim;acol++){
      awrite[arow_w+acol*BlkDimM]=-aread[acol*2+1];
    }
    aread+=lda*2;
  }
  for(acol=0;acol<kdim;acol++){
    for(arow=mdim;arow<BlkDimM;arow++){
      awrite[Mpos_to_Rpos(arow)]=ZERO_VALUE;
    }
    awrite+=BlkDimM;
  }
/* finally store "real+imaginary" */
  aread=astartpos;
  for(arow=0;arow<mdim;arow++){
    arow_w=Mpos_to_Rpos(arow);
    for(acol=0;acol<kdim;acol++){
      awrite[arow_w+acol*BlkDimM]=aread[acol*2]-aread[acol*2+1];
    }
    aread+=lda*2;
  }
  for(acol=0;acol<kdim;acol++){
    for(arow=mdim;arow<BlkDimM;arow++){
      awrite[Mpos_to_Rpos(arow)]=ZERO_VALUE;
    }
    awrite+=BlkDimM;
  }
}
static void load_tail_a_c(FLOAT *astartpos,FLOAT *ablk,int lda,int mdim){load_irreg_a_c(astartpos,ablk,lda,mdim,BlkDimK);}
static void load_tail_a_r(FLOAT *astartpos,FLOAT *ablk,int lda,int mdim){load_irreg_a_r(astartpos,ablk,lda,mdim,BlkDimK);}
static void load_tail_a_h(FLOAT *astartpos,FLOAT *ablk,int lda,int mdim){load_irreg_a_h(astartpos,ablk,lda,mdim,BlkDimK);}
static void load_irregk_a_c(FLOAT *astartpos,FLOAT *ablk,int lda,int kdim){load_irreg_a_c(astartpos,ablk,lda,BlkDimM,kdim);}
static void load_irregk_a_r(FLOAT *astartpos,FLOAT *ablk,int lda,int kdim){load_irreg_a_r(astartpos,ablk,lda,BlkDimM,kdim);}
static void load_irregk_a_h(FLOAT *astartpos,FLOAT *ablk,int lda,int kdim){load_irreg_a_h(astartpos,ablk,lda,BlkDimM,kdim);}
# ifdef DOUBLE
 #define COPY_VECTYPE __m256d
 #define COPY_LOADU _mm256_loadu_pd
 #define COPY_STOREU _mm256_storeu_pd
 #define COPY_UNPACKREAL(y1,y2) _mm256_unpacklo_pd(y1,y2)
 #define COPY_UNPACKIMAG(y1,y2) _mm256_unpackhi_pd(y1,y2)
 #define COPY_VECADD _mm256_add_pd
# else
 #define COPY_VECTYPE __m256
 #define COPY_LOADU _mm256_loadu_ps
 #define COPY_STOREU _mm256_storeu_ps
 #define COPY_UNPACKREAL(y1,y2) _mm256_blend_ps(y1,_mm256_permute_ps(y2,177),170)
 #define COPY_UNPACKIMAG(y1,y2) _mm256_blend_ps(y2,_mm256_permute_ps(y1,177),85)
 #define COPY_VECADD _mm256_add_ps
# endif
static void load_reg_a_c(FLOAT * __restrict__ astartpos,FLOAT * __restrict__ ablk,int lda){
  int acol,arow;FLOAT *aread,*awrite1,*awrite2,*awrite3;COPY_VECTYPE ain1,ain2,aout1,aout2,aout3;
  awrite1=ablk;awrite2=ablk+BlkDimM*BlkDimK;awrite3=ablk+2*BlkDimM*BlkDimK;aread=astartpos;
  for(acol=0;acol<BlkDimK;acol++){
    for(arow=0;arow<BlkDimM;arow+=COMPLEX_PER_VEC*2){
      ain1=COPY_LOADU(aread+arow*2);
      ain2=COPY_LOADU(aread+arow*2+COMPLEX_PER_VEC*2);
      aout1=COPY_UNPACKREAL(ain1,ain2);//extract real part
      aout2=COPY_UNPACKIMAG(ain1,ain2);//extract imaginary part
      aout3=COPY_VECADD(aout1,aout2);//calculate A(r+i)
      COPY_STOREU(awrite1+arow,aout1);
      COPY_STOREU(awrite2+arow,aout2);
      COPY_STOREU(awrite3+arow,aout3);
    }
    awrite1+=BlkDimM;awrite2+=BlkDimM;awrite3+=BlkDimM;aread+=lda*2;
  }
}
static void load_reg_a_r(FLOAT * __restrict__ astartpos,FLOAT * __restrict__ ablk,int lda){
  int acol,arow,arow_w1,arow_w2,arow_w3,arow_w4;FLOAT *ar1,*ar2,*ar3,*ar4,*awrite;
  for(arow=0;arow<BlkDimM;arow+=4){
    ar1=astartpos+arow*lda*2;
    ar2=ar1+lda*2;ar3=ar2+lda*2;ar4=ar3+lda*2;
    awrite=ablk;
    arow_w1=Mpos_to_Rpos(arow+0);
    arow_w2=Mpos_to_Rpos(arow+1);
    arow_w3=Mpos_to_Rpos(arow+2);
    arow_w4=Mpos_to_Rpos(arow+3);
/* first extract real part */
    for(acol=0;acol<BlkDimK;acol++){
      awrite[arow_w1]=ar1[acol*2];
      awrite[arow_w2]=ar2[acol*2];
      awrite[arow_w3]=ar3[acol*2];
      awrite[arow_w4]=ar4[acol*2];
      awrite+=BlkDimM;
    }
/* then extract imaginary part */
    for(acol=0;acol<BlkDimK;acol++){
      awrite[arow_w1]=ar1[acol*2+1];
      awrite[arow_w2]=ar2[acol*2+1];
      awrite[arow_w3]=ar3[acol*2+1];
      awrite[arow_w4]=ar4[acol*2+1];
      awrite+=BlkDimM;
    }
/* finally store "real+imaginary" */
    for(acol=0;acol<BlkDimK;acol++){
      awrite[arow_w1]=ar1[acol*2]+ar1[acol*2+1];
      awrite[arow_w2]=ar2[acol*2]+ar2[acol*2+1];
      awrite[arow_w3]=ar3[acol*2]+ar3[acol*2+1];
      awrite[arow_w4]=ar4[acol*2]+ar4[acol*2+1];
      awrite+=BlkDimM;
    }
  }
}
static void load_reg_a_h(FLOAT * __restrict__ astartpos,FLOAT * __restrict__ ablk,int lda){
  int acol,arow,arow_w1,arow_w2,arow_w3,arow_w4;FLOAT *ar1,*ar2,*ar3,*ar4,*awrite;
  for(arow=0;arow<BlkDimM;arow+=4){
    ar1=astartpos+arow*lda*2;
    ar2=ar1+lda*2;ar3=ar2+lda*2;ar4=ar3+lda*2;
    awrite=ablk;
    arow_w1=Mpos_to_Rpos(arow+0);
    arow_w2=Mpos_to_Rpos(arow+1);
    arow_w3=Mpos_to_Rpos(arow+2);
    arow_w4=Mpos_to_Rpos(arow+3);
/* first extract real part */
    for(acol=0;acol<BlkDimK;acol++){
      awrite[arow_w1]=ar1[acol*2];
      awrite[arow_w2]=ar2[acol*2];
      awrite[arow_w3]=ar3[acol*2];
      awrite[arow_w4]=ar4[acol*2];
      awrite+=BlkDimM;
    }
/* then extract imaginary part */
    for(acol=0;acol<BlkDimK;acol++){
      awrite[arow_w1]=-ar1[acol*2+1];
      awrite[arow_w2]=-ar2[acol*2+1];
      awrite[arow_w3]=-ar3[acol*2+1];
      awrite[arow_w4]=-ar4[acol*2+1];
      awrite+=BlkDimM;
    }
/* finally store "real+imaginary" */
    for(acol=0;acol<BlkDimK;acol++){
      awrite[arow_w1]=ar1[acol*2]-ar1[acol*2+1];
      awrite[arow_w2]=ar2[acol*2]-ar2[acol*2+1];
      awrite[arow_w3]=ar3[acol*2]-ar3[acol*2+1];
      awrite[arow_w4]=ar4[acol*2]-ar4[acol*2+1];
      awrite+=BlkDimM;
    }
  }
}
#define COPY_EXTRACT_mR(mem,alpha) (-(*(mem))*alpha[0]+(*(mem+1))*alpha[1])//retain
#define COPY_EXTRACT_mI(mem,alpha) (-(*(mem))*alpha[1]-(*(mem+1))*alpha[0])//retain
#define COPY_EXTRACT_mr(mem,alpha) (-(*(mem))*alpha[0]-(*(mem+1))*alpha[1])//conjug
#define COPY_EXTRACT_mi(mem,alpha) (-(*(mem))*alpha[1]+(*(mem+1))*alpha[0])//conjug
static void load_irreg_b_c(FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,int ndim,int kdim,FLOAT * __restrict__ alpha){//dense rearr(old) lazy mode
  FLOAT *bin1,*bin2,*bin3,*bin4,*bout1,*bout2,*bout3;int bcol,brow;FLOAT real,imag,sum;
  bin1=bstartpos;bin2=bin1+ldb*2;bin3=bin2+ldb*2;bin4=bin3+ldb*2;
  bout1=bblk;bout2=bblk+kdim*ndim;bout3=bblk+kdim*ndim*2;
  for(bcol=0;bcol<ndim-3;bcol+=4){
    for(brow=0;brow<kdim;brow++){
      real=COPY_EXTRACT_mR(bin1,alpha);imag=COPY_EXTRACT_mI(bin1,alpha);sum=-(real+imag);bin1+=2;
      *bout1=real;*bout2=imag;*bout3=sum;bout1++;bout2++;bout3++;
      real=COPY_EXTRACT_mR(bin2,alpha);imag=COPY_EXTRACT_mI(bin2,alpha);sum=-(real+imag);bin2+=2;
      *bout1=real;*bout2=imag;*bout3=sum;bout1++;bout2++;bout3++;
      real=COPY_EXTRACT_mR(bin3,alpha);imag=COPY_EXTRACT_mI(bin3,alpha);sum=-(real+imag);bin3+=2;
      *bout1=real;*bout2=imag;*bout3=sum;bout1++;bout2++;bout3++;
      real=COPY_EXTRACT_mR(bin4,alpha);imag=COPY_EXTRACT_mI(bin4,alpha);sum=-(real+imag);bin4+=2;
      *bout1=real;*bout2=imag;*bout3=sum;bout1++;bout2++;bout3++;
    }
    bin1+=2*(4*ldb-kdim);
    bin2+=2*(4*ldb-kdim);
    bin3+=2*(4*ldb-kdim);
    bin4+=2*(4*ldb-kdim);
  }
  for(;bcol<ndim;bcol++){
    for(brow=0;brow<kdim;brow++){
      real=COPY_EXTRACT_mR(bin1,alpha);imag=COPY_EXTRACT_mI(bin1,alpha);sum=-(real+imag);bin1+=2;
      *bout1=real;*bout2=imag;*bout3=sum;bout1++;bout2++;bout3++;
    }
    bin1+=2*(ldb-kdim);
  }
}
static void load_irreg_b_r(FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,int ndim,int kdim,FLOAT * __restrict__ alpha){//dense rearr(old) lazy mode
  FLOAT *bin,*bout1,*bout2,*bout3;int bcol,brow;FLOAT real,imag,sum;
  bin=bstartpos;
  for(brow=0;brow<kdim;brow++){
    bout1=bblk+brow*4;bout2=bout1+kdim*ndim;bout3=bout2+kdim*ndim;
    for(bcol=0;bcol<ndim-3;bcol+=4){
      real=COPY_EXTRACT_mR(bin,alpha);imag=COPY_EXTRACT_mI(bin,alpha);sum=-(real+imag);bin+=2;
      *bout1=real;*bout2=imag;*bout3=sum;bout1++;bout2++;bout3++;
      real=COPY_EXTRACT_mR(bin,alpha);imag=COPY_EXTRACT_mI(bin,alpha);sum=-(real+imag);bin+=2;
      *bout1=real;*bout2=imag;*bout3=sum;bout1++;bout2++;bout3++;
      real=COPY_EXTRACT_mR(bin,alpha);imag=COPY_EXTRACT_mI(bin,alpha);sum=-(real+imag);bin+=2;
      *bout1=real;*bout2=imag;*bout3=sum;bout1++;bout2++;bout3++;
      real=COPY_EXTRACT_mR(bin,alpha);imag=COPY_EXTRACT_mI(bin,alpha);sum=-(real+imag);bin+=2;
      *bout1=real;*bout2=imag;*bout3=sum;bout1+=4*kdim-3;bout2+=4*kdim-3;bout3+=4*kdim-3;
    }
    bout1-=3*brow;bout2-=3*brow;bout3-=3*brow;
    for(;bcol<ndim;bcol++){
      real=COPY_EXTRACT_mR(bin,alpha);imag=COPY_EXTRACT_mI(bin,alpha);sum=-(real+imag);bin+=2;
      *bout1=real;*bout2=imag;*bout3=sum;bout1+=kdim;bout2+=kdim;bout3+=kdim;
    }
    bin+=2*(ldb-ndim);
  }
}
static void load_irreg_b_h(FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,int ndim,int kdim,FLOAT * __restrict__ alpha){//dense rearr(old) lazy mode
  FLOAT *bin,*bout1,*bout2,*bout3;int bcol,brow;FLOAT real,imag,sum;
  bin=bstartpos;
  for(brow=0;brow<kdim;brow++){
    bout1=bblk+brow*4;bout2=bout1+kdim*ndim;bout3=bout2+kdim*ndim;
    for(bcol=0;bcol<ndim-3;bcol+=4){
      real=COPY_EXTRACT_mr(bin,alpha);imag=COPY_EXTRACT_mi(bin,alpha);sum=-(real+imag);bin+=2;
      *bout1=real;*bout2=imag;*bout3=sum;bout1++;bout2++;bout3++;
      real=COPY_EXTRACT_mr(bin,alpha);imag=COPY_EXTRACT_mi(bin,alpha);sum=-(real+imag);bin+=2;
      *bout1=real;*bout2=imag;*bout3=sum;bout1++;bout2++;bout3++;
      real=COPY_EXTRACT_mr(bin,alpha);imag=COPY_EXTRACT_mi(bin,alpha);sum=-(real+imag);bin+=2;
      *bout1=real;*bout2=imag;*bout3=sum;bout1++;bout2++;bout3++;
      real=COPY_EXTRACT_mr(bin,alpha);imag=COPY_EXTRACT_mi(bin,alpha);sum=-(real+imag);bin+=2;
      *bout1=real;*bout2=imag;*bout3=sum;bout1+=4*kdim-3;bout2+=4*kdim-3;bout3+=4*kdim-3;
    }
    bout1-=3*brow;bout2-=3*brow;bout3-=3*brow;
    for(;bcol<ndim;bcol++){
      real=COPY_EXTRACT_mr(bin,alpha);imag=COPY_EXTRACT_mi(bin,alpha);sum=-(real+imag);bin+=2;
      *bout1=real;*bout2=imag;*bout3=sum;bout1+=kdim;bout2+=kdim;bout3+=kdim;
    }
    bin+=2*(ldb-ndim);
  }
}
static void load_reg_b_c(FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,FLOAT * __restrict__ alpha){
 FLOAT *inb1,*inb2,*inb3,*inb4,*outb1,*outb2,*outb3;
 int bcol,brow;
 outb1=bblk;outb2=bblk+BlkDimK*BlkDimN;outb3=bblk+2*BlkDimK*BlkDimN;FLOAT real,imag,sum;
 inb1=bstartpos;
 inb2=inb1+2*ldb;
 inb3=inb2+2*ldb;
 inb4=inb3+2*ldb;
 for(bcol=0;bcol<(BlkDimN/4);bcol++){
  for(brow=0;brow<(BlkDimK/4);brow++){
   real=COPY_EXTRACT_mR(inb1,alpha);imag=COPY_EXTRACT_mI(inb1,alpha);sum=-(real+imag);inb1+=2;
   outb1[0]=real;outb2[0]=imag;outb3[0]=sum;
   real=COPY_EXTRACT_mR(inb2,alpha);imag=COPY_EXTRACT_mI(inb2,alpha);sum=-(real+imag);inb2+=2;
   outb1[1]=real;outb2[1]=imag;outb3[1]=sum;
   real=COPY_EXTRACT_mR(inb3,alpha);imag=COPY_EXTRACT_mI(inb3,alpha);sum=-(real+imag);inb3+=2;
   outb1[2]=real;outb2[2]=imag;outb3[2]=sum;
   real=COPY_EXTRACT_mR(inb4,alpha);imag=COPY_EXTRACT_mI(inb4,alpha);sum=-(real+imag);inb4+=2;
   outb1[3]=real;outb2[3]=imag;outb3[3]=sum;
   outb1+=4;outb2+=4;outb3+=4;
  }
  inb1+=ldb*2;inb2+=ldb*2;inb3+=ldb*2;inb4+=ldb*2;
  inb4-=(bcol==(BlkDimN/4)-1)*(ldb*2*BlkDimN);
  for(;brow<2*(BlkDimK/4);brow++){
   real=COPY_EXTRACT_mR(inb1,alpha);imag=COPY_EXTRACT_mI(inb1,alpha);sum=-(real+imag);inb1+=2;
   outb1[0]=real;outb2[0]=imag;outb3[0]=sum;
   real=COPY_EXTRACT_mR(inb2,alpha);imag=COPY_EXTRACT_mI(inb2,alpha);sum=-(real+imag);inb2+=2;
   outb1[1]=real;outb2[1]=imag;outb3[1]=sum;
   real=COPY_EXTRACT_mR(inb3,alpha);imag=COPY_EXTRACT_mI(inb3,alpha);sum=-(real+imag);inb3+=2;
   outb1[2]=real;outb2[2]=imag;outb3[2]=sum;
   real=COPY_EXTRACT_mR(inb4,alpha);imag=COPY_EXTRACT_mI(inb4,alpha);sum=-(real+imag);inb4+=2;
   outb1[3]=real;outb2[3]=imag;outb3[3]=sum;
   outb1+=4;outb2+=4;outb3+=4;
  }
  inb1+=ldb*2;inb2+=ldb*2;inb3+=ldb*2;inb4+=ldb*2;
  inb3-=(bcol==(BlkDimN/4)-1)*(ldb*2*BlkDimN);
  for(;brow<3*(BlkDimK/4);brow++){
   real=COPY_EXTRACT_mR(inb1,alpha);imag=COPY_EXTRACT_mI(inb1,alpha);sum=-(real+imag);inb1+=2;
   outb1[0]=real;outb2[0]=imag;outb3[0]=sum;
   real=COPY_EXTRACT_mR(inb2,alpha);imag=COPY_EXTRACT_mI(inb2,alpha);sum=-(real+imag);inb2+=2;
   outb1[1]=real;outb2[1]=imag;outb3[1]=sum;
   real=COPY_EXTRACT_mR(inb3,alpha);imag=COPY_EXTRACT_mI(inb3,alpha);sum=-(real+imag);inb3+=2;
   outb1[2]=real;outb2[2]=imag;outb3[2]=sum;
   real=COPY_EXTRACT_mR(inb4,alpha);imag=COPY_EXTRACT_mI(inb4,alpha);sum=-(real+imag);inb4+=2;
   outb1[3]=real;outb2[3]=imag;outb3[3]=sum;
   outb1+=4;outb2+=4;outb3+=4;
  }
  inb1+=ldb*2;inb2+=ldb*2;inb3+=ldb*2;inb4+=ldb*2;
  inb2-=(bcol==(BlkDimN/4)-1)*(ldb*2*BlkDimN);
  for(;brow<BlkDimK;brow++){
   real=COPY_EXTRACT_mR(inb1,alpha);imag=COPY_EXTRACT_mI(inb1,alpha);sum=-(real+imag);inb1+=2;
   outb1[0]=real;outb2[0]=imag;outb3[0]=sum;
   real=COPY_EXTRACT_mR(inb2,alpha);imag=COPY_EXTRACT_mI(inb2,alpha);sum=-(real+imag);inb2+=2;
   outb1[1]=real;outb2[1]=imag;outb3[1]=sum;
   real=COPY_EXTRACT_mR(inb3,alpha);imag=COPY_EXTRACT_mI(inb3,alpha);sum=-(real+imag);inb3+=2;
   outb1[2]=real;outb2[2]=imag;outb3[2]=sum;
   real=COPY_EXTRACT_mR(inb4,alpha);imag=COPY_EXTRACT_mI(inb4,alpha);sum=-(real+imag);inb4+=2;
   outb1[3]=real;outb2[3]=imag;outb3[3]=sum;
   outb1+=4;outb2+=4;outb3+=4;
  }
  inb1+=2*(ldb-BlkDimK);
  inb2+=2*(ldb-BlkDimK);
  inb3+=2*(ldb-BlkDimK);
  inb4+=2*(ldb-BlkDimK);
 }
}
#define COPY_BR_4x4 {\
  real=COPY_EXTRACT_mR(bin1,alpha);imag=COPY_EXTRACT_mI(bin1,alpha);sum=-(real+imag);bin1+=2;\
  bout1[0]=real;bout2[0]=imag;bout3[0]=sum;\
  real=COPY_EXTRACT_mR(bin1,alpha);imag=COPY_EXTRACT_mI(bin1,alpha);sum=-(real+imag);bin1+=2;\
  bout1[1]=real;bout2[1]=imag;bout3[1]=sum;\
  real=COPY_EXTRACT_mR(bin1,alpha);imag=COPY_EXTRACT_mI(bin1,alpha);sum=-(real+imag);bin1+=2;\
  bout1[2]=real;bout2[2]=imag;bout3[2]=sum;\
  real=COPY_EXTRACT_mR(bin1,alpha);imag=COPY_EXTRACT_mI(bin1,alpha);sum=-(real+imag);bin1+=2;\
  bout1[3]=real;bout2[3]=imag;bout3[3]=sum;\
  real=COPY_EXTRACT_mR(bin2,alpha);imag=COPY_EXTRACT_mI(bin2,alpha);sum=-(real+imag);bin2+=2;\
  bout1[4]=real;bout2[4]=imag;bout3[4]=sum;\
  real=COPY_EXTRACT_mR(bin2,alpha);imag=COPY_EXTRACT_mI(bin2,alpha);sum=-(real+imag);bin2+=2;\
  bout1[5]=real;bout2[5]=imag;bout3[5]=sum;\
  real=COPY_EXTRACT_mR(bin2,alpha);imag=COPY_EXTRACT_mI(bin2,alpha);sum=-(real+imag);bin2+=2;\
  bout1[6]=real;bout2[6]=imag;bout3[6]=sum;\
  real=COPY_EXTRACT_mR(bin2,alpha);imag=COPY_EXTRACT_mI(bin2,alpha);sum=-(real+imag);bin2+=2;\
  bout1[7]=real;bout2[7]=imag;bout3[7]=sum;\
  real=COPY_EXTRACT_mR(bin3,alpha);imag=COPY_EXTRACT_mI(bin3,alpha);sum=-(real+imag);bin3+=2;\
  bout1[8]=real;bout2[8]=imag;bout3[8]=sum;\
  real=COPY_EXTRACT_mR(bin3,alpha);imag=COPY_EXTRACT_mI(bin3,alpha);sum=-(real+imag);bin3+=2;\
  bout1[9]=real;bout2[9]=imag;bout3[9]=sum;\
  real=COPY_EXTRACT_mR(bin3,alpha);imag=COPY_EXTRACT_mI(bin3,alpha);sum=-(real+imag);bin3+=2;\
  bout1[10]=real;bout2[10]=imag;bout3[10]=sum;\
  real=COPY_EXTRACT_mR(bin3,alpha);imag=COPY_EXTRACT_mI(bin3,alpha);sum=-(real+imag);bin3+=2;\
  bout1[11]=real;bout2[11]=imag;bout3[11]=sum;\
  real=COPY_EXTRACT_mR(bin4,alpha);imag=COPY_EXTRACT_mI(bin4,alpha);sum=-(real+imag);bin4+=2;\
  bout1[12]=real;bout2[12]=imag;bout3[12]=sum;\
  real=COPY_EXTRACT_mR(bin4,alpha);imag=COPY_EXTRACT_mI(bin4,alpha);sum=-(real+imag);bin4+=2;\
  bout1[13]=real;bout2[13]=imag;bout3[13]=sum;\
  real=COPY_EXTRACT_mR(bin4,alpha);imag=COPY_EXTRACT_mI(bin4,alpha);sum=-(real+imag);bin4+=2;\
  bout1[14]=real;bout2[14]=imag;bout3[14]=sum;\
  real=COPY_EXTRACT_mR(bin4,alpha);imag=COPY_EXTRACT_mI(bin4,alpha);sum=-(real+imag);bin4+=2;\
  bout1[15]=real;bout2[15]=imag;bout3[15]=sum;\
  bout1+=4*BlkDimK;bout2+=4*BlkDimK;bout3+=4*BlkDimK;\
}
#define COPY_BH_4x4 {\
  real=COPY_EXTRACT_mr(bin1,alpha);imag=COPY_EXTRACT_mi(bin1,alpha);sum=-(real+imag);bin1+=2;\
  bout1[0]=real;bout2[0]=imag;bout3[0]=sum;\
  real=COPY_EXTRACT_mr(bin1,alpha);imag=COPY_EXTRACT_mi(bin1,alpha);sum=-(real+imag);bin1+=2;\
  bout1[1]=real;bout2[1]=imag;bout3[1]=sum;\
  real=COPY_EXTRACT_mr(bin1,alpha);imag=COPY_EXTRACT_mi(bin1,alpha);sum=-(real+imag);bin1+=2;\
  bout1[2]=real;bout2[2]=imag;bout3[2]=sum;\
  real=COPY_EXTRACT_mr(bin1,alpha);imag=COPY_EXTRACT_mi(bin1,alpha);sum=-(real+imag);bin1+=2;\
  bout1[3]=real;bout2[3]=imag;bout3[3]=sum;\
  real=COPY_EXTRACT_mr(bin2,alpha);imag=COPY_EXTRACT_mi(bin2,alpha);sum=-(real+imag);bin2+=2;\
  bout1[4]=real;bout2[4]=imag;bout3[4]=sum;\
  real=COPY_EXTRACT_mr(bin2,alpha);imag=COPY_EXTRACT_mi(bin2,alpha);sum=-(real+imag);bin2+=2;\
  bout1[5]=real;bout2[5]=imag;bout3[5]=sum;\
  real=COPY_EXTRACT_mr(bin2,alpha);imag=COPY_EXTRACT_mi(bin2,alpha);sum=-(real+imag);bin2+=2;\
  bout1[6]=real;bout2[6]=imag;bout3[6]=sum;\
  real=COPY_EXTRACT_mr(bin2,alpha);imag=COPY_EXTRACT_mi(bin2,alpha);sum=-(real+imag);bin2+=2;\
  bout1[7]=real;bout2[7]=imag;bout3[7]=sum;\
  real=COPY_EXTRACT_mr(bin3,alpha);imag=COPY_EXTRACT_mi(bin3,alpha);sum=-(real+imag);bin3+=2;\
  bout1[8]=real;bout2[8]=imag;bout3[8]=sum;\
  real=COPY_EXTRACT_mr(bin3,alpha);imag=COPY_EXTRACT_mi(bin3,alpha);sum=-(real+imag);bin3+=2;\
  bout1[9]=real;bout2[9]=imag;bout3[9]=sum;\
  real=COPY_EXTRACT_mr(bin3,alpha);imag=COPY_EXTRACT_mi(bin3,alpha);sum=-(real+imag);bin3+=2;\
  bout1[10]=real;bout2[10]=imag;bout3[10]=sum;\
  real=COPY_EXTRACT_mr(bin3,alpha);imag=COPY_EXTRACT_mi(bin3,alpha);sum=-(real+imag);bin3+=2;\
  bout1[11]=real;bout2[11]=imag;bout3[11]=sum;\
  real=COPY_EXTRACT_mr(bin4,alpha);imag=COPY_EXTRACT_mi(bin4,alpha);sum=-(real+imag);bin4+=2;\
  bout1[12]=real;bout2[12]=imag;bout3[12]=sum;\
  real=COPY_EXTRACT_mr(bin4,alpha);imag=COPY_EXTRACT_mi(bin4,alpha);sum=-(real+imag);bin4+=2;\
  bout1[13]=real;bout2[13]=imag;bout3[13]=sum;\
  real=COPY_EXTRACT_mr(bin4,alpha);imag=COPY_EXTRACT_mi(bin4,alpha);sum=-(real+imag);bin4+=2;\
  bout1[14]=real;bout2[14]=imag;bout3[14]=sum;\
  real=COPY_EXTRACT_mr(bin4,alpha);imag=COPY_EXTRACT_mi(bin4,alpha);sum=-(real+imag);bin4+=2;\
  bout1[15]=real;bout2[15]=imag;bout3[15]=sum;\
  bout1+=4*BlkDimK;bout2+=4*BlkDimK;bout3+=4*BlkDimK;\
}
#define COPY_BR_4x3 {\
  real=COPY_EXTRACT_mR(bin1,alpha);imag=COPY_EXTRACT_mI(bin1,alpha);sum=-(real+imag);bin1+=2;\
  bout1[0]=real;bout2[0]=imag;bout3[0]=sum;\
  real=COPY_EXTRACT_mR(bin1,alpha);imag=COPY_EXTRACT_mI(bin1,alpha);sum=-(real+imag);bin1+=2;\
  bout1[1]=real;bout2[1]=imag;bout3[1]=sum;\
  real=COPY_EXTRACT_mR(bin1,alpha);imag=COPY_EXTRACT_mI(bin1,alpha);sum=-(real+imag);bin1+=2;\
  bout1[2]=real;bout2[2]=imag;bout3[2]=sum;\
  real=COPY_EXTRACT_mR(bin2,alpha);imag=COPY_EXTRACT_mI(bin2,alpha);sum=-(real+imag);bin2+=2;\
  bout1[4]=real;bout2[4]=imag;bout3[4]=sum;\
  real=COPY_EXTRACT_mR(bin2,alpha);imag=COPY_EXTRACT_mI(bin2,alpha);sum=-(real+imag);bin2+=2;\
  bout1[5]=real;bout2[5]=imag;bout3[5]=sum;\
  real=COPY_EXTRACT_mR(bin2,alpha);imag=COPY_EXTRACT_mI(bin2,alpha);sum=-(real+imag);bin2+=2;\
  bout1[6]=real;bout2[6]=imag;bout3[6]=sum;\
  real=COPY_EXTRACT_mR(bin3,alpha);imag=COPY_EXTRACT_mI(bin3,alpha);sum=-(real+imag);bin3+=2;\
  bout1[8]=real;bout2[8]=imag;bout3[8]=sum;\
  real=COPY_EXTRACT_mR(bin3,alpha);imag=COPY_EXTRACT_mI(bin3,alpha);sum=-(real+imag);bin3+=2;\
  bout1[9]=real;bout2[9]=imag;bout3[9]=sum;\
  real=COPY_EXTRACT_mR(bin3,alpha);imag=COPY_EXTRACT_mI(bin3,alpha);sum=-(real+imag);bin3+=2;\
  bout1[10]=real;bout2[10]=imag;bout3[10]=sum;\
  real=COPY_EXTRACT_mR(bin4,alpha);imag=COPY_EXTRACT_mI(bin4,alpha);sum=-(real+imag);bin4+=2;\
  bout1[12]=real;bout2[12]=imag;bout3[12]=sum;\
  real=COPY_EXTRACT_mR(bin4,alpha);imag=COPY_EXTRACT_mI(bin4,alpha);sum=-(real+imag);bin4+=2;\
  bout1[13]=real;bout2[13]=imag;bout3[13]=sum;\
  real=COPY_EXTRACT_mR(bin4,alpha);imag=COPY_EXTRACT_mI(bin4,alpha);sum=-(real+imag);bin4+=2;\
  bout1[14]=real;bout2[14]=imag;bout3[14]=sum;\
}
#define COPY_BH_4x3 {\
  real=COPY_EXTRACT_mr(bin1,alpha);imag=COPY_EXTRACT_mi(bin1,alpha);sum=-(real+imag);bin1+=2;\
  bout1[0]=real;bout2[0]=imag;bout3[0]=sum;\
  real=COPY_EXTRACT_mr(bin1,alpha);imag=COPY_EXTRACT_mi(bin1,alpha);sum=-(real+imag);bin1+=2;\
  bout1[1]=real;bout2[1]=imag;bout3[1]=sum;\
  real=COPY_EXTRACT_mr(bin1,alpha);imag=COPY_EXTRACT_mi(bin1,alpha);sum=-(real+imag);bin1+=2;\
  bout1[2]=real;bout2[2]=imag;bout3[2]=sum;\
  real=COPY_EXTRACT_mr(bin2,alpha);imag=COPY_EXTRACT_mi(bin2,alpha);sum=-(real+imag);bin2+=2;\
  bout1[4]=real;bout2[4]=imag;bout3[4]=sum;\
  real=COPY_EXTRACT_mr(bin2,alpha);imag=COPY_EXTRACT_mi(bin2,alpha);sum=-(real+imag);bin2+=2;\
  bout1[5]=real;bout2[5]=imag;bout3[5]=sum;\
  real=COPY_EXTRACT_mr(bin2,alpha);imag=COPY_EXTRACT_mi(bin2,alpha);sum=-(real+imag);bin2+=2;\
  bout1[6]=real;bout2[6]=imag;bout3[6]=sum;\
  real=COPY_EXTRACT_mr(bin3,alpha);imag=COPY_EXTRACT_mi(bin3,alpha);sum=-(real+imag);bin3+=2;\
  bout1[8]=real;bout2[8]=imag;bout3[8]=sum;\
  real=COPY_EXTRACT_mr(bin3,alpha);imag=COPY_EXTRACT_mi(bin3,alpha);sum=-(real+imag);bin3+=2;\
  bout1[9]=real;bout2[9]=imag;bout3[9]=sum;\
  real=COPY_EXTRACT_mr(bin3,alpha);imag=COPY_EXTRACT_mi(bin3,alpha);sum=-(real+imag);bin3+=2;\
  bout1[10]=real;bout2[10]=imag;bout3[10]=sum;\
  real=COPY_EXTRACT_mr(bin4,alpha);imag=COPY_EXTRACT_mi(bin4,alpha);sum=-(real+imag);bin4+=2;\
  bout1[12]=real;bout2[12]=imag;bout3[12]=sum;\
  real=COPY_EXTRACT_mr(bin4,alpha);imag=COPY_EXTRACT_mi(bin4,alpha);sum=-(real+imag);bin4+=2;\
  bout1[13]=real;bout2[13]=imag;bout3[13]=sum;\
  real=COPY_EXTRACT_mr(bin4,alpha);imag=COPY_EXTRACT_mi(bin4,alpha);sum=-(real+imag);bin4+=2;\
  bout1[14]=real;bout2[14]=imag;bout3[14]=sum;\
}
#define COPY_BR_4x2 {\
  real=COPY_EXTRACT_mR(bin1,alpha);imag=COPY_EXTRACT_mI(bin1,alpha);sum=-(real+imag);bin1+=2;\
  bout1[0]=real;bout2[0]=imag;bout3[0]=sum;\
  real=COPY_EXTRACT_mR(bin1,alpha);imag=COPY_EXTRACT_mI(bin1,alpha);sum=-(real+imag);bin1+=2;\
  bout1[1]=real;bout2[1]=imag;bout3[1]=sum;\
  real=COPY_EXTRACT_mR(bin2,alpha);imag=COPY_EXTRACT_mI(bin2,alpha);sum=-(real+imag);bin2+=2;\
  bout1[4]=real;bout2[4]=imag;bout3[4]=sum;\
  real=COPY_EXTRACT_mR(bin2,alpha);imag=COPY_EXTRACT_mI(bin2,alpha);sum=-(real+imag);bin2+=2;\
  bout1[5]=real;bout2[5]=imag;bout3[5]=sum;\
  real=COPY_EXTRACT_mR(bin3,alpha);imag=COPY_EXTRACT_mI(bin3,alpha);sum=-(real+imag);bin3+=2;\
  bout1[8]=real;bout2[8]=imag;bout3[8]=sum;\
  real=COPY_EXTRACT_mR(bin3,alpha);imag=COPY_EXTRACT_mI(bin3,alpha);sum=-(real+imag);bin3+=2;\
  bout1[9]=real;bout2[9]=imag;bout3[9]=sum;\
  real=COPY_EXTRACT_mR(bin4,alpha);imag=COPY_EXTRACT_mI(bin4,alpha);sum=-(real+imag);bin4+=2;\
  bout1[12]=real;bout2[12]=imag;bout3[12]=sum;\
  real=COPY_EXTRACT_mR(bin4,alpha);imag=COPY_EXTRACT_mI(bin4,alpha);sum=-(real+imag);bin4+=2;\
  bout1[13]=real;bout2[13]=imag;bout3[13]=sum;\
}
#define COPY_BH_4x2 {\
  real=COPY_EXTRACT_mr(bin1,alpha);imag=COPY_EXTRACT_mi(bin1,alpha);sum=-(real+imag);bin1+=2;\
  bout1[0]=real;bout2[0]=imag;bout3[0]=sum;\
  real=COPY_EXTRACT_mr(bin1,alpha);imag=COPY_EXTRACT_mi(bin1,alpha);sum=-(real+imag);bin1+=2;\
  bout1[1]=real;bout2[1]=imag;bout3[1]=sum;\
  real=COPY_EXTRACT_mr(bin2,alpha);imag=COPY_EXTRACT_mi(bin2,alpha);sum=-(real+imag);bin2+=2;\
  bout1[4]=real;bout2[4]=imag;bout3[4]=sum;\
  real=COPY_EXTRACT_mr(bin2,alpha);imag=COPY_EXTRACT_mi(bin2,alpha);sum=-(real+imag);bin2+=2;\
  bout1[5]=real;bout2[5]=imag;bout3[5]=sum;\
  real=COPY_EXTRACT_mr(bin3,alpha);imag=COPY_EXTRACT_mi(bin3,alpha);sum=-(real+imag);bin3+=2;\
  bout1[8]=real;bout2[8]=imag;bout3[8]=sum;\
  real=COPY_EXTRACT_mr(bin3,alpha);imag=COPY_EXTRACT_mi(bin3,alpha);sum=-(real+imag);bin3+=2;\
  bout1[9]=real;bout2[9]=imag;bout3[9]=sum;\
  real=COPY_EXTRACT_mr(bin4,alpha);imag=COPY_EXTRACT_mi(bin4,alpha);sum=-(real+imag);bin4+=2;\
  bout1[12]=real;bout2[12]=imag;bout3[12]=sum;\
  real=COPY_EXTRACT_mr(bin4,alpha);imag=COPY_EXTRACT_mi(bin4,alpha);sum=-(real+imag);bin4+=2;\
  bout1[13]=real;bout2[13]=imag;bout3[13]=sum;\
}
#define COPY_BR_4x1 {\
  real=COPY_EXTRACT_mR(bin1,alpha);imag=COPY_EXTRACT_mI(bin1,alpha);sum=-(real+imag);bin1+=2;\
  bout1[0]=real;bout2[0]=imag;bout3[0]=sum;\
  real=COPY_EXTRACT_mR(bin2,alpha);imag=COPY_EXTRACT_mI(bin2,alpha);sum=-(real+imag);bin2+=2;\
  bout1[4]=real;bout2[4]=imag;bout3[4]=sum;\
  real=COPY_EXTRACT_mR(bin3,alpha);imag=COPY_EXTRACT_mI(bin3,alpha);sum=-(real+imag);bin3+=2;\
  bout1[8]=real;bout2[8]=imag;bout3[8]=sum;\
  real=COPY_EXTRACT_mR(bin4,alpha);imag=COPY_EXTRACT_mI(bin4,alpha);sum=-(real+imag);bin4+=2;\
  bout1[12]=real;bout2[12]=imag;bout3[12]=sum;\
}
#define COPY_BH_4x1 {\
  real=COPY_EXTRACT_mr(bin1,alpha);imag=COPY_EXTRACT_mi(bin1,alpha);sum=-(real+imag);bin1+=2;\
  bout1[0]=real;bout2[0]=imag;bout3[0]=sum;\
  real=COPY_EXTRACT_mr(bin2,alpha);imag=COPY_EXTRACT_mi(bin2,alpha);sum=-(real+imag);bin2+=2;\
  bout1[4]=real;bout2[4]=imag;bout3[4]=sum;\
  real=COPY_EXTRACT_mr(bin3,alpha);imag=COPY_EXTRACT_mi(bin3,alpha);sum=-(real+imag);bin3+=2;\
  bout1[8]=real;bout2[8]=imag;bout3[8]=sum;\
  real=COPY_EXTRACT_mr(bin4,alpha);imag=COPY_EXTRACT_mi(bin4,alpha);sum=-(real+imag);bin4+=2;\
  bout1[12]=real;bout2[12]=imag;bout3[12]=sum;\
}
static void load_reg_b_r(FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,FLOAT * __restrict__ alpha){
  FLOAT *bin1,*bin2,*bin3,*bin4,*bout1,*bout2,*bout3;int bcol,brow;FLOAT real,imag,sum;
  bin1=bstartpos;bin2=bin1+2*ldb;bin3=bin2+2*ldb;bin4=bin3+2*ldb;int bshift=2*(4*ldb-BlkDimN);
  for(brow=0;brow<(BlkDimK/4);brow+=4){
    bout1=bblk+brow*4;bout2=bout1+BlkDimN*BlkDimK;bout3=bout2+BlkDimN*BlkDimK;
    for(bcol=0;bcol<(BlkDimN/4);bcol++) COPY_BR_4x4
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
  for(;brow<2*(BlkDimK/4);brow+=4){
    bout1=bblk+brow*4+((BlkDimN/4)-1)*4*BlkDimK+3;bout2=bout1+BlkDimN*BlkDimK;bout3=bout2+BlkDimN*BlkDimK;
    COPY_BR_4x1
    bout1=bblk+brow*4;bout2=bout1+BlkDimN*BlkDimK;bout3=bout2+BlkDimN*BlkDimK;
    for(bcol=1;bcol<(BlkDimN/4);bcol++) COPY_BR_4x4
    COPY_BR_4x3
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
  for(;brow<3*(BlkDimK/4);brow+=4){
    bout1=bblk+brow*4+((BlkDimN/4)-1)*4*BlkDimK+2;bout2=bout1+BlkDimN*BlkDimK;bout3=bout2+BlkDimN*BlkDimK;
    COPY_BR_4x2
    bout1=bblk+brow*4;bout2=bout1+BlkDimN*BlkDimK;bout3=bout2+BlkDimN*BlkDimK;
    for(bcol=1;bcol<(BlkDimN/4);bcol++) COPY_BR_4x4
    COPY_BR_4x2
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
  for(;brow<4*(BlkDimK/4);brow+=4){
    bout1=bblk+brow*4+((BlkDimN/4)-1)*4*BlkDimK+1;bout2=bout1+BlkDimN*BlkDimK;bout3=bout2+BlkDimN*BlkDimK;
    COPY_BR_4x3
    bout1=bblk+brow*4;bout2=bout1+BlkDimN*BlkDimK;bout3=bout2+BlkDimN*BlkDimK;
    for(bcol=1;bcol<(BlkDimN/4);bcol++) COPY_BR_4x4
    COPY_BR_4x1
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
}
static void load_reg_b_h(FLOAT * __restrict__ bstartpos,FLOAT * __restrict__ bblk,int ldb,FLOAT * __restrict__ alpha){
  FLOAT *bin1,*bin2,*bin3,*bin4,*bout1,*bout2,*bout3;int bcol,brow;FLOAT real,imag,sum;
  bin1=bstartpos;bin2=bin1+2*ldb;bin3=bin2+2*ldb;bin4=bin3+2*ldb;int bshift=2*(4*ldb-BlkDimN);
  for(brow=0;brow<(BlkDimK/4);brow+=4){
    bout1=bblk+brow*4;bout2=bout1+BlkDimN*BlkDimK;bout3=bout2+BlkDimN*BlkDimK;
    for(bcol=0;bcol<(BlkDimN/4);bcol++) COPY_BH_4x4
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
  for(;brow<2*(BlkDimK/4);brow+=4){
    bout1=bblk+brow*4+((BlkDimN/4)-1)*4*BlkDimK+3;bout2=bout1+BlkDimN*BlkDimK;bout3=bout2+BlkDimN*BlkDimK;
    COPY_BH_4x1
    bout1=bblk+brow*4;bout2=bout1+BlkDimN*BlkDimK;bout3=bout2+BlkDimN*BlkDimK;
    for(bcol=1;bcol<(BlkDimN/4);bcol++) COPY_BH_4x4
    COPY_BH_4x3
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
  for(;brow<3*(BlkDimK/4);brow+=4){
    bout1=bblk+brow*4+((BlkDimN/4)-1)*4*BlkDimK+2;bout2=bout1+BlkDimN*BlkDimK;bout3=bout2+BlkDimN*BlkDimK;
    COPY_BH_4x2
    bout1=bblk+brow*4;bout2=bout1+BlkDimN*BlkDimK;bout3=bout2+BlkDimN*BlkDimK;
    for(bcol=1;bcol<(BlkDimN/4);bcol++) COPY_BH_4x4
    COPY_BH_4x2
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
  for(;brow<4*(BlkDimK/4);brow+=4){
    bout1=bblk+brow*4+((BlkDimN/4)-1)*4*BlkDimK+1;bout2=bout1+BlkDimN*BlkDimK;bout3=bout2+BlkDimN*BlkDimK;
    COPY_BH_4x3
    bout1=bblk+brow*4;bout2=bout1+BlkDimN*BlkDimK;bout3=bout2+BlkDimN*BlkDimK;
    for(bcol=1;bcol<(BlkDimN/4);bcol++) COPY_BH_4x4
    COPY_BH_4x1
    bin1+=bshift;bin2+=bshift;bin3+=bshift;bin4+=bshift;
  }
}
static void cmultbeta(FLOAT * __restrict__ c,int ldc,int m,int n,FLOAT * __restrict__ beta){
  int i,j;FLOAT *C0,*C;FLOAT real,imag;
  if(beta[0]==0.0 && beta[1]==0.0) return;
  C0=c;
  for(i=0;i<n;i++){
    C=C0;
    for(j=0;j<m;j++){
      real=*C;imag=*(C+1);
      *C=real*beta[0]-imag*beta[1];
      *(C+1)=real*beta[1]+imag*beta[0];
      C+=2;
    }
    C0+=ldc*2;
  }
}
