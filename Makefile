CC = gcc
CCFLAGS = -fopenmp --shared -fPIC -march=haswell -O2
SRCFILE = src/gemm3m_kernel.S src/gemm3m_driver.c
INCFILE = src/gemm3m_kernel_irreg.c src/gemm3m_copy.c

default: ZGEMM3M.so CGEMM3M.so

ZGEMM3M.so: $(SRCFILE) $(INCFILE)
	$(CC) -DDOUBLE -DBlkDimK=256 -DBlkDimN=128 -DB_PR_ELEM=64 -DA_PR_BYTE=256 $(CCFLAGS) $(SRCFILE) -o $@
  
CGEMM3M.so: $(SRCFILE) $(INCFILE)
	$(CC) -DBlkDimK=256 -DBlkDimN=256 -DB_PR_ELEM=64 -DA_PR_BYTE=256 $(CCFLAGS) $(SRCFILE) -o $@

clean:
	rm -f *GEMM3M.so

