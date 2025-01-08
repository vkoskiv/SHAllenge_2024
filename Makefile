MAKEFLAGS += --no-builtin-rules

nickminer: nickminer.c
	gcc -o nickminer nickminer.c -lssl -lcrypto

nickminer_cuda: nickminer.cu
	nvcc -o nickminer_cuda nickminer.cu -lineinfo -g -lcurl
