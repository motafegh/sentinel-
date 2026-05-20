#include <stdio.h>
int main() { printf("CUDA version: %d\n", __CUDACC_VER_MAJOR__ * 100 + __CUDACC_VER_MINOR__); return 0; }
