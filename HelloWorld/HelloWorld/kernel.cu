#include <stdio.h>

__global__ void kernel(void){}

int main(void)
{
	kernel<<<1,1>>>();
	printf("hello world");
	return 0;
}