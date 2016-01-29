#include "common.h"
#include <Windows.h>    
#include <conio.h>

#define N 5000

typedef unsigned long long timestamp_t;
double getRealTime()
{
	FILETIME tm;
	ULONGLONG t;
	GetSystemTimeAsFileTime(&tm);
	t = ((ULONGLONG)tm.dwHighDateTime << 32) | (ULONGLONG)tm.dwLowDateTime;
	return (double)t / 10000000.0;
}


__global__ void add(int *a, int *b, int *c)
{
	//int tid = blockIdx.x; //in N blocks
	int tid = threadIdx.x; //in N threads
	if(tid<N)
	{
		c[tid] = a[tid] + b[tid];
	}
}

int main(void)
{
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;
	double startTime, endTime;

	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N*sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N*sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N*sizeof(int)));

	for(int i=0; i<N; i++)
	{
		a[i] = -i;
		b[i] = i * i;
	}
	startTime = getRealTime();

	HANDLE_ERROR(cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice));

	add<<<1,N>>>(dev_a, dev_b, dev_c); //in N threads
	//add<<<N,1>>>(dev_a, dev_b, dev_c); //in N blocks

	HANDLE_ERROR(cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost));

	endTime = getRealTime();

	for(int i=0; i<N; i++)
	{
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}
	printf("Time used: %0.20lf\n", (endTime - startTime));

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}