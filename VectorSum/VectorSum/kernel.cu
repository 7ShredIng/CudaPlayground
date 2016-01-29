#include "common.h"


#define N 5000

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
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N*sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N*sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N*sizeof(int)));

	for(int i=0; i<N; i++)
	{
		a[i] = -i;
		b[i] = i * i;
	}
	cudaEventRecord(start, 0);

	HANDLE_ERROR(cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice));

	add<<<1,N>>>(dev_a, dev_b, dev_c); //in N threads
	//add<<<N,1>>>(dev_a, dev_b, dev_c); //in N blocks

	HANDLE_ERROR(cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost));

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	for(int i=0; i<N; i++)
	{
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}
	cudaEventElapsedTime(&time, start, stop);
	printf("Time in kernel: %f ms\n", time);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}