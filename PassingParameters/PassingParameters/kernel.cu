#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include "device_launch_parameters.h"
#include <stdlib.h>
#include "..\..\common.h"

__global__ void add(const int value1, const int value2, int *result)
{
	*result = value1 + value2;
}

int main(void)
{
	int result;
	int *dev_result;
	HANDLE_ERROR(cudaMalloc((void**)&dev_result, sizeof(int)));

	add<<<1,1>>>(2, 7, dev_result);
	HANDLE_ERROR( cudaMemcpy(&result, dev_result, sizeof(int), cudaMemcpyDeviceToHost));

	printf("result 2 + 7 = %d \n", result);
	HANDLE_ERROR(cudaFree(dev_result));

	return 0;
}
