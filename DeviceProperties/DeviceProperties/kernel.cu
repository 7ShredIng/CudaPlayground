
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <common.h>

#include <stdio.h>
int main(void)
{
	cudaDeviceProp DeviceProperties;
	int count;

	HANDLE_ERROR(cudaGetDeviceCount(&count));
	for(int i=0; i<count; i++)
	{
		HANDLE_ERROR(cudaGetDeviceProperties(&DeviceProperties, i));
		printf("Name: %\ns", DeviceProperties.name);
		printf("Compute capability: %d - %d\n", DeviceProperties.major, DeviceProperties.minor);
		//printf("Clock rate: %s\n", DeviceProperties.clockRate);
		printf("Device copy overlap: ");
		if(DeviceProperties.deviceOverlap)
		{
			printf("Enabled\n");
		}
		else
		{
			printf("Disabled\n");
		}
		printf("Kernel execution timeout: ");
		if(DeviceProperties.kernelExecTimeoutEnabled)
		{
			printf("Enabled\n");
		}
		else
		{
			printf("Disabled\n");
		}
		printf("Total global mem: %ld\n", DeviceProperties.totalGlobalMem);
		printf("Total constant mem: %ld\n", DeviceProperties.totalConstMem);
		printf("Max mem pitch: %ld\n", DeviceProperties.memPitch);
		printf("Texture Alignment: %ld\n", DeviceProperties.textureAlignment);
		printf("Multiprocessor count: %d\n", DeviceProperties.multiProcessorCount);
		printf("Shared mem per mp: %ld\n", DeviceProperties.sharedMemPerBlock);
		printf("Registers per mp: %d\n", DeviceProperties.regsPerBlock);
		printf("Threads in warp: %d\n", DeviceProperties.warpSize);
		printf("Max threads per block: %d\n", DeviceProperties.maxThreadsPerBlock);
		printf("Max thread dimensions: %d %d %d\n", DeviceProperties.maxThreadsDim[0], DeviceProperties.maxThreadsDim[1], DeviceProperties.maxThreadsDim[2]);
		printf("Max grid dimensions: %d %d %d\n", DeviceProperties.maxGridSize[0], DeviceProperties.maxGridSize[1], DeviceProperties.maxGridSize[2]);
	}

	memset( &DeviceProperties, 0, sizeof(cudaDeviceProp));
	DeviceProperties.major = 1;
	DeviceProperties.minor = 3;
	int device;
	HANDLE_ERROR(cudaChooseDevice(&device, &DeviceProperties));
	printf("ID of device closest to revision 1.3: %d\n ", device);
	HANDLE_ERROR(cudaSetDevice(device));
}