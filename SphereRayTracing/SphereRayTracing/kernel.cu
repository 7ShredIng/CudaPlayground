#include <common.h>
#include <cuda.h>
#include <math.h>
#include <cpu_bitmap.h>

#define INF 2e10f
#define rnd(x)(x*rand() / RAND_MAX)
#define SPHERES 20
#define DIM 1024

struct Sphere
{
	float r,b,g;
	float radius;
	float x,y,z;
	__device__ float hit(float ox, float oy, float *n)
	{
		float dx = ox - x;
		float dy = oy - y;
		if(dx*dx + dy*dy < radius*radius)
		{
			float dz = sqrtf(radius*radius - dx*dx - dy*dy);
			*n = dz / sqrtf(radius*radius);
			return dz + z;
		}
		return -INF;
	}
};

__device__ __constant__ Sphere sphere[SPHERES];

__global__ void kernel(/*Sphere* sphere, */unsigned char* ptr)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	float ox = (x - DIM/2);
	float oy = (y - DIM/2);
	float r = 0, g = 0, b = 0;
	float maxz = -INF;

	for(int i = 0; i < SPHERES; i++)
	{
		float n, distance = sphere[i].hit(ox, oy, &n);
		if(distance > maxz)
		{
			float fscale = n;
			r = sphere[i].r * fscale;
			g = sphere[i].g * fscale;
			b = sphere[i].b * fscale;
			maxz = distance;
		}
	}

	ptr[offset*4 + 0] = (int)(r*255);
	ptr[offset*4 + 1] = (int)(g*255);
	ptr[offset*4 + 2] = (int)(b*255);
	ptr[offset*4 + 3] = 255;
}

struct DataBlock {
    unsigned char* dev_bitmap;
//    Sphere* sphere;
};

int main(void)
{
	DataBlock data;
	//Sphere* sphere;
	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	CPUBitmap bitmap(DIM, DIM, &data);
	unsigned char* dev_bitmap;

	HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));
	HANDLE_ERROR(cudaMalloc((void**)&sphere, sizeof(Sphere)*SPHERES));

	Sphere *temp_sphere = (Sphere*)malloc(sizeof(Sphere)*SPHERES);
	for(int i = 0; i < SPHERES; i++)
	{
		temp_sphere[i].r = rnd(1.0f);
		temp_sphere[i].g = rnd(1.0f);
		temp_sphere[i].b = rnd(1.0f);
		temp_sphere[i].x = rnd(1000.0f) - 500;
		temp_sphere[i].y = rnd(1000.0f) - 500;
		temp_sphere[i].z = rnd(1000.0f) - 500;
		temp_sphere[i].radius = rnd(100.0f) + 20;
	}
	
	//HANDLE_ERROR(cudaMemcpy(sphere, temp_sphere, sizeof(Sphere)*SPHERES, cudaMemcpyHostToDevice));
	//size_t offset = 0;
	cudaError_t e2 = cudaMemcpyToSymbol(sphere, temp_sphere, sizeof(Sphere)*SPHERES/*, offset, cudaMemcpyHostToDevice*/);
	free(temp_sphere);

	dim3 grids(DIM/16, DIM/16);
	dim3 threads(16,16);
	kernel<<<grids,threads>>>(/*sphere, */dev_bitmap);

	HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));
	
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Elapsed Time for Sphere Ray Tracing: %3.1f ms\n", elapsedTime);
	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));
	
	bitmap.display_and_exit();

	cudaFree(dev_bitmap);
	cudaFree(sphere);
}
