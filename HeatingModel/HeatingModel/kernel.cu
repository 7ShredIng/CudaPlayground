#include "cuda.h"
#include <common.h>
#include <cpu_anim.h>
#include <device_launch_parameters.h>
#include "cuda_runtime.h"
#include "cuda_texture_types.h"
#include "texture_types.h"

#define DIM 1024
#define SPEED 0.25f
#define PI 3.1415926535897932f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f

texture<float, cudaTextureType2D, cudaReadModeElementType> texConstSrc;
texture<float, cudaTextureType2D, cudaReadModeElementType> texIn;
texture<float, cudaTextureType2D, cudaReadModeElementType> texOut;

struct DataBlock
{
	unsigned char* output_bitmap;
	float* dev_inSrc;
	float* dev_outSrc;
	float* dev_constSrc;
	CPUAnimBitmap* bitmap;
	cudaEvent_t start, stop;
	float totalTime;
	float frames;
};

__global__ void copy_const_kernel(float* iptr)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x *gridDim.x;

	float c = tex2D(texConstSrc, x, y);
	if(c != 0)
	{
		iptr[offset] = c;
	}
}

__global__ void blend_kernel(float* dst, bool dstOut)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x *gridDim.x;

	int left = offset - 1;
	int right = offset + 1;
	if(x == 0)
	{
		left++;
	}
	if(x == (DIM - 1))
	{
		 
	int top = offset - DIM;
	int bottom = offset + DIM;
	if(y == 0)
	{
		top += DIM;
	}
	if(y == (DIM - 1))
	{
		bottom -= DIM;
	}

	float t, l, c, r, b;
	if(dstOut)
	{
		t = tex2D(texIn, x, y-1);
		l = tex2D(texIn, x-1, y);
		c = tex2D(texIn, x, y);
		r = tex2D(texIn, x+1, y);
		b = tex2D(texIn, x, y+1);
	}else{
		t = tex2D(texOut, x, y-1);
		l = tex2D(texOut, x-1, y);
		c = tex2D(texOut, x, y);
		r = tex2D(texOut, x+1, y);
		b = tex2D(texOut, x, y+1);
	}
	dst[offset] = c + SPEED * (t + b + r + l - 4*c);
}

void anim_gpu(DataBlock* data, int ticks)
{
	HANDLE_ERROR(cudaEventRecord(data->start, 0));
	dim3 blocks(DIM/16, DIM/16);
	dim3 threads(16,16);
	CPUAnimBitmap* bitmap = data->bitmap;

	volatile bool dstOut = true;
	for(int i = 0; i < 90; i++)
	{
		float *in, *out;
		if(dstOut)
		{
			in = data->dev_inSrc;
			out = data->dev_outSrc;
		}else{
			out = data->dev_inSrc;
			in = data->dev_outSrc;
		}
		copy_const_kernel<<<blocks,threads>>>(in);
		blend_kernel<<<blocks,threads>>>(out, dstOut);
		dstOut = !dstOut;
	}

	float_to_color<<<blocks,threads>>>(data->output_bitmap, data->dev_inSrc);

	HANDLE_ERROR(cudaMemcpy(bitmap->get_ptr(), data->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaEventRecord(data->stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(data->stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, data->start, data->stop));
	data->totalTime += elapsedTime;
	++data->frames;
	printf("Average Time per Frame: %3.1f ms\n", data->totalTime/data->frames);
}

void anim_exit(DataBlock* data)
{
	HANDLE_ERROR(cudaUnbindTexture(texIn));
	HANDLE_ERROR(cudaUnbindTexture(texOut));
	HANDLE_ERROR(cudaUnbindTexture(texConstSrc));

	cudaFree(data->dev_inSrc);
	cudaFree(data->dev_outSrc);
	cudaFree(data->dev_constSrc);

	HANDLE_ERROR(cudaEventDestroy(data->start));
	HANDLE_ERROR(cudaEventDestroy(data->stop));
}

int main(void)
{
	DataBlock data;
	CPUAnimBitmap bitmap(DIM, DIM, &data);
	data.bitmap = &bitmap;
	data.totalTime = 0;
	data.frames = 0;

	HANDLE_ERROR(cudaEventCreate(&data.start));
	HANDLE_ERROR(cudaEventCreate(&data.stop));

	HANDLE_ERROR(cudaMalloc((void**)&data.output_bitmap, bitmap.image_size()));
	HANDLE_ERROR(cudaMalloc((void**)&data.dev_inSrc, bitmap.image_size()));
	HANDLE_ERROR(cudaMalloc((void**)&data.dev_outSrc, bitmap.image_size()));
	HANDLE_ERROR(cudaMalloc((void**)&data.dev_constSrc, bitmap.image_size()));

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>(); 
	HANDLE_ERROR(cudaBindTexture2D(NULL, texConstSrc, data.dev_constSrc, desc, DIM, DIM, sizeof(float)*DIM));
	HANDLE_ERROR(cudaBindTexture2D(NULL, texIn, data.dev_inSrc, desc, DIM, DIM, sizeof(float)*DIM));
	HANDLE_ERROR(cudaBindTexture2D(NULL, texOut, data.dev_outSrc, desc, DIM, DIM, sizeof(float)*DIM));

	float* temp = (float*)malloc(bitmap.image_size());
	for(int i = 0; i < DIM*DIM; i++)
	{
		temp[i] = 0;
		int x = i % DIM;
		int y = i / DIM;
		if((x > 300) && (x < 600) && (y > 310) && (y < 601))
		{
			temp[i] = MAX_TEMP;
		}
	}
	temp[DIM*100+100] = (MAX_TEMP + MIN_TEMP) / 2;
	temp[DIM*700+100] = MIN_TEMP;
	temp[DIM*300+300] = MIN_TEMP;
	temp[DIM*200+700] = MIN_TEMP;
	for(int y = 800; y < 900; y++)
	{
		for(int x = 400; x < 500; x++)
		{
			temp[x+y*DIM] = MIN_TEMP;
		}
	}
	HANDLE_ERROR(cudaMemcpy(data.dev_constSrc, temp, bitmap.image_size(), cudaMemcpyHostToDevice));
	for(int y = 800; y < DIM; y++)
	{
		for(int x = 0; x < 200; x++)
		{
			temp[x+y*DIM] = MAX_TEMP;
		}
	}
	HANDLE_ERROR(cudaMemcpy(data.dev_inSrc, temp, bitmap.image_size(), cudaMemcpyHostToDevice));
	free(temp);
	bitmap.anim_and_exit((void(*)(void*,int))anim_gpu, (void(*)(void*))anim_exit);
}