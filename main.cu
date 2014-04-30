#include "image.h"
#include "common/cpu_anim.h"
#include <iostream>
#include <time.h>

#define DIM 16
#define MAX_DIM 256

bool add = true;

void cleanup(DataBlock *d) {
	cudaFree(d->dev_bitmap);
	cudaFree(d->dev_output);
}

__device__ int getPixelIndex(int x, int y, int width) {
	int offset = y * width + x;
	return offset * 4;
}
__global__ void kernel(unsigned char *pixels, unsigned char *output, int width,
		int height, int radius) {
	int x, y;
	x = blockIdx.x * blockDim.x + threadIdx.x;
	y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height) {
		unsigned int averageColor[3] = { 0, 0, 0 };
		int pixelOffset = getPixelIndex(x, y, width);

		int count = 0;

		int currentOffset;
		for (int d_x = -radius; d_x <= radius; ++d_x) {
			int currentX = x + d_x;
			if (currentX < width && currentX >= 0) {
				for (int d_y = -radius; d_y <= radius; ++d_y) {
					int currentY = y + d_y;
					if (currentY < height && currentY >= 0) {
						++count;
						currentOffset = getPixelIndex(currentX, currentY,
								width);
						averageColor[0] += pixels[currentOffset];
						averageColor[1] += pixels[currentOffset + 1];
						averageColor[2] += pixels[currentOffset + 2];
					}
				}
			}
		}
		averageColor[0] /= count;
		averageColor[1] /= count;
		averageColor[2] /= count;

		output[pixelOffset + 0] = averageColor[0];
		output[pixelOffset + 1] = averageColor[1];
		output[pixelOffset + 2] = averageColor[2];
		output[pixelOffset + 3] = 255;
	}
}

__global__ void kernelPixellization(unsigned char *pixels, unsigned char *ouput,
		int width, int height, int radius) {
	__shared__ int moyenne[MAX_DIM][3];
	int x = blockIdx.x * radius;
	int y = blockIdx.y * radius + threadIdx.x;

	__shared__ int finalMoyenne[3];
	__shared__ int count[MAX_DIM];

	for (int i(0); i < 3; ++i) {
		moyenne[threadIdx.x][i] = 0;
	}
	count[threadIdx.x] = 0;
	if (y < height) {
		for (int d_x = 0; d_x < radius; ++d_x) {
			if (d_x + x < width) {
				int offset = getPixelIndex(x + d_x, y, width);
				++count[threadIdx.x];
				for (int i = 0; i < 3; ++i) {
					moyenne[threadIdx.x][i] += pixels[offset + i];
				}
			}
		}
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		for (int i = 0; i < 3; ++i) {
			finalMoyenne[i] = 0;
		}
		int finalCount = 0;
		for (int i = 0; i < radius; ++i) {
			finalCount += count[i];
			for (int j = 0; j < 3; ++j) {
				finalMoyenne[j] += moyenne[i][j];
			}
		}
		for (int i = 0; i < 3; ++i) {
			finalMoyenne[i] /= finalCount;
		}
	}
	__syncthreads();

	if (y < height) {
		for (int d_x = 0; d_x < radius; ++d_x) {
			if (d_x + x < width) {
				int offset = getPixelIndex(x + d_x, y, width);
				for (int i(0); i < 3; ++i) {
					ouput[offset + i] = finalMoyenne[i];
				}
			}
		}
	}
}

void generateFramePixellisation(DataBlock *d, int ticks) {
	if ((ticks % 128) == 0)
		add = !add;

	d->radius += (add ? 1 : -1);
	if (d->radius <= 0)
		d->radius = 1;
	std::cout << d->radius << std::endl;
	cudaEventRecord(d->start, 0);
	std::size_t pixelsSize = sizeof(unsigned char) * d->image.width()
			* d->image.height() * 4;
	cudaMemcpy(d->dev_bitmap, d->image.pixels(), pixelsSize,
			cudaMemcpyHostToDevice);

	kernelPixellization<<<
			dim3((d->image.width() / d->radius) + 1,
					(d->image.height() / d->radius) + 1), d->radius>>>(
			d->dev_bitmap, d->dev_output, d->image.width(), d->image.height(),
			d->radius);
	cudaEventRecord(d->stop, 0);
	cudaEventSynchronize(d->stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, d->start, d->stop);
	std::cout << "Elapsed time: " << elapsedTime << std::endl;

	cudaMemcpy(d->bitmap->get_ptr(), d->dev_output, pixelsSize,
			cudaMemcpyDeviceToHost);
}
void generateFrame(DataBlock *d, int ticks) {
	if ((ticks % 10) == 0)
		add = !add;

//if ((ticks % 5) == 0) {
	d->radius = (add) ? d->radius + 1 : d->radius - 1;

	cudaEventRecord(d->start, 0);
	std::size_t pixelsSize = sizeof(unsigned char) * d->image.width()
			* d->image.height() * 4;
	cudaMemcpy(d->dev_bitmap, d->image.pixels(), pixelsSize,
			cudaMemcpyHostToDevice);

	kernel<<<dim3((d->image.width() / 15) + 1, (d->image.height() / 15) + 1),
			dim3(16, 16)>>>(d->dev_bitmap, d->dev_output, d->image.width(),
			d->image.height(), d->radius);
	cudaEventRecord(d->stop, 0);
	cudaEventSynchronize(d->stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, d->start, d->stop);
	std::cout << "Elapsed time: " << elapsedTime << std::endl;

	cudaMemcpy(d->bitmap->get_ptr(), d->dev_output, pixelsSize,
			cudaMemcpyDeviceToHost);
//}
}

int main() {
	DataBlock data;
	data.radius = 1;
	data.image.loadFromFile("madame.jpg");
	CPUAnimBitmap bitmap(data.image.width(), data.image.height(), &data);
	data.bitmap = &bitmap;

	std::size_t pixelsSize = sizeof(unsigned char) * data.image.width()
			* data.image.height() * 4;

	cudaMalloc(&data.dev_output, pixelsSize);
	cudaMalloc(&data.dev_bitmap, pixelsSize);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	bitmap.anim_and_exit((void (*)(void*, int))generateFramePixellisation, (void (*)(void*))cleanup ) ;

	return 0;
}
