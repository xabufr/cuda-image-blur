#include "image.h"
#include "common/cpu_anim.h"
#include <iostream>
#include <time.h>

bool add = true;

void cleanup( DataBlock *d ) {
	cudaFree( d->dev_bitmap );
	cudaFree( d->dev_output);
}

__device__ int getPixelIndex(int x, int y, int width) {
	int offset = y * width + x;
	return offset * 4;
}
__global__ void kernel(unsigned char *pixels, unsigned char *output, int width, int height,
		int radius) {
	int x, y;
	x = blockIdx.x * blockDim.x + threadIdx.x;
	y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height) {
		unsigned int averageColor[3] = {0, 0, 0};
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

void generateFrame(DataBlock *d, int ticks)
{
	if ((ticks % 50) == 0)
		add = !add;

	if ((ticks % 5) == 0)
		d->radius = (add)?d->radius+1:d->radius-1;

	std::size_t pixelsSize = sizeof(unsigned char) * d->image.width()
				* d->image.height() * 4;
	cudaMemcpy(d->dev_bitmap, d->image.pixels(), pixelsSize, cudaMemcpyHostToDevice);

	kernel<<<dim3((d->image.width() / 16) + 1, (d->image.height() / 16) + 1),
			dim3(17, 17)>>>(d->dev_bitmap, d->dev_output,
					d->image.width(), d->image.height(), d->radius);

	cudaMemcpy(d->bitmap->get_ptr(), d->dev_output, pixelsSize,
			cudaMemcpyDeviceToHost);
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

	bitmap.anim_and_exit( (void (*)(void*,int))generateFrame, (void (*)(void*))cleanup );

	return 0;
}
