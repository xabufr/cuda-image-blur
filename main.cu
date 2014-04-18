#include "image.h"
#include "common/cpu_bitmap.h"
#include <iostream>

__device__ int getPixelIndex(int x, int y, int width) {
	int offset = y * width + x;
	return offset * 4;
}
__global__ void kernel(unsigned char *pixels, int width, int height,
		int radius) {
	int x, y;
	x = blockIdx.x * blockDim.x + threadIdx.x;
	y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height) {
		unsigned int averageColor[3];
		int pixelOffset = getPixelIndex(x, y, width);
		averageColor[0] = pixels[pixelOffset];
		averageColor[1] = pixels[pixelOffset + 1];
		averageColor[2] = pixels[pixelOffset + 2];

		int count = 1;

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
		__syncthreads();
		pixels[pixelOffset + 0] = averageColor[0];
		pixels[pixelOffset + 1] = averageColor[1];
		pixels[pixelOffset + 2] = averageColor[2];
	} else {
		__syncthreads();
	}
}

int main() {
	Image image;
	image.loadFromFile("madame.jpg");
	CPUBitmap bitmap(image.width(), image.height());
	unsigned char *cudaPixels;

	std::size_t pixelsSize = sizeof(unsigned char) * image.width()
			* image.height() * 4;

	cudaMalloc(&cudaPixels, pixelsSize);
	cudaMemcpy(cudaPixels, image.pixels(), pixelsSize, cudaMemcpyHostToDevice);
	std::cout << image.width() / 16 << ":"<< image.height() / 16 << std::endl;

	kernel<<<dim3((image.width() / 16) + 1, (image.height() / 16) + 1),
			dim3(17, 17)>>>(cudaPixels,
			image.width(), image.height(), 0);

	cudaMemcpy(bitmap.get_ptr(), cudaPixels, pixelsSize,
			cudaMemcpyDeviceToHost);
	cudaFree(cudaPixels);

	bitmap.display_and_exit();

	return 0;
}
