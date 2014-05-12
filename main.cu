#include "image.h"
#include "common/cpu_bitmap.h"
#include <iostream>
#include "Sphere.h"
#include <time.h>

#define SPHERE_COUNT 100

__constant__ Sphere spheresKernel[SPHERE_COUNT];

__device__ int getPixelIndex(int x, int y, int width) {
	int offset = y * width + x;
	return offset * 4;
}

__global__ void kernel(unsigned char *outPixels, int width, int height) {
	int x = blockIdx.x * 16 + threadIdx.x;
	int y = blockIdx.y * 16 + threadIdx.y;
	Position pos = Position::fromCoord(x, y, 0);

	if (x < width && y < height) {
		int offset = getPixelIndex(x, y, width);
		Sphere *optimalSphere = NULL;
		float squareDistToCenter;
		float optiDist;
		for (int i(0); i < SPHERE_COUNT; ++i) {
			float currentSquareDistToCenter;
			float dist = spheresKernel[i].hit(pos, currentSquareDistToCenter);
			if (dist > 0) {
				if (optimalSphere == NULL || optiDist >= dist) {
					optimalSphere = &spheresKernel[i];
					optiDist = dist;
					squareDistToCenter = currentSquareDistToCenter;
				}
			}
		}
		if (optimalSphere != NULL) {
			float ombre =
					1
							- (squareDistToCenter
									/ (optimalSphere->radius()
											* optimalSphere->radius()));
			outPixels[offset + 0] = optimalSphere->color().r * ombre;
			outPixels[offset + 1] = optimalSphere->color().g * ombre;
			outPixels[offset + 2] = optimalSphere->color().b * ombre;
			outPixels[offset + 3] = 255;
		} else {
			outPixels[offset + 0] = 0;
			outPixels[offset + 1] = 0;
			outPixels[offset + 2] = 0;
			outPixels[offset + 3] = 255;
		}
	}
}

int main() {
	const int width = 800, height = 600;
	CPUBitmap bitmap(width, height);
	Sphere spheres[SPHERE_COUNT];
	for (int i(0); i < SPHERE_COUNT; ++i) {
		spheres[i] = Sphere::random();
	}

//	spheres[0] = Sphere(Color(255, 0, 0), Position(175, 175, 1000), 100);
//	spheres[1] = Sphere(Color(0, 254, 0), Position(100, 100, 1000), 100);
//	spheres[2] = Sphere(Color(0, 0, 253), Position(180, 65, 970), 50);

	unsigned char *pixels = bitmap.get_ptr();
	for (int i = 0; i < width * height * 4; i += 4) {
		pixels[i + 0] = 128;
		pixels[i + 1] = 128;
		pixels[i + 2] = 128;
		pixels[i + 3] = 255;
	}

	unsigned char *pixelsKernel;
	cudaMalloc(&pixelsKernel, sizeof(unsigned char) * width * height * 4);
	//cudaMalloc(&spheresKernel, sizeof(Sphere) * SPHERE_COUNT);
	cudaMemcpyToSymbol(spheresKernel, spheres, sizeof(Sphere) * SPHERE_COUNT, 0,
			cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	kernel<<<dim3(width / 16 + 1, height / 16 + 1), dim3(16, 16)>>>(
			pixelsKernel, width, height);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	std::cout << elapsedTime << " ms\n";

	cudaMemcpy(pixels, pixelsKernel, sizeof(unsigned char) * width * height * 4,
			cudaMemcpyDeviceToHost);
	cudaFree(pixelsKernel);
	bitmap.display_and_exit();
	return EXIT_SUCCESS;
}
