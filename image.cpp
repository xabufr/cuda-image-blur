#include "image.h"
#include <CImg.h>
#include <SDL/SDL_image.h>

Image::Image() :
		m_pixels(0), m_width(0), m_height(0) {
}

Image::~Image() {
	unload();
}

#include <iostream>
void Image::loadFromFile(const std::string& file) {
	SDL_Surface *image = IMG_Load(file.c_str());
	m_width = image->w;
	m_height = image->h;

	m_pixels = new unsigned char[m_width * m_height * 4];
	unsigned char bitsPerPixel = image->format->BitsPerPixel / 8;
    unsigned char* data = static_cast<unsigned char*>(image->pixels);
    std::cout << (int) bitsPerPixel << std::endl;
	for (std::size_t i = 0; i < m_width * m_height * bitsPerPixel; i +=
			bitsPerPixel) {
		m_pixels[i + 0] = data[i + 0];
		m_pixels[i + 1] = data[i + 1];
		m_pixels[i + 2] = data[i + 2];
		m_pixels[i + 3] = 255;
	}

	SDL_FreeSurface(image);
}

void Image::unload() {
	if (isLoaded()) {
		delete[] m_pixels;
	}
}

bool Image::isLoaded() {
	return m_pixels != 0;
}

unsigned char* Image::pixels() {
	return m_pixels;
}

unsigned int Image::width() const {
	return m_width;
}

void Image::copyPixels(unsigned char* dst) const {
	memcpy(dst, m_pixels, m_width * m_height * 4);
}

unsigned int Image::height() const {
	return m_height;
}
