#include "image.h"
#include <SDL/SDL_image.h>

Image::Image() :
		m_pixels(0), m_width(0), m_height(0) {
}

Image::~Image() {
	unload();
}

SDL_Surface* Image__loadAndConvertFormat(const std::string &file) {
	SDL_Surface *image = IMG_Load(file.c_str());
	SDL_PixelFormat newFormat = *image->format;
	newFormat.BitsPerPixel = 32;
	newFormat.BytesPerPixel = 4;
	Uint32 rmask, gmask, bmask, amask;

#if SDL_BYTEORDER == SDL_BIG_ENDIAN

	rmask = 0xff000000;
	gmask = 0x00ff0000;
	bmask = 0x0000ff00;
	amask = 0x000000ff;
#else

	rmask = 0x000000ff;
	gmask = 0x0000ff00;
	bmask = 0x00ff0000;
	amask = 0xff000000;
#endif

	newFormat.Rmask = rmask;
	newFormat.Gmask = gmask;
	newFormat.Bmask = bmask;
	newFormat.Amask = amask;

	SDL_Surface *converted = SDL_ConvertSurface(image, &newFormat,
			SDL_SWSURFACE);
	SDL_FreeSurface(image);

	return converted;
}

void Image__invertPixelsAndCopyTo(const SDL_Surface *src, unsigned char *dest) {
	unsigned char* data = static_cast<unsigned char*>(src->pixels);
	std::size_t line_size_in_byte = src->pitch;

	for (int current_line = 0; current_line < src->h; ++current_line) {
		memcpy(&dest[current_line * line_size_in_byte],
				&data[(src->h - 1 - current_line) * line_size_in_byte],
				line_size_in_byte);
	}
}

void Image::loadFromFile(const std::string& file) {
	SDL_Surface *image = Image__loadAndConvertFormat(file);
	m_width = image->w;
	m_height = image->h;

	m_pixels = new unsigned char[m_width * m_height * 4];

	unsigned char* data = static_cast<unsigned char*>(image->pixels);

	Image__invertPixelsAndCopyTo(image, m_pixels);

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
