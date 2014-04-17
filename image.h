#ifndef H_IMAGE
#define H_IMAGE

#include <string>

class Image
{
private:
	unsigned char *m_pixels;
	unsigned int m_width, m_height;
public:
	Image();
	~Image();

	void loadFromFile(const std::string &file);
	void unload();
	bool isLoaded();
	void copyPixels(unsigned char* dst) const;

	unsigned char *pixels();
	unsigned int width() const;
	unsigned int height() const;
};

#endif
