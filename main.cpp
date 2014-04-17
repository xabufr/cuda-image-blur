#include "image.h"
#include "common/cpu_bitmap.h"

int main()
{
	Image image;
	image.loadFromFile("madame.jpg");
	CPUBitmap bitmap(image.width(), image.height());
	image.copyPixels(bitmap.get_ptr());

	bitmap.display_and_exit();


	return 0;
}
