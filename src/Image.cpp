#include <cmath>
#include "Image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


Image::Image()
{
	w = h = 0;
}


void Image::init(unsigned int width, unsigned int height)
{
	w = width;
	h = height;
	pixels.resize(w * h);
}

unsigned int Image::width() const
{
	return w;
}

unsigned int Image::height() const
{
	return h;
}

glm::vec3 &Image::operator()(unsigned int x, unsigned int y)
{
	return pixels[y * w + x];
}

const glm::vec3 &Image::operator()(unsigned int x, unsigned int y) const
{
	return pixels[y * w + x];
}

void Image::fill(const glm::vec3 &color)
{
	for(unsigned int y=0; y<h; y++)
		for(unsigned int x=0; x<w; x++)
			(*this)(x, y) = color;
}

void Image::drawHorizontalLine(unsigned int x0, unsigned int x1, unsigned int y, const glm::vec3 &color)
{
	if(y < 0 || y >= h)
		return;
	for(unsigned int x=glm::max(x0, 0u); x<=glm::min(x1, w-1); x++)
		(*this)(x, y) = color;
}

void Image::drawVerticalLine(unsigned int x, unsigned int y0, unsigned int y1, const glm::vec3 &color)
{
	if(x < 0 || x >= w)
		return;
	for(unsigned int y=glm::max(y0, 0u); y<=glm::min(y1, h-1); y++)
		(*this)(x, y) = color;
}

void Image::drawRectangle(unsigned int x0, unsigned int y0, unsigned int x1, unsigned int y1, const glm::vec3 &color)
{
	drawHorizontalLine(x0, x1, y0, color);
	drawHorizontalLine(x0, x1, y1, color);
	drawVerticalLine(x0, y0, y1, color);
	drawVerticalLine(x1, y0, y1, color);
}

void Image::drawFilledCircle(unsigned int x, unsigned int y, float radius, const glm::vec3 &color)
{
	for(unsigned int py=glm::max(0, int(floor(y-radius))); py<=glm::min(int(ceil(y+radius)), int(h-1)); py++)
		for(unsigned int px=glm::max(0, int(floor(x-radius))); px<=glm::min(int(ceil(x+radius)), int(w-1)); px++)
		{
			if(((px - x) * (px - x) + (py - y) * (py - y)) < radius * radius)
				(*this)(px, py) = color;
		}
}

bool Image::savePNG(const std::string &filename) const
{
	unsigned char *data;
	
	data = new unsigned char[w*h*3];
	for(unsigned int pos=0; pos<w*h; pos++)
	{
		data[3*pos] = (unsigned char)(255.f * pixels[pos].r);
		data[3*pos+1] = (unsigned char)(255.f * pixels[pos].g);
		data[3*pos+2] = (unsigned char)(255.f * pixels[pos].b);
	}
	bool bSuccessful = stbi_write_png(filename.c_str(), w, h, 3, data, 3*w);
	delete [] data;

	return bSuccessful;
}

