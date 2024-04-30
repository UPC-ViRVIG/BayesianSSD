#include <cmath>
#include <glm/glm.hpp>
#include "ScalarField.h"


glm::vec3 palette[7] = {
	glm::vec3(0.0f, 0.0f, 1.0f), 
	glm::vec3(0.0f, 0.5f, 1.0f), 
	glm::vec3(0.0f, 1.0f, 1.0f), 
	glm::vec3(1.0f, 1.0f, 1.0f), 
	glm::vec3(1.0f, 1.0f, 0.0f), 
	glm::vec3(1.0f, 0.5f, 0.0f), 
	glm::vec3(1.0f, 0.0f, 0.0f)
};


void ScalarField::init(unsigned int width, unsigned int height)
{
	w = width;
	h = height;
	values.resize(w*h);
}
	
unsigned int ScalarField::width() const
{
	return w;
}
	
unsigned int ScalarField::height() const
{
	return h;
}
	
float &ScalarField::operator()(unsigned int xIndex, unsigned int yIndex)
{
	return values[yIndex * w + xIndex];
}
	
const float &ScalarField::operator()(unsigned int xIndex, unsigned int yIndex) const
{
	return values[yIndex * w + xIndex];
}

float ScalarField::sampleAtLocation(glm::vec2 &pos) const
{
	glm::ivec2 cell;
	
	cell.x = int(floor((w - 1) * pos.x));
	cell.x = glm::max(0, glm::min(cell.x, int(w-2)));
	cell.y = int(floor((h - 1) * pos.y));
	cell.y = glm::max(0, glm::min(cell.y, int(h-2)));
	
	glm::vec2 inCellPos;
	
	inCellPos.x = (w - 1) * pos.x - cell.x;
	inCellPos.y = (h - 1) * pos.y - cell.y;
	
	float value = (1.f - inCellPos.x) * (1.f - inCellPos.y) * values[cell.y * w + cell.x];
	value += inCellPos.x * (1.f - inCellPos.y) * values[cell.y * w + cell.x + 1];
	value += (1.f - inCellPos.x) * inCellPos.y * values[(cell.y + 1) * w + cell.x];
	value += inCellPos.x * inCellPos.y * values[(cell.y + 1) * w + cell.x + 1];
	
	return value;
}

void ScalarField::scale(int width, int height, ScalarField &scaled) const
{
	glm::vec2 pos;

	scaled.w = width;
	scaled.h = height;
	scaled.values.resize(width * height);
	for(unsigned int y=0; y<height; y++)
		for(unsigned int x=0; x<width; x++)
		{
			pos.x = float(x) / width;
			pos.y = float(y) / height;
			scaled.values[y * width + x] = sampleAtLocation(pos);
		}
}

void ScalarField::scaleBy2(ScalarField &scaled) const
{
	scaled.w = 2 * w - 1;
	scaled.h = 2 * h - 1;
	scaled.values.resize(scaled.w * scaled.h);
	for(unsigned int y=0; y<scaled.h; y++)
		for(unsigned int x=0; x<scaled.w; x++)
		{
			if(((x & 1) == 0) && ((y & 1) == 0))
				scaled.values[y * scaled.w + x] = values[(y >> 1) * w + (x >> 1)];
			else if(((x & 1) == 1) && ((y & 1) == 0))
				scaled.values[y * scaled.w + x] = (values[(y >> 1) * w + (x >> 1)] + 
				                                   values[(y >> 1) * w + (x >> 1) + 1]) / 2.f;
			else if(((x & 1) == 0) && ((y & 1) == 1))
				scaled.values[y * scaled.w + x] = (values[(y >> 1) * w + (x >> 1)] + 
				                                   values[((y >> 1) + 1) * w + (x >> 1)]) / 2.f;
			else if(((x & 1) == 1) && ((y & 1) == 1))
				scaled.values[y * scaled.w + x] = (values[(y >> 1) * w + (x >> 1)] + 
				                                   values[((y >> 1) + 1) * w + (x >> 1)] + 
				                                   values[(y >> 1) * w + (x >> 1) + 1] + 
				                                   values[((y >> 1) + 1) * w + (x >> 1) + 1]) / 4.f;
		}
}

Image *ScalarField::toImage(float maxValue, float zeroThickness) const
{
	Image *img;
	float scaledValue, lambda;
	unsigned int colorIndex;
	
	img = new Image();
	img->init(w, h);
	for(unsigned int y=0; y<h; y++)
		for(unsigned int x=0; x<w; x++)
		{
			//scaledValue = fabs(values[y * w + x]) / maxValue;
			scaledValue = values[y * w + x] / maxValue;
			scaledValue = (scaledValue + 1.0f) / 2.0f;
			scaledValue = max(0.0f, min(scaledValue, 1.0f));
			colorIndex = min(int(6.0f * scaledValue), 6);
			lambda = 6.0f * (scaledValue - colorIndex / 6.0f);
			(*img)(x, y) = glm::mix(palette[colorIndex], palette[colorIndex+1], lambda);
			if(zeroThickness != 0.0f && fabs(values[y * w + x]) < zeroThickness)
				(*img)(x, y) = glm::vec3(0.0f);
			/*
			if(values[y * w + x] > 0.0f)
				(*img)(x, y) = glm::vec3(1.0f, 0.0f, 0.0f);
			else
				(*img)(x, y) = glm::vec3(0.0f, 0.0f, 1.0f);
			*/
		}
	
	return img;
}


