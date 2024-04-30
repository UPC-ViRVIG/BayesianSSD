#include <iostream>
#include "PointCloud.h"
#include "Quadtree.h"
#include "SmoothSurfaceReconstruction.h"
#include "ScalarField.h"
#include "EigenSquareSolver.h"
#include "Image.h"


int main(int argc, char *argv[])
{
	PointCloud cloud;
	if(argc != 3 && argc != 4)
	{
		cout << "Wrong arguments!" << endl << endl;
		cout << "\t" << argv[0] << " <Input cloud> <Output image> [resolution]" << endl;
		return -1;
	}
	
	if(!cloud.readFromFile(argv[1]))
	{
		cout << "Input cloud could not be read!" << endl;
		return -1;
	}

	
	std::unique_ptr<BilinearQuadtree> scalarField = 
		SmoothSurfaceReconstruction::computeWithQuad2D<EigenSquareSolver>(cloud, {4, 1.0f, 1.0f, 1.0f});		
	
	Quadtree& qtree = scalarField->getQuadtree();

	const std::array<glm::vec3, 7> palette = {
		glm::vec3(0.0f, 0.0f, 1.0f), 
		glm::vec3(0.0f, 0.5f, 1.0f), 
		glm::vec3(0.0f, 1.0f, 1.0f), 
		glm::vec3(1.0f, 1.0f, 1.0f), 
		glm::vec3(1.0f, 1.0f, 0.0f), 
		glm::vec3(1.0f, 0.5f, 0.0f), 
		glm::vec3(1.0f, 0.0f, 0.0f)
	};

	Image image;
	image.init(1024, 1024);
	const float maxValue = scalarField->getMaxAbsValue();
	for(uint32_t i=0; i < image.width(); i++)
	{
		for(uint32_t j=0; j < image.height(); j++)
		{
			glm::vec2 pos = glm::vec2((static_cast<float>(i)+0.5f) / static_cast<float>(image.width()), 
									  (static_cast<float>(j)+0.5f) / static_cast<float>(image.height()));

			// Field color
			float val = scalarField->eval(pos);
			val = glm::clamp(val / maxValue, -1.0f, 0.999999f);
			val = static_cast<float>(palette.size() - 1) * 0.5f * (val + 1.0f);
			uint32_t cid = glm::floor(val);
			glm::vec3 bgColor = glm::mix(palette[cid], palette[cid+1], glm::fract(val));

			// Grid color
			std::optional<Quadtree::Node> node;
			qtree.getNode(pos, node);
			if(!node) continue;
			glm::vec2 invSize = 1.0f / (node->maxCoord - node->minCoord);
			glm::vec2 np = (pos - node->minCoord) * invSize;
			np = glm::abs(2.0f * (np - glm::vec2(0.5f)));
			float gw = glm::smoothstep(1.0f - 0.005f * invSize.x, 1.0f - 0.003f * invSize.x, glm::max(np.x, np.y));
			image(i, j) = glm::mix(bgColor, glm::vec3(0.0f), gw);
		}
	}

	for (glm::vec2 point : cloud.getPoints())
	{
		image.drawFilledCircle(static_cast<uint32_t>(image.width() * point.x), static_cast<uint32_t>(image.height() * point.y), 4, glm::vec3(0.35f, 0.35f, 0.35f));
	}

	image.savePNG("quadtree.png"); 


	/*
	int resolution;
	
	if(argc == 4)
		resolution = atoi(argv[3]);
	else
		//resolution = 33;
		resolution = 65;
		//resolution = 129;
		//resolution = 257;
	
	qtree.compute(cloud, 6, field);
	Image qtreeImg;
	qtreeImg.init(2048, 2048);
	qtree.draw(qtreeImg);
	qtreeImg.savePNG("quadtree.png");

	img = field.toImage(16.f * resolution / 128.f, 0.0f);
	//img = field.toImage(1.0f, 0.0f);
	if(!img->savePNG(argv[2]))
	{
		cout << "Could not save file!" << endl;
		delete img;
		return -1;
	}
	delete img;

	*/

	return 0;
}


