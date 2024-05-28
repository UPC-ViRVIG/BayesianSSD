#include <iostream>
#include "SmoothSurfaceReconstruction.h"
#include "EigenSquareSolver.h"
#include "Image.h"

template<typename SF>
void drawScalarField(Image& image, SF& scalarField, 
					 std::optional<std::reference_wrapper<PointCloud<2>>> cloud = std::optional<std::reference_wrapper<PointCloud<2>>>(), 
					 bool drawGrid = true)
{
	NodeTree<2>& qtree = scalarField.getNodeTree();

	const std::array<glm::vec3, 7> palette = {
		glm::vec3(0.0f, 0.0f, 1.0f), 
		glm::vec3(0.0f, 0.5f, 1.0f), 
		glm::vec3(0.0f, 1.0f, 1.0f), 
		glm::vec3(1.0f, 1.0f, 1.0f), 
		glm::vec3(1.0f, 1.0f, 0.0f), 
		glm::vec3(1.0f, 0.5f, 0.0f), 
		glm::vec3(1.0f, 0.0f, 0.0f)
	};
	
	const float maxValue = scalarField.getMaxAbsValue();
	for(uint32_t i=0; i < image.width(); i++)
	{
		for(uint32_t j=0; j < image.height(); j++)
		{
			glm::vec2 pos = glm::vec2((static_cast<float>(i)+0.5f) / static_cast<float>(image.width()), 
									  (static_cast<float>(j)+0.5f) / static_cast<float>(image.height()));

			// Field color
			const float pval = scalarField.eval(pos);
			float val = glm::clamp(pval / maxValue, -1.0f, 0.999999f);
			val = static_cast<float>(palette.size() - 1) * 0.5f * (val + 1.0f);
			uint32_t cid = glm::floor(val);
			glm::vec3 bgColor = glm::mix(palette[cid], palette[cid+1], glm::fract(val));

			float aval = glm::abs(pval / maxValue);
			// Isosurface line
			float surfaceColorWeight = glm::clamp(1.0 - glm::pow(aval/0.009f, 12), 0.0, 1.0);

			// Isolines
			float valToLevel = 0.5f - glm::abs(glm::fract(aval / 0.03f) - 0.5);
			float linesColorWeight = glm::clamp(1.0 - glm::pow(valToLevel * 0.03f/0.0015f, 12), 0.0, 1.0);

			// Grid color
			float gridColorWeight = 0.0f;
			if(drawGrid) 
			{
				std::optional<NodeTree<2>::Node> node;
				qtree.getNode(pos, node);
				if(!node) continue;
				glm::vec2 invSize = 1.0f / (node->maxCoord - node->minCoord);
				glm::vec2 np = (pos - node->minCoord) * invSize;
				np = glm::abs(2.0f * (np - glm::vec2(0.5f)));
				gridColorWeight = glm::smoothstep(1.0f - 0.005f * invSize.x, 1.0f - 0.003f * invSize.x, glm::max(np.x, np.y));
			}

			image(i, j) = glm::mix(bgColor, glm::vec3(0.0f), glm::max(glm::max(surfaceColorWeight, linesColorWeight), gridColorWeight));
		}
	}

	if(cloud)
	{
		for (glm::vec2 point : cloud->get().getPoints())
		{
			image.drawFilledCircle(static_cast<uint32_t>(image.width() * point.x), static_cast<uint32_t>(image.height() * point.y), 4, glm::vec3(0.35f, 0.35f, 0.35f));
		}
	}

	// image.drawFilledCircle(static_cast<uint32_t>(image.width() * point.x), static_cast<uint32_t>(image.height() * point.y), 4, glm::vec3(1.0f, 0.0f, 0.0f));
}


template<typename SF>
void drawGradientMaginitudeImage(Image& image, SF& scalarField)
{
	NodeTree<2>& qtree = scalarField.getNodeTree();

	const std::array<glm::vec3, 2> palette = {
		glm::vec3(1.0f, 1.0f, 1.0f),
		glm::vec3(1.0f, 0.0f, 0.0f)
	};

	float max=0.0f;
	float mean=0.0f;
	for(uint32_t i=0; i < image.width(); i++)
	{
		for(uint32_t j=0; j < image.height(); j++)
		{
			glm::vec2 pos = glm::vec2((static_cast<float>(i)+0.5f) / static_cast<float>(image.width()), 
									  (static_cast<float>(j)+0.5f) / static_cast<float>(image.height()));

			const glm::vec2 pval = scalarField.evalGrad(pos);
			const float len = glm::length(pval);
			max = glm::max(max, glm::abs(len-1.0f));
			mean += len;
		}
	}

	mean = mean / static_cast<float>(image.width() * image.height());
	std::cout << "Grad magnitude mean " << mean << std::endl;
	
	const float maxValue = scalarField.getMaxAbsValue();
	for(uint32_t i=0; i < image.width(); i++)
	{
		for(uint32_t j=0; j < image.height(); j++)
		{
			glm::vec2 pos = glm::vec2((static_cast<float>(i)+0.5f) / static_cast<float>(image.width()), 
									  (static_cast<float>(j)+0.5f) / static_cast<float>(image.height()));

			// Field color
			const glm::vec2 pval = scalarField.evalGrad(pos);
			const float len = glm::abs(glm::length(pval)-1.0f);
			float val = glm::clamp(len / max, 0.0f, 0.999999f);
			val = static_cast<float>(palette.size() - 1) * val;
			uint32_t cid = glm::floor(val);
			glm::vec3 bgColor = glm::mix(palette[cid], palette[cid+1], glm::fract(val));

			float sval = scalarField.eval(pos);
			float aval = glm::abs(sval / maxValue);
			// Isosurface line
			float surfaceColorWeight = glm::clamp(1.0 - glm::pow(aval/0.009f, 12), 0.0, 1.0);
			surfaceColorWeight = 0;

			// Isolines
			float valToLevel = 0.5f - glm::abs(glm::fract(aval / 0.03f) - 0.5);
			float linesColorWeight = glm::clamp(1.0 - glm::pow(valToLevel * 0.03f/0.0015f, 12), 0.0, 1.0);
			linesColorWeight = 0;

			image(i, j) = glm::mix(bgColor, glm::vec3(0.0f), glm::max(surfaceColorWeight, linesColorWeight));
		}
	}
}

int main(int argc, char *argv[])
{
	PointCloud<2> cloud;
	if(argc != 3 && argc != 4)
	{
		std::cout << "Wrong arguments!" << std::endl << std::endl;
		std::cout << "\t" << argv[0] << " <Input cloud> <Output image> [resolution]" << std::endl;
		return -1;
	}
	
	if(!cloud.readFromFile(argv[1]))
	{
		std::cout << "Input cloud could not be read!" << std::endl;
		return -1;
	}

	for(glm::vec2& pos : cloud.getPoints())
	{
		pos += glm::vec2(0.01f);
	}

	// for(auto& v : cloud.getPoints())
	// {
	// 	v = 0.2f * (v - NodeTree<2>::vec(0.5f)) + NodeTree<2>::vec(0.5f);
	// }

	const uint32_t maxDepth = 6;
	NodeTree<2>::Config quadConfig = {
		.minCoord = NodeTree<2>::vec(0.0f),
		.maxCoord = NodeTree<2>::vec(1.0f),
		.pointFilterMaxDistance = 0.0f / static_cast<float>(1 << maxDepth),
		//.pointFilterMaxDistance = 0.0f,
		.constraintNeighbourNodes = true,
		.maxDepth = maxDepth
	};

	SmoothSurfaceReconstruction::Config<2> config = {
		.nodeTreeConfig = quadConfig,
		.posWeight = 100.0f, 
        .gradientWeight = 100.0f,
        .smoothWeight = 1.0f
	};

	// std::unique_ptr<CubicNodeTree<2>> scalarField = 
	// 	SmoothSurfaceReconstruction::compute2DCubicNodeTree<2>(cloud, config);

	std::unique_ptr<LinearNodeTree<2>> scalarField = 
		SmoothSurfaceReconstruction::computeLinearNodeTree<2, EigenSquareSolver>(cloud, config);

	Image image;
	image.init(1024, 1024);
	drawScalarField(image, *scalarField, cloud, true);
	image.savePNG("quadtree.png"); 

	// Image imageG;
	// imageG.init(1024, 1024);
	// drawGradientMaginitudeImage(imageG, *scalarField);
	// imageG.savePNG("quadtreeGrad.png"); 

	return 0;
}


