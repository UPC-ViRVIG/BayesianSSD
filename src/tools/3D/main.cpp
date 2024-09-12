#include <iostream>
#include "SmoothSurfaceReconstruction.h"
#include "EigenSquareSolver.h"
#include "MyImage.h"

#include <cereal/archives/portable_binary.hpp>

void drawScalarField(Image& image, LinearNodeTree<3>& scalarField, 
					 std::optional<std::reference_wrapper<PointCloud<3>>> cloud = std::optional<std::reference_wrapper<PointCloud<3>>>(), 
					 bool drawGrid = true)
{
	NodeTree<3>& qtree = scalarField.getNodeTree();

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

            glm::vec3 rpos = glm::vec3(pos.x, pos.y, 0.5f);
            rpos = scalarField.getMinCoord() + rpos * (scalarField.getMaxCoord() - scalarField.getMinCoord());

			// Field color
			const float pval = scalarField.eval(rpos);
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
			// if(drawGrid) 
			// {
			// 	std::optional<NodeTree<2>::Node> node;
			// 	qtree.getNode(pos, node);
			// 	if(!node) continue;
			// 	glm::vec2 invSize = 1.0f / (node->maxCoord - node->minCoord);
			// 	glm::vec2 np = (pos - node->minCoord) * invSize;
			// 	np = glm::abs(2.0f * (np - glm::vec2(0.5f)));
			// 	gridColorWeight = glm::smoothstep(1.0f - 0.005f * invSize.x, 1.0f - 0.003f * invSize.x, glm::max(np.x, np.y));
			// }

			image(i, j) = glm::mix(bgColor, glm::vec3(0.0f), glm::max(glm::max(surfaceColorWeight, linesColorWeight), gridColorWeight));
		}
	}

	// if(cloud)
	// {
	// 	for (glm::vec2 point : cloud->get().getPoints())
	// 	{
	// 		image.drawFilledCircle(static_cast<uint32_t>(image.width() * point.x), static_cast<uint32_t>(image.height() * point.y), 4, glm::vec3(0.35f, 0.35f, 0.35f));
	// 	}
	// }
}

int main(int argc, char *argv[])
{
	PointCloud<3> cloud;
	if(argc != 2 && argc != 3)
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

    glm::vec3 min(INFINITY);
    glm::vec3 max(-INFINITY);
    for(glm::vec3 p : cloud.getPoints())
    {
        for(uint32_t i=0; i < 3; i++)
        {
            min[i] = glm::min(min[i], p[i]);
            max[i] = glm::max(max[i], p[i]);
        }
    }

    const glm::vec3 size = max - min;
    const glm::vec3 center = 0.5f * (max + min);
    float maxSize = glm::max(size.x, glm::max(size.y, size.z));
    // Add margin
    maxSize = 1.2f * maxSize;

	const uint32_t maxDepth = 6;
	NodeTree<3>::Config quadConfig = {
		.minCoord = center - glm::vec3(0.5f * maxSize),
		.maxCoord = center + glm::vec3(0.5f * maxSize),
		.pointFilterMaxDistance = maxSize / static_cast<float>(1 << maxDepth),
        // .pointFilterMaxDistance = 0.0f,
		.constraintNeighbourNodes = false,
		.maxDepth = maxDepth
	};

	SmoothSurfaceReconstruction::Config<3> config = {
		.nodeTreeConfig = quadConfig,
		.posWeight = 1.0f, 
        .gradientWeight = 1.0f,
        .smoothWeight = 1.0f
	};

	std::unique_ptr<LinearNodeTree<3>> scalarField = 
		SmoothSurfaceReconstruction::computeLinearNodeTree<3, EigenSquareSolver>(cloud, config);

    // Export octree
    std::ofstream os("./output/test.bin", std::ios::out | std::ios::binary);
    cereal::PortableBinaryOutputArchive archive(os);
    archive(*scalarField);

    // Generate mesh
    // Image image;
	// image.init(1024, 1024);
	// drawScalarField(image, *scalarField);
	// image.savePNG("quadtree.png"); 


	return 0;
}