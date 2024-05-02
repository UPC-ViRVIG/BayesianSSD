#include <iostream>
#include "SmoothSurfaceReconstruction.h"
#include "EigenSquareSolver.h"

int main(int argc, char *argv[])
{
	PointCloud<3> cloud;
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
		.pointFilterMaxDistance = maxSize / static_cast<float>(1 << maxDepth) ,
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

    // Generate mesh

	return 0;
}