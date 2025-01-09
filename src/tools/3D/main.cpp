#include <iostream>
#include "SmoothSurfaceReconstruction.h"
#include "EigenSquareSolver.h"
#include "MyImage.h"
#include "Timer.h"

#include <cereal/archives/portable_binary.hpp>

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

	cloud.computeNormals();

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
    maxSize = 1.4f * maxSize;

	const uint32_t maxDepth = 5;
	NodeTree<3>::Config quadConfig = {
		.minCoord = center - glm::vec3(0.5f * maxSize),
		.maxCoord = center + glm::vec3(0.5f * maxSize),
		.pointFilterMaxDistance = 1.55f * maxSize / static_cast<float>(1 << maxDepth),
		.constraintNeighbourNodes = true,
		.maxDepth = maxDepth
	};

	SmoothSurfaceReconstruction::Config<3> config = {
		.posWeight = 1.0f, 
        .gradientWeight = 1.0f/0.5f,
        .smoothWeight = 1.0f/100.0f,
		.algorithm = SmoothSurfaceReconstruction::Algorithm::VAR,
		.computeVariance = false
	};

	Timer timer;
	timer.start();

	NodeTree<3> octree;
	octree.compute(cloud, quadConfig);

	std::optional<LinearNodeTree<3>> covScalarField;

	std::unique_ptr<LinearNodeTree<3>> scalarField =
		SmoothSurfaceReconstruction::computeLinearNodeTree<3>(std::move(octree), cloud, config, covScalarField);

	std::cout << "Compute linear node " << timer.getElapsedSeconds() << std::endl;

    // Export octree
    std::ofstream os("./output/test.bin", std::ios::out | std::ios::binary);
    cereal::PortableBinaryOutputArchive archive(os);
    archive(*scalarField);

	std::cout << "output" << std::endl;

	return 0;
}