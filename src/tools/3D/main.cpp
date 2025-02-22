#define EIGEN_MAX_ALIGN_BYTES 64

#include <iostream>
#include "SmoothSurfaceReconstruction.h"
#include "EigenSquareSolver.h"
#include "MyImage.h"
#include "Timer.h"
#include <omp.h>
#include <MeshReconstruction.h>
#include <happly.h>
#include <json_struct/json_struct.h>

#include <cereal/archives/portable_binary.hpp>



struct InputConfig
{
	std::string pointCloudName;
	std::string outputName;
	float bbMargin;
	float octreeMaxDepth;
	float octreeSubRuleInVoxels;
	float gradiantVariance;
	float smoothnessVariance;
	bool computeVariance;

	JS_OBJ(pointCloudName, outputName, bbMargin, octreeMaxDepth, octreeSubRuleInVoxels, gradiantVariance, smoothnessVariance, computeVariance);
};

char* loadFromFile(std::string path, size_t* length)
{
    std::ifstream file;
	file.open(path, std::ios_base::in | std::ios_base::binary);
	if (!file.good()) return nullptr;
	file.seekg(0, std::ios::end);
	*length = file.tellg();
	(*length)++;
	char* ret = new char[*length];
	file.seekg(0, std::ios::beg);
	file.read(ret, *length);
	file.close();
	ret[(*length) - 1] = 0;
	return ret;
}

int main(int argc, char *argv[])
{
#ifdef OPENMP_AVAILABLE
	const uint32_t numThreads = 20;
	std::cout << "Using OpenMP, " << numThreads << " threads" << std::endl;
	omp_set_num_threads(20);
	Eigen::setNbThreads(20);
#endif

	PointCloud<3> cloud;
	if(argc != 2 && argc != 3)
	{
		std::cout << "Wrong arguments!" << std::endl << std::endl;
		std::cout << "\t" << argv[0] << " <Input cloud> <Output image> [resolution]" << std::endl;
		return -1;
	}

	
	size_t json_data_size;
	char* json_data = loadFromFile(argv[1], &json_data_size);
	JS::ParseContext context(json_data, json_data_size);
	InputConfig inConfig;
	context.parseTo(inConfig);
	
	if(!cloud.readFromFile("./data/" + inConfig.pointCloudName + ".ply"))
	{
		std::cout << "Input cloud could not be read!" << std::endl;
		return -1;
	}

	Timer timer;
	timer.start();
	cloud.computeNormals(0.5);
	std::cout << "Time computing normals: " << timer.getElapsedSeconds() << std::endl;

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
    maxSize = (1.0f + inConfig.bbMargin) * maxSize;

	const uint32_t maxDepth = inConfig.octreeMaxDepth;
	NodeTree<3>::Config quadConfig = {
		.minCoord = center - glm::vec3(0.5f * maxSize),
		.maxCoord = center + glm::vec3(0.5f * maxSize),
		.pointFilterMaxDistance = inConfig.octreeSubRuleInVoxels * maxSize / static_cast<float>(1 << maxDepth),
		.constraintNeighbourNodes = true,
		.maxDepth = maxDepth
	};

	SmoothSurfaceReconstruction::Config<3> config = {
		.posWeight = 1.0f, 
        .gradientWeight = 1.0f/inConfig.gradiantVariance,
        // .smoothWeight = 1.0f/150.0f,
		.smoothWeight = 1.0f/inConfig.smoothnessVariance,
		.algorithm = SmoothSurfaceReconstruction::Algorithm::VAR,
		.computeVariance = inConfig.computeVariance,
		.invAlgorithm = SmoothSurfaceReconstruction::InverseAlgorithm::BASE_RED
	};

	Timer timer;
	timer.start();

	NodeTree<3> octree;
	octree.compute(cloud, quadConfig);

	std::cout << "Octree creation " << timer.getElapsedSeconds() << std::endl;

	std::optional<LinearNodeTree<3>> covScalarField;

	std::unique_ptr<LinearNodeTree<3>> scalarField =
		SmoothSurfaceReconstruction::computeLinearNodeTree<3>(std::move(octree), cloud, config, covScalarField);

	std::cout << "Compute linear node " << timer.getElapsedSeconds() << std::endl;


	// Execute Marching Cubes
	auto toVec3 = [](glm::vec3 p) -> MeshReconstruction::Vec3
	{
		return MeshReconstruction::Vec3{p.x, p.y, p.z};
	};
	auto fSdf = [&](MeshReconstruction::Vec3 const& point) -> double
	{
		return static_cast<double>(scalarField->eval(glm::vec3(point.x, point.y, point.z)));
	};
	auto gSdf = [&](MeshReconstruction::Vec3 const& point) -> MeshReconstruction::Vec3
	{
		glm::vec3 grad = scalarField->evalGrad(glm::vec3(point.x, point.y, point.z));
		return toVec3(grad);
	};
	const glm::vec3 bbSize = scalarField->getMaxCoord() - scalarField->getMinCoord();
	MeshReconstruction::Rect3 mcDomain { toVec3(scalarField->getMinCoord()), 
										 toVec3(bbSize)};
	MeshReconstruction::Vec3 cubeSize = toVec3(glm::vec3(bbSize / static_cast<float>(1 << (maxDepth+1))));
	MeshReconstruction::Mesh mesh = MeshReconstruction::MarchCube(fSdf, mcDomain, cubeSize, 0.0, gSdf);

	std::cout << "Num vertices " << mesh.vertices.size() << std::endl;

	// Export point cloud
	cloud.writeToFile("./output/" + inConfig.outputName + "_input.ply");

	// Export mesh
	happly::PLYData plyOut;
	plyOut.addVertexPositions(*reinterpret_cast<std::vector<std::array<double, 3>>*>(&mesh.vertices));
	std::vector<std::vector<uint32_t>> indices;
	indices.reserve(mesh.triangles.size());
	for(auto a : mesh.triangles) indices.emplace_back(a.begin(), a.end());
	plyOut.addFaceIndices(indices);
	plyOut.write("./output/" + inConfig.outputName + ".ply", happly::DataFormat::ASCII);
	std::cout << "Done" << std::endl;
	
    // Export octree
	{
		std::ofstream os("./output/" + inConfig.outputName + ".bin", std::ios::out | std::ios::binary);
		cereal::PortableBinaryOutputArchive archive(os);
		archive(*scalarField);
	}

	if(config.computeVariance)
	{
		std::ofstream os("./output/" + inConfig.outputName + "_var.bin", std::ios::out | std::ios::binary);
		cereal::PortableBinaryOutputArchive archive(os);
		archive(*covScalarField);
	}

	std::cout << "output" << std::endl;

	return 0;
}