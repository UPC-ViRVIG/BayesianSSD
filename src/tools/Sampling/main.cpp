#define EIGEN_MAX_ALIGN_BYTES 64

#include <iostream>
#include "SmoothSurfaceReconstruction.h"
#include "EigenSquareSolver.h"
#include "MyImage.h"
#include "Timer.h"
#include <omp.h>
#include <MeshReconstruction.h>
#include <happly.h>
#include <random>
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
	float normalsDistanceFactor;
	float normalsPointsCorrelation;
	uint32_t normalsNumNearPoints;
	float gradiantXYVariance;
	float mulStd;

	JS_OBJ(pointCloudName, outputName, bbMargin, octreeMaxDepth, octreeSubRuleInVoxels, gradiantVariance, smoothnessVariance, computeVariance, normalsDistanceFactor, normalsPointsCorrelation, normalsNumNearPoints, gradiantXYVariance, mulStd);
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
	omp_set_num_threads(numThreads);
	Eigen::setNbThreads(numThreads);
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

    // Export point cloud
	cloud.writeToFile("./output/" + inConfig.outputName + "_input.ply");

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
		.gradientXYWeight = 1.0f/inConfig.gradiantXYVariance,
        // .smoothWeight = 1.0f/150.0f,
		.smoothWeight = 1.0f/inConfig.smoothnessVariance,
		.algorithm = SmoothSurfaceReconstruction::Algorithm::BAYESIAN,
		.computeVariance = inConfig.computeVariance,
		.invAlgorithm = SmoothSurfaceReconstruction::InverseAlgorithm::BASE_RED
	};

    std::vector<float> orgPVar = cloud.getVariances();
    {
        constexpr uint32_t Dim = 3;
        using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud<Dim>>, PointCloud<Dim>, Dim /* dim */>;
	    my_kd_tree_t kdtree(Dim /* dim */, cloud, {10 /* max depth */});


        std::vector<uint32_t> nearIndexCache(2);
	    std::vector<float> nearDistSqrCache(2);
        double meanPairDistance = 0.0;
        double invSize = 1.0 / static_cast<double>(cloud.size());
        for(uint32_t i=0; i < cloud.size(); i++)
	    {
            uint32_t numResults = kdtree.knnSearch(reinterpret_cast<const float*>(&cloud.point(i)), 2, nearIndexCache.data(), nearDistSqrCache.data());
            for(uint32_t j=0; j < numResults; j++)
            {
                if(i != nearIndexCache[j])
                {
                    meanPairDistance += invSize * glm::length(cloud.point(i) - cloud.point(nearIndexCache[j]));
                    break;
                }
            }   
        }

        std::vector<nanoflann::ResultItem<uint32_t, float>> nearPoints;
        std::vector<float> pVar = cloud.getVariances();
        double meanSqPairDistance = 2.0 * meanPairDistance * meanPairDistance;
        for(uint32_t i=0; i < cloud.size(); i++)
        {
            uint32_t numResults = kdtree.radiusSearch(reinterpret_cast<const float*>(&cloud.point(i)), 4 * meanPairDistance, nearPoints);
            const double stdV1 = glm::sqrt(static_cast<double>(cloud.variance(i)));
            for(uint32_t j=0; j < numResults; j++)
            {
                const uint32_t jIdx = nearPoints[j].first;
                const glm::vec3 d = cloud.point(i) - cloud.point(jIdx);
                pVar[i] += stdV1 * glm::sqrt(static_cast<double>(cloud.variance(jIdx))) * glm::exp(-glm::dot(d, d)/(2 * meanSqPairDistance));
            }
        }
        cloud.getVariances() = pVar;
    }

    Timer timer;
	timer.start();

	NodeTree<3> octree;
	octree.compute(cloud, quadConfig);
	std::cout << "Octree creation " << timer.getElapsedSeconds() << std::endl;

    cloud.computeNormals(inConfig.normalsNumNearPoints, inConfig.normalsDistanceFactor, inConfig.normalsPointsCorrelation);

    std::optional<LinearNodeTree<3>> orgCovScalarField;
	std::unique_ptr<LinearNodeTree<3>> orgScalarField =
		SmoothSurfaceReconstruction::computeLinearNodeTree<3>(std::move(octree), cloud, config, orgCovScalarField);
	std::cout << "Compute linear node " << timer.getElapsedSeconds() << std::endl;


    std::random_device rd; // Obtain a random seed from the hardware
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::normal_distribution<> dis(0.0, 1.0); // Define the normal distribution

    std::vector<std::vector<float>> values;
    std::vector<double> meanValues(orgScalarField->getVertexValues().size());
    std::vector<double> varValues(orgScalarField->getVertexValues().size());
    auto computeMeanAndVar = [&]()
    {
        for(double& v : meanValues) v = 0.0;
        for(double& v : varValues) v = 0.0;
        double invSize = 1.0 / static_cast<double>(values.size());
        for(uint32_t i=0; i < values.size(); i++)
        {
            for(uint32_t j=0; j < meanValues.size(); j++)
            {
                meanValues[j] += invSize * values[i][j];
            }
        }
        for(uint32_t i=0; i < values.size(); i++)
        {
            for(uint32_t j=0; j < varValues.size(); j++)
            {
                varValues[j] += invSize * (values[i][j] - meanValues[j]) * (values[i][j] - meanValues[j]);
            }
        }

        invSize = 1.0 / static_cast<double>(meanValues.size());
        double meanError = 0.0;
        double stdError = 0.0;
        for(uint32_t j=0; j < meanValues.size(); j++)
        {
            meanError += invSize * glm::abs(meanValues[j] - orgScalarField->getVertexValues()[j]);
            stdError += invSize * glm::sqrt(varValues[j]) / glm::sqrt(orgCovScalarField->getVertexValues()[j]);
        }

        std::cout << meanError << ", " << stdError << std::endl;
    };

    PointCloud<3> sCloud;
    for(uint32_t it=0; it < 1000; it++)
    {
        // Add noise
        sCloud = cloud;
        for (int i = 0; i < sCloud.size(); ++i) {
            const double std = glm::sqrt(orgPVar[i]);
            glm::vec3 r;
            for(uint32_t d=0; d < 3; d++) r[d] = std * dis(gen);
            sCloud.point(i) = sCloud.point(i) + r;
        }

	    sCloud.computeNormals(inConfig.normalsNumNearPoints, inConfig.normalsDistanceFactor, inConfig.normalsPointsCorrelation);

        for(uint32_t i=0; i < sCloud.size(); i++)
        {
            sCloud.getVariances()[i] *= inConfig.mulStd * inConfig.mulStd;
        }

        config.computeVariance = false;
        std::optional<LinearNodeTree<3>> covScalarField;
        std::unique_ptr<LinearNodeTree<3>> scalarField =
            SmoothSurfaceReconstruction::computeLinearNodeTree<3>(std::move(octree), sCloud, config, covScalarField);

        values.push_back(scalarField->getVertexValues());
        if(it != 0 && it % 200 == 0)
        {
            computeMeanAndVar();        
        }
    }

    computeMeanAndVar();
    return 0;

	// Execute Marching Cubes
	// auto toVec3 = [](glm::vec3 p) -> MeshReconstruction::Vec3
	// {
	// 	return MeshReconstruction::Vec3{p.x, p.y, p.z};
	// };
	// auto fSdf = [&](MeshReconstruction::Vec3 const& point) -> double
	// {
	// 	return static_cast<double>(scalarField->eval(glm::vec3(point.x, point.y, point.z)));
	// };
	// auto gSdf = [&](MeshReconstruction::Vec3 const& point) -> MeshReconstruction::Vec3
	// {
	// 	glm::vec3 grad = scalarField->evalGrad(glm::vec3(point.x, point.y, point.z));
	// 	return toVec3(grad);
	// };
	// const glm::vec3 bbSize = scalarField->getMaxCoord() - scalarField->getMinCoord();
	// MeshReconstruction::Rect3 mcDomain { toVec3(scalarField->getMinCoord()), 
	// 									 toVec3(bbSize)};
	// MeshReconstruction::Vec3 cubeSize = toVec3(glm::vec3(bbSize / static_cast<float>(1 << (maxDepth+1))));
	// MeshReconstruction::Mesh mesh = MeshReconstruction::MarchCube(fSdf, mcDomain, cubeSize, 0.0, gSdf);

	// std::cout << "Num vertices " << mesh.vertices.size() << std::endl;

	// // Export mesh
	// happly::PLYData plyOut;
	// plyOut.addVertexPositions(*reinterpret_cast<std::vector<std::array<double, 3>>*>(&mesh.vertices));
	// std::vector<float> verticesStd;
	// for(uint32_t i=0; i < mesh.vertices.size(); i++)
	// {
	// 	float std = glm::sqrt(covScalarField.value().eval(glm::vec3(mesh.vertices[i].x, mesh.vertices[i].y, mesh.vertices[i].z)));
	// 	verticesStd.push_back(std);
	// }
	// plyOut.getElement("vertex").addProperty("noise_std", verticesStd);
	// std::vector<std::vector<uint32_t>> indices;
	// indices.reserve(mesh.triangles.size());
	// for(auto a : mesh.triangles) indices.emplace_back(a.begin(), a.end());
	// plyOut.addFaceIndices(indices);
	// plyOut.write("./output/" + inConfig.outputName + ".ply", happly::DataFormat::ASCII);
	// std::cout << "Done" << std::endl;
	
    // // Export octree
	// {
	// 	std::ofstream os("./output/" + inConfig.outputName + ".bin", std::ios::out | std::ios::binary);
	// 	cereal::PortableBinaryOutputArchive archive(os);
	// 	archive(*scalarField);
	// }

	// if(config.computeVariance)
	// {
	// 	std::ofstream os("./output/" + inConfig.outputName + "_var.bin", std::ios::out | std::ios::binary);
	// 	cereal::PortableBinaryOutputArchive archive(os);
	// 	archive(*covScalarField);
	// }

	// std::cout << "output" << std::endl;

	// return 0;
}