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
	float gradientStd;
	float smoothnessStd;
	bool computeVariance;
	float normalsDistanceFactor;
	float normalsPointsCorrelation;
	uint32_t normalsNumNearPoints;
	float gradiantXYVariance;
	float mulStd;
	uint32_t invRedMatRank;

	JS_OBJ(pointCloudName, outputName, bbMargin, octreeMaxDepth, octreeSubRuleInVoxels, gradientStd, smoothnessStd, computeVariance, normalsDistanceFactor, normalsPointsCorrelation, normalsNumNearPoints, gradiantXYVariance, mulStd, invRedMatRank);
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

	uint32_t mode = 1;
	if(argc > 2)
	{
		mode = std::stoi(std::string(argv[2]));
	}

	
	size_t json_data_size;
	char* json_data = loadFromFile(argv[1], &json_data_size);
	JS::ParseContext context(json_data, json_data_size);
	InputConfig inConfig;
	context.parseTo(inConfig);

	// inConfig.gradiantXYVariance = 1.0;
	// inConfig.mulStd = 1.0;
	// inConfig.invRedMatRank = 512;
	
	if(!cloud.readFromFile("./data/" + inConfig.pointCloudName + ".ply"))
	{
		std::cout << "Input cloud could not be read!" << std::endl;
		return -1;
	}

	// {
    //     constexpr uint32_t Dim = 3;
    //     using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud<Dim>>, PointCloud<Dim>, Dim /* dim */>;
	//     my_kd_tree_t kdtree(Dim /* dim */, cloud, {10 /* max depth */});


    //     std::vector<uint32_t> nearIndexCache(2);
	//     std::vector<float> nearDistSqrCache(2);
    //     double meanPairDistance = 0.0;
    //     double invSize = 1.0 / static_cast<double>(cloud.size());
    //     for(uint32_t i=0; i < cloud.size(); i++)
	//     {
    //         uint32_t numResults = kdtree.knnSearch(reinterpret_cast<const float*>(&cloud.point(i)), 2, nearIndexCache.data(), nearDistSqrCache.data());
    //         for(uint32_t j=0; j < numResults; j++)
    //         {
    //             if(i != nearIndexCache[j])
    //             {
    //                 meanPairDistance += invSize * glm::length(cloud.point(i) - cloud.point(nearIndexCache[j]));
    //                 break;
    //             }
    //         }   
    //     }

    //     std::vector<nanoflann::ResultItem<uint32_t, float>> nearPoints;
    //     std::vector<float> pVar = cloud.getVariances();
    //     double meanSqPairDistance = 2.0 * meanPairDistance * meanPairDistance;
    //     for(uint32_t i=0; i < cloud.size(); i++)
    //     {
    //         uint32_t numResults = kdtree.radiusSearch(reinterpret_cast<const float*>(&cloud.point(i)), 4 * meanPairDistance, nearPoints);
    //         const double stdV1 = glm::sqrt(static_cast<double>(cloud.variance(i)));
    //         for(uint32_t j=0; j < numResults; j++)
    //         {
    //             const uint32_t jIdx = nearPoints[j].first;
    //             const glm::vec3 d = cloud.point(i) - cloud.point(jIdx);
    //             pVar[i] += stdV1 * glm::sqrt(static_cast<double>(cloud.variance(jIdx))) * glm::exp(-glm::dot(d, d)/(2 * meanSqPairDistance));
    //         }
    //     }
    //     cloud.getVariances() = pVar;
    // }

	PointCloud<3> orgCloud = cloud;
	
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
	const float bbRadius = 0.5f * glm::distance(min, max);

	Timer timer;
	timer.start();
	cloud.computeNormals(inConfig.normalsNumNearPoints, 0.0, inConfig.normalsPointsCorrelation, inConfig.normalsDistanceFactor / bbRadius);
	// cloud.computeNormals(inConfig.normalsNumNearPoints, inConfig.normalsDistanceFactor, inConfig.normalsPointsCorrelation, 0.0f);
	// cloud.fillNormalsData(0.0001);
	// std::cout << "Time computing normals: " << timer.getElapsedSeconds() << std::endl;

	for(uint32_t i=0; i < cloud.size(); i++)
	{
		cloud.getVariances()[i] *= inConfig.mulStd * inConfig.mulStd;
	}

	// Export point cloud
	cloud.writeToFile("./output/" + inConfig.outputName + "_input.ply");
	
	std::cout << timer.getElapsedSeconds() << std::endl;
	// Compute errors
	// double meanAngleError = 0.0;
	// const double invSize = 1.0 / static_cast<double>(cloud.size());
	// for(uint32_t i=0; i < cloud.size(); i++)
	// {
	// 	const double val = glm::acos(static_cast<double>(glm::min(glm::dot(cloud.normal(i), orgCloud.normal(i)), 0.999999f)));
	// 	meanAngleError += val * invSize;
	// }
	// std::cout << meanAngleError << std::endl;
	// double varAngleError = 0.0;
	// std::vector<float> xyErrors;
	// for(uint32_t i=0; i < cloud.size(); i++)
	// {
	// 	const double val = glm::acos(static_cast<double>(glm::min(glm::dot(cloud.normal(i), orgCloud.normal(i)), 0.999999f)));
	// 	varAngleError += val * val * invSize;
	// 	auto& icovT = cloud.normalInvCovarianceDes(i);
	// 	Eigen::Vector<double, 3> g;
	// 	g(0) = orgCloud.normal(i)[0]; g(1) = orgCloud.normal(i)[1]; g(2) = orgCloud.normal(i)[2];
	// 	g = std::get<0>(icovT) * g;
	// 	xyErrors.push_back(glm::sqrt(std::get<1>(icovT)(0)) * g(0));
	// 	xyErrors.push_back(glm::sqrt(std::get<1>(icovT)(1)) * g(1));
	// }
	// write_array_to_file(xyErrors, "./output/stats/" + inConfig.outputName + "_xyErrors.bin");
	// std::cout << glm::sqrt(varAngleError) << std::endl;

	// return 0; // compute only normals;

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
        .gradientWeight = 1.0f/inConfig.gradientStd,
		.gradientXYWeight = 1.0f/inConfig.gradiantXYVariance,
        // .smoothWeight = 1.0f/150.0f,
		.smoothWeight = 1.0f/inConfig.smoothnessStd,
		.algorithm = SmoothSurfaceReconstruction::Algorithm::BAYESIAN,
		.computeVariance = inConfig.computeVariance,
		.invAlgorithm = static_cast<SmoothSurfaceReconstruction::InverseAlgorithm>(mode),
		.invRedMatRank = inConfig.invRedMatRank
	};

	timer.start();

	NodeTree<3> octree;
	octree.compute(cloud, quadConfig);

	std::cout << "Octree creation " << timer.getElapsedSeconds() << std::endl;

	NodeTree<3> coctree;
	bool computeSimpleVariance = mode == 5 && config.computeVariance;
	if(computeSimpleVariance) 
	{
		config.computeVariance = false;
		coctree = octree;
	}
	std::optional<LinearNodeTree<3>> covScalarField;
	std::unique_ptr<LinearNodeTree<3>> scalarField =
		SmoothSurfaceReconstruction::computeLinearNodeTree<3>(std::move(octree), cloud, config, covScalarField);

	if(computeSimpleVariance)
	{
		config.computeVariance = true;
		quadConfig.maxDepth -= 3;
		config.invAlgorithm = SmoothSurfaceReconstruction::InverseAlgorithm::FULL;
		NodeTree<3> octreeS;
		octreeS.compute(cloud, quadConfig);

		std::optional<LinearNodeTree<3>> covScalarFieldS;
		std::unique_ptr<LinearNodeTree<3>> scalarFieldS =
			SmoothSurfaceReconstruction::computeLinearNodeTree<3>(std::move(octreeS), cloud, config, covScalarFieldS);

		// std::vector<float> interVar;
		// for(uint32_t i=0; i < unkownVertices.size(); i++)
		// {
		// 	interVar.push_back(covScalarFieldS->eval(unkownVertices[i]));
		// }
		// write_array_to_file(interVar, "./vecVarSimp.bin");

		const auto& qVertices = scalarField->getNodeTree().getVertices();
		std::vector<float> verticesValue(qVertices.size());
		for(uint32_t i=0; i < qVertices.size(); i++)
		{
			verticesValue[i] = covScalarFieldS->eval(qVertices[i]);
		}
		covScalarField = LinearNodeTree<3>(std::move(coctree), std::move(verticesValue));
	}

	std::cout << "Compute linear node " << timer.getElapsedSeconds() << std::endl;


	// std::map<std::tuple<int, int, int>, double> newValues;

	// std::ifstream inputFile("gridData.bin", std::ios::binary);
    // if (!inputFile.is_open()) {
    //     std::cerr << "Error opening binary file!" << std::endl;
    //     return 1;
    // }

    // int pos[3];
    // double v;
    // while (inputFile.read(reinterpret_cast<char*>(&pos[0]), sizeof(pos[0]))) {
    //     if (!inputFile.read(reinterpret_cast<char*>(&pos[1]), sizeof(pos[1]))) break;
    //     if (!inputFile.read(reinterpret_cast<char*>(&pos[2]), sizeof(pos[2]))) break;
    //     if (!inputFile.read(reinterpret_cast<char*>(&v), sizeof(v))) break;
    //     newValues[std::make_tuple(pos[0], pos[1], pos[2])] = v;
    // }
	// std::cout << "NumValues " << newValues.size() << std::endl;

    // inputFile.close();

	// uint32_t notFound = 0;
	// for(uint32_t i=0; i < scalarField->getNodeTree().getNumVertices(); i++)
	// {
	// 	glm::vec3 p = scalarField->getNodeTree().getVertices()[i];
	// 	glm::ivec3 ip = glm::round(64.0f * (p - scalarField->getNodeTree().getMinCoord()) / (scalarField->getNodeTree().getMaxCoord() - scalarField->getNodeTree().getMinCoord()));
	// 	auto it = newValues.find(std::make_tuple(ip.x, ip.y, ip.z));
	// 	if(it != newValues.end())
	// 	{
	// 		scalarField->getVertexValues()[i] = it->second;
	// 		if(i % 50000 == 0) std::cout << it->second << std::endl;
	// 	} else notFound++;
	// }
	// std::cout << "Not Found " << notFound << std::endl;


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
	// MeshReconstruction::Vec3 cubeSize = toVec3(glm::vec3(bbSize / static_cast<float>(1 << (maxDepth+1))));
	MeshReconstruction::Vec3 cubeSize = toVec3(glm::vec3(bbSize / static_cast<float>(1 << maxDepth-1)));
	MeshReconstruction::Mesh mesh = MeshReconstruction::MarchCube(fSdf, mcDomain, cubeSize, 0.0, gSdf);

	std::cout << "Num vertices " << mesh.vertices.size() << std::endl;

	// Export mesh
	happly::PLYData plyOut;
	plyOut.addVertexPositions(*reinterpret_cast<std::vector<std::array<double, 3>>*>(&mesh.vertices));
	if(inConfig.computeVariance)
	{
		std::vector<float> verticesStd;
		for(uint32_t i=0; i < mesh.vertices.size(); i++)
		{
			float std = glm::sqrt(covScalarField.value().eval(glm::vec3(mesh.vertices[i].x, mesh.vertices[i].y, mesh.vertices[i].z)));
			verticesStd.push_back(std);
		}
		plyOut.getElement("vertex").addProperty("noise_std", verticesStd);
	}
	std::vector<std::vector<uint32_t>> indices;
	indices.reserve(mesh.triangles.size());
	for(auto a : mesh.triangles) indices.emplace_back(a.begin(), a.end());
	plyOut.addFaceIndices(indices);
	plyOut.write("./output/" + inConfig.outputName + ".ply", happly::DataFormat::Binary);
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