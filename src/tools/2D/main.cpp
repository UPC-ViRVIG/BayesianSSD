#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <numbers>
#include <random>
#include "SmoothSurfaceReconstruction.h"
#include "GPReconstruction.h"
#include "PoissonReconstruction.h"
#include "EigenSquareSolver.h"
#include "MyImage.h"
#include "ScalarFieldRender.h"

#include <Eigen/Cholesky>

#define M_PI 3.14159265359

template<typename SF>
void drawGradientMaginitudeImage(Image& image, SF& scalarField)
{
	NodeTree<2>& qtree = scalarField.getNodeTree();

	const std::array<glm::vec3, 2> palette = {
		glm::vec3(1.0f, 1.0f, 1.0f),
		glm::vec3(1.0f, 0.0f, 0.0f)
	};

	glm::vec2 minCoord = scalarField.getMinCoord();
	glm::vec2 size = scalarField.getMaxCoord() - scalarField.getMinCoord();
	glm::vec2 invSize = 1.0f / size;
	float invSizeMag = glm::length(glm::vec2(image.width(), image.height()) * invSize);
	float winSizeMag = glm::length(glm::vec2(image.width(), image.height()));	
	const float maxValue = scalarField.getMaxAbsValue();
	for(uint32_t i=0; i < image.width(); i++)
	{
		for(uint32_t j=0; j < image.height(); j++)
		{
			glm::vec2 pos = glm::vec2((static_cast<float>(i)+0.5f) / static_cast<float>(image.width()), 
									  (static_cast<float>(j)+0.5f) / static_cast<float>(image.height()));
			pos = minCoord + pos * size;

			// Field color
			const glm::vec2 pval = scalarField.evalGrad(pos);
			const float len = glm::abs(glm::length(pval)-1.0f);
			float val = glm::clamp(len, 0.0f, 0.999999f);
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

template<typename SF>
void drawScalarField(Image& image, SF& scalarField, 
					 std::optional<std::reference_wrapper<PointCloud<2>>> cloud = std::optional<std::reference_wrapper<PointCloud<2>>>(), 
					 bool drawGrid = true,
					 std::optional<std::reference_wrapper<std::vector<float>>> verticesEnergy = std::optional<std::reference_wrapper<std::vector<float>>>())
{
	const NodeTree<2>& qtree = scalarField.getNodeTree();

	const std::vector<glm::vec3> sdfPalette = {
		glm::vec3(0.0f, 0.0f, 1.0f), 
		glm::vec3(0.0f, 0.5f, 1.0f), 
		glm::vec3(0.0f, 1.0f, 1.0f), 
		glm::vec3(1.0f, 1.0f, 1.0f), 
		glm::vec3(1.0f, 1.0f, 0.0f), 
		glm::vec3(1.0f, 0.5f, 0.0f), 
		glm::vec3(1.0f, 0.0f, 0.0f)
	};
	
	const float maxValue = scalarField.getMaxAbsValue();
	ScalarFieldRender::renderScalarField(scalarField, image, [maxValue](float val) { return 0.5f * (1.0f + val / maxValue); }, sdfPalette,
										 1.0f, 0.0f, 1.0f, 
										 0.5f, 0.007f, 0.8f);

	if(drawGrid)
	{
		ScalarFieldRender::renderNodeTreeGrid(qtree, image, 2.0f, 0.6f);
	}

	// const std::array<glm::vec3, 2> palette = {
	// 	glm::vec3(1.0f, 1.0f, 1.0f), 
	// 	glm::vec3(1.0f, 0.0f, 0.0f)
	// };

	if(cloud)
	{
		glm::vec2 minCoord = scalarField.getMinCoord();
		glm::vec2 size = scalarField.getMaxCoord() - scalarField.getMinCoord();
		for (glm::vec2 point : cloud->get().getPoints())
		{
			point = (point - minCoord) / size;
			image.drawFilledCircle(static_cast<uint32_t>(image.width() * point.x), static_cast<uint32_t>(image.height() * point.y), 10, glm::vec3(0.35f, 0.35f, 0.35f));
			// image.drawCircle(static_cast<uint32_t>(image.width() * point.x), static_cast<uint32_t>(image.height() * point.y),
			// 				 3.0 * glm::sqrt(cloud->get().variance(i)), 5.0f, glm::mix(redPalette[0], redPalette[1], glm::vec3(0.35f, 0.35f, 0.35f)));
		}
	}

	if(verticesEnergy)
	{
		std::vector<float>& vEnergy = verticesEnergy->get();
		const std::array<glm::vec3, 2> energyPalette = {
			glm::vec3(1.0f, 1.0f, 1.0f), 
			glm::vec3(1.0f, 0.0f, 0.0f)
		};

		const float maxV = *std::max_element(vEnergy.begin(), vEnergy.end());
		for(uint32_t i=0; i < qtree.getNumVertices(); i++)
		{
			const glm::vec2 point = qtree.getVertices()[i];
			for(uint32_t j = 0; j < 4; j++)
			{
				if(vEnergy[4 * i + j] == 0.0f) continue;
				float val = static_cast<float>(energyPalette.size()-1) * glm::clamp(vEnergy[4 * i + j] / maxV, 0.0f, 0.9999f);
				uint32_t cid = glm::floor(val);
				glm::vec3 color = glm::mix(energyPalette[cid], energyPalette[cid+1], glm::fract(val));
				glm::vec2 offset((j & 0b10) == 0 ? -2 : 2, (j & 0b01) == 0 ? -2 : 2);
				if(image.width() * point.x + offset.x < 0.0f ||
				   image.height() * point.y + offset.y < 0.0f) continue;
				image.drawFilledCircle(static_cast<uint32_t>(image.width() * point.x + offset.x), static_cast<uint32_t>(image.height() * point.y + offset.y), 3, color);
			}
		}
	}

	// const auto& tJoinVertices = qtree.getTJointVerticesIndex();
	// for(uint32_t vertId : tJoinVertices)
	// {
	// 	const glm::vec2 point = qtree.getVertices()[vertId];
	// 	image.drawFilledCircle(static_cast<uint32_t>(image.width() * point.x), static_cast<uint32_t>(image.height() * point.y), 5, glm::vec3(0.5f, 0.5f, 0.5f));
	// }

	// image.drawFilledCircle(static_cast<uint32_t>(image.width() * 0), static_cast<uint32_t>(image.height() * 0), 4, glm::vec3(1.0f, 0.0f, 0.0f));
}

void drawCovField(Image& image, LinearNodeTree<2>& scalarField,
				  std::optional<std::reference_wrapper<PointCloud<2>>> cloud = std::optional<std::reference_wrapper<PointCloud<2>>>())
{
	const NodeTree<2>& qtree = scalarField.getNodeTree();

	const std::vector<glm::vec3> sdfPalette = {
		glm::vec3(0.0f, 0.0f, 1.0f), 
		glm::vec3(0.0f, 0.5f, 1.0f), 
		glm::vec3(0.0f, 1.0f, 1.0f), 
		glm::vec3(1.0f, 1.0f, 1.0f), 
		glm::vec3(1.0f, 1.0f, 0.0f), 
		glm::vec3(1.0f, 0.5f, 0.0f), 
		glm::vec3(1.0f, 0.0f, 0.0f)
	};

	std::vector<glm::vec3> viridisPalette = {
		glm::vec3(68.0f/255.0f, 1.0f/255.0f, 84.0f/255.0f),
		glm::vec3(65.0f/255.0f, 68.0f/255.0f, 135.0f/255.0f),
		glm::vec3(42.0f/255.0f, 120.0f/255.0f, 142.0f/255.0f),
		glm::vec3(34.0f/255.0f, 163.0f/255.0f, 132.0f/255.0f),
		glm::vec3(122.0f/255.0f, 209.0f/255.0f, 81.0f/255.0f),
		glm::vec3(253.0f/255.0f, 231.0f/255.0f, 37.0f/255.0f)
	};

	const std::vector<glm::vec3> redPalette = {
		glm::vec3(1.0f, 1.0f, 1.0f), 
		glm::vec3(1.0f, 0.0f, 0.0f)
	};

	auto op = [](float val) { return glm::sqrt(glm::abs(val)); };

	float maxValue = op(scalarField.getMaxAbsValue());
	const float minValue = op(scalarField.getMinAbsValue());
	std::cout << "min " << minValue << " // max " << maxValue << std::endl;
	maxValue = 0.99 * maxValue;
	ScalarFieldRender::renderScalarField(scalarField, image, 
										 [minValue, maxValue, &op](float val) { return (op(val) - minValue) / (maxValue - minValue); }, 
										 viridisPalette,
										 0.0f, 0.0f, 1.0f, 
										 0.0f, 0.001f, 0.8f);

	if(cloud)
	{
		float min = INFINITY;
		float max = -INFINITY;
		for(float v : cloud->get().getVariances())
		{
			min = glm::min(min, v); max = glm::max(max, v);
		}

		if(max - min < 1e-6) max = min + 1.0f;

		glm::vec2 minCoord = scalarField.getMinCoord();
		glm::vec2 size = scalarField.getMaxCoord() - scalarField.getMinCoord();
		for (uint32_t i = 0; i < cloud->get().size(); i++)
		{
			glm::vec2 point = (cloud->get().point(i) - minCoord) / size;
			const float colorVal = (cloud->get().variance(i) - min) / (max - min);
			image.drawFilledCircle(static_cast<uint32_t>(image.width() * point.x), static_cast<uint32_t>(image.height() * point.y), 10.0f, glm::mix(redPalette[0], redPalette[1], colorVal));

			image.drawCircle(static_cast<uint32_t>(image.width() * point.x), static_cast<uint32_t>(image.height() * point.y),
							 3.0 * glm::sqrt(cloud->get().variance(i)), 5.0f, glm::mix(redPalette[0], redPalette[1], colorVal));
		}
	}
}

void drawCollisionField(Image& image, LinearNodeTree<2>& muField, LinearNodeTree<2>& varField, bool printDensity,
				        std::optional<std::reference_wrapper<PointCloud<2>>> cloud = std::optional<std::reference_wrapper<PointCloud<2>>>())
{
	const NodeTree<2>& qtree = muField.getNodeTree();

	std::vector<glm::vec3> viridisPalette = {
		glm::vec3(68.0f/255.0f, 1.0f/255.0f, 84.0f/255.0f),
		glm::vec3(65.0f/255.0f, 68.0f/255.0f, 135.0f/255.0f),
		glm::vec3(42.0f/255.0f, 120.0f/255.0f, 142.0f/255.0f),
		glm::vec3(34.0f/255.0f, 163.0f/255.0f, 132.0f/255.0f),
		glm::vec3(122.0f/255.0f, 209.0f/255.0f, 81.0f/255.0f),
		glm::vec3(253.0f/255.0f, 231.0f/255.0f, 37.0f/255.0f)
	};

	const std::vector<glm::vec3> redPalette = {
		glm::vec3(1.0f, 1.0f, 1.0f), 
		glm::vec3(1.0f, 0.0f, 0.0f)
	};

	auto cdf = [](float x, float mu, float std)
	{
		return 0.5f * (1.0f + glm::abs(erf((x - mu) / (glm::sqrt(2.0f) * std))));
	};

	auto op = [&](glm::vec2 pos) 
	{
		float mu = muField.eval(pos);
		float std = glm::sqrt(glm::abs(varField.eval(pos)));
		if(printDensity)
		{
			return 1.0f / (glm::sqrt(2.0f * static_cast<float>(std::numbers::pi)) * std) * glm::exp(-0.5f * mu * mu / (std * std));
		}
		else
		{
			if(mu > 0.0f)
			{
				return 1.0f - cdf(0.0f, mu, std);
			}
			else
			{
				return cdf(0.0f, mu, std);
			}
		}
	};

	ScalarFieldRender::renderColorField(muField, op, image, viridisPalette);

	if(cloud)
	{
		float min = INFINITY;
		float max = -INFINITY;
		for(float v : cloud->get().getVariances())
		{
			min = glm::min(min, v); max = glm::max(max, v);
		}

		if(max - min < 1e-6) max = min + 1.0f;

		glm::vec2 minCoord = muField.getMinCoord();
		glm::vec2 size = muField.getMaxCoord() - muField.getMinCoord();
		for (uint32_t i = 0; i < cloud->get().size(); i++)
		{
			glm::vec2 point = (cloud->get().point(i) - minCoord) / size;
			const float colorVal = (cloud->get().variance(i) - min) / (max - min);
			image.drawFilledCircle(static_cast<uint32_t>(image.width() * point.x), static_cast<uint32_t>(image.height() * point.y), 3.0f, glm::mix(redPalette[0], redPalette[1], colorVal));
			image.drawCircle(static_cast<uint32_t>(image.width() * point.x), static_cast<uint32_t>(image.height() * point.y),
							 image.width() * 3.0 * glm::sqrt(cloud->get().variance(i)) / size[0], 3.0f, glm::mix(redPalette[0], redPalette[1], colorVal));
		}
	}
}
	

// template <typename T>
// void write_array_to_file(const std::vector<T>& arr, const std::string& filename) {
//   std::ofstream file(filename, std::ios::binary);
//   if (!file.is_open()) {
//     throw std::runtime_error("Failed to open file for writing.");
//   }

//   // Write the size of the array as the first element
//   size_t size = arr.size();
//   file.write(reinterpret_cast<const char*>(&size), sizeof(size));

//   // Write each element of the array
//   for (const T& element : arr) {
//     file.write(reinterpret_cast<const char*>(&element), sizeof(element));
//   }

//   file.close();
// }

std::vector<float> read_array_from_txt_file(const std::string& file_path) {
	std::ifstream file(file_path);
	if (!file.is_open()) {
		std::cerr << "Error opening file: " << file_path << std::endl;
		return {};
	}

	std::vector<float> array;
	std::string line;

	while (std::getline(file, line)) 
	{
		array.push_back(std::stof(line));
	}

	return array;
}

void draw1DField(LinearNodeTree<2>& muField, LinearNodeTree<2>& varField)
{
	const std::array<glm::vec3, 2> palette = {
		glm::vec3(1.0f, 1.0f, 1.0f),
		glm::vec3(1.0f, 0.0f, 0.0f)
	};

	glm::vec2 minCoord = muField.getMinCoord();
	glm::vec2 size = muField.getMaxCoord() - muField.getMinCoord();
	glm::vec2 invSize = 1.0f / size;
	int width = 2048;
	int height = 2048;
	float invSizeMag = glm::length(glm::vec2(width, height) * invSize);
	float winSizeMag = glm::length(glm::vec2(width, height));	
	const float maxValue = muField.getMaxAbsValue();
	uint32_t i = static_cast<uint32_t>(0.7f * static_cast<float>(width));

	std::vector<float> muVec;
	std::vector<float> varVec;
	for(uint32_t j=0; j < height; j++)
	{
		glm::vec2 pos = glm::vec2((static_cast<float>(i)+0.5f) / static_cast<float>(width), 
									(static_cast<float>(j)+0.5f) / static_cast<float>(height));
		pos = minCoord + pos * size;

		float mu = muField.eval(pos);
		float var = varField.eval(pos);
		muVec.push_back(mu);
		varVec.push_back(var);
	}

	std::cout << muVec.size() << std::endl;
	write_array_to_file(muVec, "muVec.bin");
	write_array_to_file(varVec, "varVec.bin");
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
	
	if(!cloud.readFromFile(argv[1], true))
	{
		std::cout << "Input cloud could not be read!" << std::endl;
		return -1;
	}

	for(glm::vec2& pos : cloud.getPoints())
	{
		pos += glm::vec2(0.01f);
	}

	glm::vec2 min(INFINITY);
    glm::vec2 max(-INFINITY);
    for(glm::vec2 p : cloud.getPoints())
    {
        for(uint32_t i=0; i < 2; i++)
        {
            min[i] = glm::min(min[i], p[i]);
            max[i] = glm::max(max[i], p[i]);
        }
    }

    const glm::vec2 size = max - min;
    const glm::vec2 center = 0.5f * (max + min);
    float maxSize = glm::max(size.x, size.y);
    // Add margin
    maxSize = 1.2f * maxSize;

	const uint32_t maxDepth = 6;
	NodeTree<2>::Config quadConfig = {
		// .minCoord = NodeTree<2>::vec(0.0f),
		// .maxCoord = NodeTree<2>::vec(1.0f),
		.minCoord = center - glm::vec2(0.5f * maxSize),
		.maxCoord = center + glm::vec2(0.5f * maxSize),
		.pointFilterMaxDistance = 1.23f * maxSize / static_cast<float>(1 << maxDepth),
		.constraintNeighbourNodes = true,
		.maxDepth = maxDepth
	};

	NodeTree<2> quad;
	quad.compute(cloud, quadConfig);

	// std::vector<float> values = read_array_from_txt_file("./test/data.txt");
	// std::vector<float> vValues(values.size());
	// for(uint32_t i=0; i < 64; i++)
	// {
	// 	for(uint32_t j=0; j < 64; j++)
	// 	{
	// 		glm::vec2 p(float(i)/64.0f + 1e-6, float(j)/64.0f + 1e-6);
	// 		std::optional<NodeTree<2>::Node> node;
	// 		quad.getNode(p, node);
	// 		if(node)
	// 		{
	// 			vValues[node->controlPointsIdx[0]] = values[j * 65 + i];
	// 		}
	// 	}
	// }

	// std::unique_ptr<LinearNodeTree<2>> scalarField = std::make_unique<LinearNodeTree<2>>(std::move(quad), std::move(vValues));

	SmoothSurfaceReconstruction::Config<2> config = {
		.posWeight = 1.0f, 
        .gradientWeight = 1/0.5f,
        .smoothWeight = 1/1.0f,
		.algorithm = SmoothSurfaceReconstruction::Algorithm::VAR,
		.computeVariance = true
	};

	// PoissonReconstruction::Config<2> config = {};
	
	std::vector<float> eigenValues;
	std::vector<float> verticesEnergy;
	// std::unique_ptr<CubicNodeTree<2>> scalarField = 
	// 	SmoothSurfaceReconstruction::compute2DCubicNodeTree<2>(cloud, config, std::nullopt, verticesEnergy, eigenValues);

	// std::unique_ptr<CubicNodeTree<2>> scalarField = 
	// 	SmoothSurfaceReconstruction::compute2DCubicNodeTree<2>(std::move(quad), cloud, config, verticesEnergy, eigenValues);
		
	// SmoothSurfaceReconstruction::computeCubicNodeLoss<2>(*scalarField, cloud, config);

	std::optional<LinearNodeTree<2>> covScalarField;
	std::optional<Eigen::MatrixXd> invCovMat;
	std::optional<Eigen::MatrixXd> covMat;
	std::optional<Eigen::SparseMatrix<double>> matP;
	std::optional<Eigen::SparseMatrix<double>> matN;
	std::optional<Eigen::SparseMatrix<double>> matS;
	std::optional<Eigen::VectorXd> vecW;

	std::unique_ptr<LinearNodeTree<2>> scalarField = 
		SmoothSurfaceReconstruction::computeLinearNodeTree<2>(std::move(quad), cloud, config, covScalarField, invCovMat, covMat, matP, matN, matS, vecW);

	// std::unique_ptr<LinearNodeTree<2>> scalarField = 
	// 	GPReconstruction::computeLinearNodeTree<2>(std::move(quad), cloud, covScalarField);


	// Eigen::JacobiSVD<Eigen::MatrixXd> svd(covMat.value(), Eigen::ComputeThinU | Eigen::ComputeThinV);
	// Eigen::VectorXd sv = svd.singularValues();
	// uint32_t numZeros = 0;
	// for(uint32_t i=0; i < sv.size(); i++)
	// {
	// 	if(sv(i) > 1e-9)
	// 	{
	// 		sv(i) = 1.0f / sv(i);
	// 	}
	// 	else
	// 	{
	// 		numZeros++;
	// 		sv(i) = 0.0f;
	// 	}
	// }
	// std::cout << "num zeros " << numZeros << std::endl;
	// Eigen::MatrixXd invCovMat = svd.matrixV() * sv.asDiagonal() * svd.matrixU().adjoint();

	std::cout << "End compute" << std::endl;

	// Eigen::LLT<Eigen::MatrixXd> covLLT(covMat.value());
	// Eigen::MatrixXd L = covLLT.matrixL();
	// std::cout << "End compute Cholensky" << std::endl;

	// Eigen::VectorXd basicGaussianValues(vecW.value().rows());

	// std::random_device rd{};
    // std::mt19937 gen{rd()};

	// std::normal_distribution gaussianSampler;

	// std::vector<double> sumW(vecW.value().rows(), 0.0);
	// auto vSumW = Eigen::Map<Eigen::VectorXd>(sumW.data(), sumW.size());

	// std::cout << "Start sampling" << std::endl;
	// // const uint32_t numSamples = 2048000;
	// const uint32_t numSamples = 20480;
	// double sumWeights = 0.0;
	// std::vector<double> differences;
	// double mg = 0.0;
	// double mp = 0.0;
	// double mw = 0.0;
	// double mw2 = 0.0;
	// double imw = 0.0;
	// double imw2 = 0.0;
	// for(uint32_t s = 0; s < numSamples; s++)
	// {
	// 	for(uint32_t i=0; i < basicGaussianValues.rows(); i++)
	// 	{
	// 		basicGaussianValues(i) = gaussianSampler(gen);
	// 	}

	// 	const double rstd = 1.0;
	// 	const double rinvcov = 1.0 / (rstd * rstd);
	// 	Eigen::VectorXd newW = vecW.value() + rstd * L * basicGaussianValues;
	// 	double g = 1.54203e201 * glm::exp(-0.5 * (newW - vecW.value()).transpose() * rinvcov * invCovMat.value() * (newW - vecW.value()));
	// 	// double g = 5.78e202 * glm::exp(-0.5 * (newW - vecW.value()).transpose() * rinvcov * invCovMat.value() * (newW - vecW.value()));
	// 	mg += g / static_cast<double>(numSamples);
	// 	double p = 1.54203e207 * SmoothSurfaceReconstruction::evaulatePosteriorFunc(cloud, config, matP.value(), matN.value(), matS.value(), newW);
	// 	// double p = 10474275180.2 * SmoothSurfaceReconstruction::evaulatePosteriorFunc(cloud, config, matP.value(), matN.value(), matS.value(), newW);
	// 	mp += p / static_cast<double>(numSamples);
	// 	double weight = p / g;
	// 	mw += weight / static_cast<double>(numSamples);
	// 	mw2 += weight * weight / static_cast<double>(numSamples);
	// 	double iweight = g / p;
	// 	imw += iweight / static_cast<double>(numSamples);
	// 	imw2 += iweight * iweight / static_cast<double>(numSamples);
	// 	if(glm::isnan(weight)) continue;
	// 	auto diff = (newW - vecW.value());
	// 	for(uint32_t i=0; i < newW.rows(); i++)
	// 	{
	// 		vSumW(i) += weight * diff(i) * diff(i);
	// 	}
	// 	sumWeights += weight;
	// 	if(std::any_of(sumW.begin(), sumW.end(), [] (double v) { return glm::isnan(v); }))
	// 	{
	// 		std::cout << "nan " << std::endl;
	// 	}
	// 	// std::cout << s  << " " << weight << ", "; 

	// 	if(s % 100 == 0)
	// 	{
	// 		double d = 0.0;
	// 		for(uint32_t i=0; i < newW.rows(); i++)
	// 		{
	// 			const double v = (vSumW(i) / sumWeights - covMat.value()(i, i));
	// 			d += v * v;
	// 		}
	// 		differences.push_back(d);

	// 		double as = 0.0;
	// 		for(uint32_t i=0; i < newW.rows(); i++)
	// 		{
	// 			const double v = (vSumW(i) - covMat.value()(i, i));
	// 			as += glm::abs(v) / static_cast<double>(newW.rows());
	// 		}

	// 		if(s % 10000 == 0 || d > 1e10)
	// 		{
	// 			std::cout << s  << ": " << as << " // " << sumWeights  << " // " << d << std::endl;
	// 		}
	// 	}
	// }

	// std::cout << std::endl;

	// std::cout << mg << " // " << mp << std::endl;
	// std::cout << mw << " // " << glm::sqrt(mw2 - mw * mw)  << std::endl;
	// std::cout << imw << " // " << glm::sqrt(imw2 - imw * imw)  << std::endl;

	// std::cout << "End sampling" << std::endl;

	// vSumW = vSumW / sumWeights;

	// std::vector<float> fSumW(sumW.size());
	// for(uint32_t i=0; i < fSumW.size(); i++)
	// {
	// 	fSumW[i] = sumW[i];
	// }

	// covScalarField = LinearNodeTree<2>(std::move(covScalarField->getNodeTree()), std::move(fSumW));

	// write_array_to_file(differences, "diff2.bin");

	// NodeTree<2>::Config quadConfig2 = {
	// 	.minCoord = NodeTree<2>::vec(0.0f),
	// 	.maxCoord = NodeTree<2>::vec(1.0f),
	// 	.pointFilterMaxDistance = 0.0f / static_cast<float>(1 << maxDepth),
	// 	//.pointFilterMaxDistance = 0.0f,
	// 	.constraintNeighbourNodes = true,
	// 	.maxDepth = maxDepth
	// };

	// NodeTree<2> quad;
	// quad.compute(cloud, quadConfig2);

	// std::unique_ptr<CubicNodeTree<2>> scalarField = std::make_unique<CubicNodeTree<2>>(std::move(quad), *scalarField1);

	// config.nodeTreeConfig = quadConfig2;
	// SmoothSurfaceReconstruction::computeCubicNodeLoss<2>(*scalarField, cloud, config);

	// std::unique_ptr<LinearNodeTree<2>> scalarField = 
	// 	SmoothSurfaceReconstruction::computeLinearNodeTree<2, EigenSquareSolver>(cloud, config);
	
	std::cout << "print image" << std::endl;
	Image image;
	image.init(2048, 2048);
	drawScalarField(image, *scalarField, cloud, true);
	image.savePNG("quadtree.png");

	Image cimage;
	cimage.init(2048, 2048);
	drawCovField(cimage, *covScalarField, cloud);
	cimage.savePNG("quadtreeCov.png");

	{
		Image colimage;
		colimage.init(2048, 2048);
		drawCollisionField(colimage, *scalarField, *covScalarField, false, cloud);
		colimage.savePNG("quadtreePInside.png");
	}

	{
		Image colimage;
		colimage.init(2048, 2048);
		drawCollisionField(colimage, *scalarField, *covScalarField, true, cloud);
		colimage.savePNG("quadtreePSurface.png");
	}

	Image imageG;
	imageG.init(2048, 2048);
	drawGradientMaginitudeImage(imageG, *scalarField);
	imageG.savePNG("quadtreeGrad.png"); 

	// draw1DField(*scalarField, *covScalarField);

	return 0;
}