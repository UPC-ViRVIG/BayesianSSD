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
#include <json_struct/json_struct.h>

#include <Eigen/Cholesky>

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
	std::string inverseMethod;
	uint32_t baseRedRank;
	uint32_t octreeRedDepth;
	float mulPointsStd;
	bool computeNormals;
	uint32_t normalsNumNearPoints;
	float normalsDistanceFactor;
	float defaultNormalsStd;

	JS_OBJ(pointCloudName, outputName, bbMargin, octreeMaxDepth, octreeSubRuleInVoxels, gradientStd, smoothnessStd, computeVariance, 
		   inverseMethod, baseRedRank, octreeRedDepth, mulPointsStd, 
		   computeNormals, normalsNumNearPoints, normalsDistanceFactor, defaultNormalsStd);
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

	// const std::vector<glm::vec3> sdfPalette = {
	// 	glm::vec3(0.0f, 0.0f, 1.0f), 
	// 	glm::vec3(0.0f, 0.5f, 1.0f), 
	// 	glm::vec3(0.0f, 1.0f, 1.0f), 
	// 	glm::vec3(1.0f, 1.0f, 1.0f), 
	// 	glm::vec3(1.0f, 1.0f, 0.0f), 
	// 	glm::vec3(1.0f, 0.5f, 0.0f), 
	// 	glm::vec3(1.0f, 0.0f, 0.0f)
	// };

	const std::vector<glm::vec3> sdfPalette = {
		0.5f * glm::vec3(0.0f, 0.0f, 1.0f) + 0.45f * glm::vec3(1.0f), 
		0.5f * glm::vec3(0.0f, 0.5f, 1.0f) + 0.45f * glm::vec3(1.0f), 
		0.5f * glm::vec3(0.0f, 1.0f, 1.0f) + 0.45f * glm::vec3(1.0f), 
		0.5f * glm::vec3(1.0f, 1.0f, 1.0f) + 0.45f * glm::vec3(1.0f), 
		0.5f * glm::vec3(1.0f, 1.0f, 0.0f) + 0.45f * glm::vec3(1.0f), 
		0.5f * glm::vec3(1.0f, 0.5f, 0.0f) + 0.45f * glm::vec3(1.0f), 
		0.5f * glm::vec3(1.0f, 0.0f, 0.0f) + 0.45f * glm::vec3(1.0f)
	};
	
	const float maxValue = scalarField.getMaxAbsValue();
	// ScalarFieldRender::renderScalarField(scalarField, image, [maxValue](float val) { return 0.5f * (1.0f + val / maxValue); }, sdfPalette,
	// 									 1.0f, 0.0f, 1.0f, 
	// 									 0.5f, 0.007f, 0.8f);

	auto op = [&](float value) 
	{
		value = value / maxValue;
		if(value > 0.0)
		{
			return 4.f / 6.f + 3.f * value / 9.0f;
		}
		else
		{
			return 2.f / 6.f+3.f * value / 9.0f;
		}
	};

	// ScalarFieldRender::renderScalarField(scalarField, image, op, sdfPalette,
	// 									 14.0f, 0.0f, 1.0f, 
	// 									 0.9f, 0.011f, 0.6f);

	ScalarFieldRender::renderScalarField(scalarField, image, op, sdfPalette,
										22.0f, 0.0f, 1.0f, 
										8.0f, 0.008f, 0.6f);


	if(drawGrid)
	{
		ScalarFieldRender::renderNodeTreeGrid(qtree, image, 10.0f, 0.6f);
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
			// image.drawFilledCircle(static_cast<uint32_t>(image.width() * point.x), static_cast<uint32_t>(image.height() * point.y), 10, glm::vec3(0.35f, 0.35f, 0.35f));
			image.drawFilledCircle(static_cast<uint32_t>(image.width() * point.x), static_cast<uint32_t>(image.height() * point.y), 5.0f, glm::vec3(0.35f, 0.35f, 0.35f));
			
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

			// image.drawCircle(static_cast<uint32_t>(image.width() * point.x), static_cast<uint32_t>(image.height() * point.y),
			// 				 3.0 * glm::sqrt(cloud->get().variance(i)), 5.0f, glm::mix(redPalette[0], redPalette[1], colorVal));
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

	const std::vector<glm::vec3> magmaPalette = {
		glm::vec3(0.0f/255.0f, 0.0f/255.0f, 5.0f/255.0f),
		glm::vec3(26.0f/255.0f, 10.0f/255.0f, 64.0f/255.0f), 
		glm::vec3(75.0f/255.0f, 0.0f/255.0f, 108.0f/255.0f), 
		glm::vec3(132.0f/255.0f, 24.0f/255.0f, 109.0f/255.0f), 
		glm::vec3(198.0f/255.0f, 43.0f/255.0f, 91.0f/255.0f), 
		glm::vec3(243.0f/255.0f, 95.0f/255.0f, 74.0f/255.0f), 
		glm::vec3(252.0f/255.0f, 172.0f/255.0f, 109.0f/255.0f), 
		glm::vec3(251.0f/255.0f, 255.0f/255.0f, 178.0f/255.0f),
		glm::vec3(1.0f)
	};

	auto cdf = [](float x, float mu, float std)
	{
		return 0.5f * (1.0f + glm::abs(erf((x - mu) / (glm::sqrt(2.0f) * std))));
	};

	float minStd = 0.4;
	// for(float& v : varField.getVertexValues())
	// {
	// 	minStd = glm::min(v, minStd);
	// }
	// minStd = glm::sqrt(minStd);

	auto op = [&](glm::vec2 pos) 
	{
		float mu = muField.eval(pos);
		float std = glm::sqrt(glm::abs(varField.eval(pos)));
		if(printDensity)
		{
			float nStd = std / minStd;
			// return 1.0f / (glm::sqrt(2.0f * static_cast<float>(std::numbers::pi)) * std) * glm::exp(-0.5f * mu * mu / (std * std));
			return 1.0f / nStd * glm::exp(-0.5f * mu * mu / (std * std));
			// return glm::exp(-0.5f * mu * mu / (std * std));
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

	if(printDensity)
	{
		ScalarFieldRender::renderColorField(muField, [&](glm::vec2 pos) { return 1.0f - op(pos); }, image, magmaPalette);
	}
	else
	{
		ScalarFieldRender::renderColorField(muField, op, image, viridisPalette);
	}

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
			// image.drawCircle(static_cast<uint32_t>(image.width() * point.x), static_cast<uint32_t>(image.height() * point.y),
			// 				 image.width() * 3.0 * glm::sqrt(cloud->get().variance(i)) / size[0], 3.0f, glm::mix(redPalette[0], redPalette[1], colorVal));
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
	
	if(!cloud.readFromFile("./data/" + inConfig.pointCloudName + ".txt", true))
	{
		std::cout << "Input cloud could not be read!" << std::endl;
		return -1;
	}

	for(float& v : cloud.getVariances())
	{
		v *= inConfig.mulPointsStd * inConfig.mulPointsStd;
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
	const float bbRadius = 0.5f * glm::distance(min, max);

	// Compute normals
	if(inConfig.computeNormals)
	{
		cloud.computeNormals(inConfig.normalsNumNearPoints, 0.0, 0.0, inConfig.normalsDistanceFactor / bbRadius);
	}
	else
	{
		cloud.fillNormalsData(inConfig.defaultNormalsStd);
	}


    // Add margin
    maxSize = (1.0f + inConfig.bbMargin) * maxSize;

	const uint32_t maxDepth = inConfig.octreeMaxDepth;
	NodeTree<2>::Config quadConfig = {
		// .minCoord = NodeTree<2>::vec(0.0f),
		// .maxCoord = NodeTree<2>::vec(1.0f),
		.minCoord = center - glm::vec2(0.5f * maxSize),
		.maxCoord = center + glm::vec2(0.5f * maxSize),
		.pointFilterMaxDistance = inConfig.octreeSubRuleInVoxels * maxSize / static_cast<float>(1 << maxDepth),
		.constraintNeighbourNodes = true,
		.maxDepth = maxDepth
	};

	std::cout << "Min: " << quadConfig.minCoord.x << ", " << quadConfig.minCoord.y << std::endl;
	std::cout << "Max: " << quadConfig.maxCoord.x << ", " << quadConfig.maxCoord.y << std::endl;
	std::cout << "Size: " << maxSize << std::endl;

	NodeTree<2> quad;
	quad.compute(cloud, quadConfig);

	std::cout << "Octree generated" << std::endl;

	SmoothSurfaceReconstruction::InverseAlgorithm invAlgorithm = SmoothSurfaceReconstruction::InverseAlgorithm::FULL;
	if(inConfig.inverseMethod == "base_red") invAlgorithm = SmoothSurfaceReconstruction::InverseAlgorithm::BASE_RED;
	const bool computeSimpleVariance = inConfig.computeVariance && inConfig.inverseMethod == "octree_red";

	SmoothSurfaceReconstruction::Config<2> config = {
		.posWeight = 1.0f, 
        .gradientWeight = 1.f/inConfig.gradientStd,
        .smoothWeight = 1.f/inConfig.smoothnessStd,
		.algorithm = SmoothSurfaceReconstruction::Algorithm::BAYESIAN,
		.computeVariance = inConfig.computeVariance,
		.invAlgorithm = invAlgorithm,
		.invRedMatRank = inConfig.baseRedRank
	};

	NodeTree<2> cquad;
	std::optional<LinearNodeTree<2>> covScalarField;

	if(computeSimpleVariance)
	{
		config.computeVariance = false;
		cquad = quad;
	}
	std::unique_ptr<LinearNodeTree<2>> scalarField = 
		SmoothSurfaceReconstruction::computeLinearNodeTree<2>(std::move(quad), cloud, config, covScalarField);

	if(computeSimpleVariance)
	{
		config.computeVariance = true;
		quadConfig.maxDepth = inConfig.octreeRedDepth;
		config.invAlgorithm = SmoothSurfaceReconstruction::InverseAlgorithm::FULL;
		NodeTree<2> quadS;
		quadS.compute(cloud, quadConfig);

		std::optional<LinearNodeTree<2>> covScalarFieldS;
		std::unique_ptr<LinearNodeTree<2>> scalarFieldS = 
			SmoothSurfaceReconstruction::computeLinearNodeTree<2>(std::move(quadS), cloud, config, covScalarFieldS);

		const auto& qVertices = scalarField->getNodeTree().getVertices();
		std::vector<float> verticesValue(qVertices.size());
		for(uint32_t i=0; i < qVertices.size(); i++)
		{
			verticesValue[i] = covScalarFieldS->eval(qVertices[i]);
		}
		covScalarField = LinearNodeTree<2>(std::move(cquad), std::move(verticesValue));
	}


	// std::unique_ptr<LinearNodeTree<2>> scalarField = 
	// 	GPReconstruction::computeLinearNodeTree<2>(std::move(quad), cloud, covScalarField);

	// Export point cloud
	cloud.writeToFile("./output/" + inConfig.outputName + "_input.txt");

	//return 0;
	// std::vector<float> muVector;
	// std::vector<float> varVector;
	// glm::vec2 minB = scalarField->getMinCoord();
	// glm::vec2 maxB = scalarField->getMaxCoord();
	// glm::vec2 centerB = 0.5f * (minB + maxB);
	// for(uint32_t i=0; i < 2048; i++)
	// {
	// 	centerB.x = minB.x + (maxB.x - minB.x) * static_cast<float>(i)/2048.0;
	// 	muVector.push_back(scalarField->eval(centerB));
	// 	varVector.push_back(covScalarField->eval(centerB));
	// }

	// write_array_to_file(muVector, "muLine.txt");
	// write_array_to_file(varVector, "varLine.txt");
	
	std::cout << "print image" << std::endl;
	Image image;
	image.init(2048, 2048);
	drawScalarField(image, *scalarField, cloud, true);
	image.savePNG("./output/" + inConfig.outputName + "_mu.png");

	if(inConfig.computeVariance)
	{
		Image cimage;
		cimage.init(2048, 2048);
		drawCovField(cimage, *covScalarField, cloud);
		cimage.savePNG("./output/" + inConfig.outputName + "_std.png");

		{
			Image colimage;
			colimage.init(2048, 2048);
			drawCollisionField(colimage, *scalarField, *covScalarField, false, cloud);
			colimage.savePNG("./output/" + inConfig.outputName + "_pIn.png");
		}

		{
			Image colimage;
			colimage.init(2048, 2048);
			drawCollisionField(colimage, *scalarField, *covScalarField, true, cloud);
			colimage.savePNG("./output/" + inConfig.outputName + "_pSur.png");
		}
	}

	// Image imageG;
	// imageG.init(2048, 2048);
	// drawGradientMaginitudeImage(imageG, *scalarField);
	// imageG.savePNG("quadtreeGrad.png"); 

	// draw1DField(*scalarField, *covScalarField);

	return 0;
}