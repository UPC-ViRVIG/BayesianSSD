#include <iostream>
#include <vector>
#include <fstream>
#include "SmoothSurfaceReconstruction.h"
#include "PoissonReconstruction.h"
#include "EigenSquareSolver.h"
#include "MyImage.h"

template<typename SF>
void drawScalarField(Image& image, SF& scalarField, 
					 std::optional<std::reference_wrapper<PointCloud<2>>> cloud = std::optional<std::reference_wrapper<PointCloud<2>>>(), 
					 bool drawGrid = true,
					 std::optional<std::reference_wrapper<std::vector<float>>> verticesEnergy = std::optional<std::reference_wrapper<std::vector<float>>>())
{
	const NodeTree<2>& qtree = scalarField.getNodeTree();

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
	glm::vec2 minCoord = scalarField.getMinCoord();
	glm::vec2 size = scalarField.getMaxCoord() - scalarField.getMinCoord();
	for(uint32_t i=0; i < image.width(); i++)
	{
		for(uint32_t j=0; j < image.height(); j++)
		{
			glm::vec2 pos = glm::vec2((static_cast<float>(i)+0.5f) / static_cast<float>(image.width()), 
									  (static_cast<float>(j)+0.5f) / static_cast<float>(image.height()));
			pos = minCoord + pos * size;

			// Field color
			const float pval = scalarField.eval(pos);
			float val = glm::clamp(pval / maxValue, -1.0f, 0.999999f);
			val = static_cast<float>(palette.size() - 1) * 0.5f * (val + 1.0f);
			uint32_t cid = glm::floor(val);
			glm::vec3 bgColor = glm::mix(palette[cid], palette[cid+1], glm::fract(val));

			// Gradient color
			// const glm::vec2 grad = scalarField.evalGrad(pos);
			// glm::vec3 bgColor = glm::vec3(glm::clamp(glm::abs(grad.x), 0.0f, 0.99999f), glm::clamp(glm::abs(grad.y), 0.0f, 0.99999f), 0.0f);

			float aval = glm::abs(pval / maxValue);
			// Isosurface line
			float surfaceColorWeight = glm::clamp(1.0 - glm::pow(aval/0.004f, 12), 0.0, 1.0);
			// float surfaceColorWeight = glm::clamp(1.0 - glm::pow(aval/0.01f, 12), 0.0, 1.0);

			// Isolines
			float valToLevel = 0.5f - glm::abs(glm::fract(aval / 0.03f) - 0.5);
			float linesColorWeight = glm::clamp(1.0 - glm::pow(valToLevel * 0.03f/0.0015f, 12), 0.0, 1.0);
			// linesColorWeight = 0.0f;

			// Grid color
			float gridColorWeight = 0.0f;
			if (drawGrid) 
			{
				std::optional<NodeTree<2>::Node> node;
				qtree.getNode(pos, node);
				if(!node) continue;
				glm::vec2 invSize = 1.0f / (node->maxCoord - node->minCoord);
				glm::vec2 np = (pos - node->minCoord) * invSize;
				np = glm::abs(2.0f * (np - glm::vec2(0.5f)));
				// invSize *= 100.0f;
				gridColorWeight = 0.8f * glm::smoothstep(1.0f - 0.005f * invSize.x, 1.0f - 0.003f * invSize.x, glm::max(np.x, np.y));
			}

			image(i, j) = glm::mix(bgColor, glm::vec3(0.0f), glm::max(glm::max(surfaceColorWeight, linesColorWeight), gridColorWeight));
		}
	}

	if(cloud)
	{
		for (glm::vec2 point : cloud->get().getPoints())
		{
			point = (point-minCoord) / size;
			image.drawFilledCircle(static_cast<uint32_t>(image.width() * point.x), static_cast<uint32_t>(image.height() * point.y), 10, glm::vec3(0.35f, 0.35f, 0.35f));
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

	glm::vec2 min(INFINITY);
    glm::vec2 max(-INFINITY);
    for(glm::vec2 p : cloud.getPoints())
    {
        for(uint32_t i=0; i < 3; i++)
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

	const uint32_t maxDepth = 5;
	NodeTree<2>::Config quadConfig = {
		// .minCoord = NodeTree<2>::vec(0.0f),
		// .maxCoord = NodeTree<2>::vec(1.0f),
		.minCoord = center - glm::vec2(0.5f * maxSize),
		.maxCoord = center + glm::vec2(0.5f * maxSize),
		.pointFilterMaxDistance = 100.33f * maxSize / static_cast<float>(1 << maxDepth),
		//.pointFilterMaxDistance = 0.0f,
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
		.posWeight = 200.0f, 
        .gradientWeight = 200.0f,
        .smoothWeight = 1.0f
	};

	// PoissonReconstruction::Config<2> config = {};
	
	std::vector<float> eigenValues;
	std::vector<float> verticesEnergy;
	// std::unique_ptr<CubicNodeTree<2>> scalarField = 
	// 	SmoothSurfaceReconstruction::compute2DCubicNodeTree<2>(cloud, config, std::nullopt, verticesEnergy, eigenValues);

	std::unique_ptr<CubicNodeTree<2>> scalarField = 
		SmoothSurfaceReconstruction::compute2DCubicNodeTree<2>(std::move(quad), cloud, config, verticesEnergy, eigenValues);

	// std::unique_ptr<LinearNodeTree<2>> scalarField = 
	// 	PoissonReconstruction::computeLinearNodeTree<2>(std::move(quad), cloud, config);

	SmoothSurfaceReconstruction::computeCubicNodeLoss<2>(*scalarField, cloud, config);

	write_array_to_file(eigenValues, "eigenValues.bin");

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

	// Image imageG;
	// imageG.init(2048, 2048);
	// drawGradientMaginitudeImage(imageG, *scalarField);
	// imageG.savePNG("quadtreeGrad.png"); 

	return 0;
}