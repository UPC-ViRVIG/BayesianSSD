#ifndef SCALAR_FIELD_H
#define SCALAR_FIELD_H

#include <vector>
#include <optional>
#include <cereal.hpp>
#include "Vector.h"
#include "NodeTree.h"
#include "InterpolationMethod.h"
#include <cereal/types/vector.hpp>

template<uint32_t Dim>
class ScalarField
{
public:
	using vec = VStruct<Dim>::type;

	virtual float eval(vec point) = 0;
	virtual vec getMinCoord() = 0;
	virtual vec getMaxCoord() = 0;
};

template<uint32_t Dim>
class LinearNodeTree : public ScalarField<Dim>
{
public:
	using vec = ScalarField<Dim>::vec;

	LinearNodeTree(NodeTree<Dim>&& nodeTree, std::vector<float>&& verticesValues) :
		nodeTree(nodeTree), vValues(verticesValues) {}

	float eval(vec point)
	{
		using Inter = MultivariateLinearInterpolation<Dim>;
		std::optional<NodeTree<Dim>::Node> node;
		nodeTree.getNode(point, node);
		if(!node)
		{
			std::cout << "out of bounds" << std::endl;
			return 100.0f;
		}

		std::array<float, Inter::NumControlPoints> weigths;
		Inter::eval(node->transformToLocalCoord(point), weigths);
		float res = 0.0f;
		for(uint32_t i = 0; i < Inter::NumControlPoints; i++)
		{
			res += vValues[node->controlPointsIdx[i]] * weigths[i];
		}
		return res;
	}

	vec getMinCoord() { return nodeTree.getMinOctreeCoord(); }
	vec getMaxCoord() { return nodeTree.getMaxOctreeCoord(); }

	NodeTree<Dim>& getNodeTree() { return nodeTree; }

	float getMaxAbsValue()
	{
		float absMax = 0.0f;
		for(float v : vValues) absMax = glm::max(absMax, glm::abs(v));
		return absMax;
	}

private:
	NodeTree<Dim> nodeTree;
	std::vector<float> vValues;
};

#endif 


