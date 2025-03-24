#ifndef SCALAR_FIELD_H
#define SCALAR_FIELD_H

#include <functional>
#include <vector>
#include <optional>
#include "Vector.h"
#include "NodeTree.h"
#include "InterpolationMethod.h"
#include <cereal/types/vector.hpp>
#include "UsefullSerializations.h"

template<uint32_t Dim>
class ScalarField
{
public:
	using vec = VStruct<Dim>::type;

	virtual float eval(vec point) const = 0;
	virtual vec getMinCoord() = 0;
	virtual vec getMaxCoord() = 0;

	virtual vec evalGrad(vec point) const
	{
		// TODO: implement numeric gradiant estimation
		return vec(0.0f);
	}
};

template<uint32_t Dim>
class LinearNodeTree : public ScalarField<Dim>
{
public:
	using vec = ScalarField<Dim>::vec;

	LinearNodeTree(NodeTree<Dim>&& nodeTree, std::vector<float>&& verticesValues) :
		nodeTree(nodeTree), vValues(verticesValues) {}

	float eval(vec point) const
	{
		using Inter = MultivariateLinearInterpolation<Dim>;
		std::optional<typename NodeTree<Dim>::Node> node;
		nodeTree.getNode(point, node);
		if(!node)
		{
			//std::cout << "out of bounds" << std::endl;
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

	vec evalGrad(vec point) const override
	{
		using Inter = MultivariateLinearInterpolation<Dim>;
		std::optional<typename NodeTree<Dim>::Node> node;
		nodeTree.getNode(point, node);
		if(!node)
		{
			//std::cout << "out of bounds" << std::endl;
			return vec(0.0f);
		}

		std::array<std::array<float, Inter::NumControlPoints>, Dim> weigths;
		const vec nodeSize = node->maxCoord - node->minCoord;
		Inter::evalGrad(node->transformToLocalCoord(point), nodeSize, weigths);
		vec res;
		for(uint32_t d = 0; d < Dim; d++)
		{
			res[d] = 0.0f;
			for(uint32_t i = 0; i < Inter::NumControlPoints; i++)
			{
				res[d] += vValues[node->controlPointsIdx[i]] * weigths[d][i];
			}
		}
		return res;
	}

	vec getMinCoord() { return nodeTree.getMinCoord(); }
	vec getMaxCoord() { return nodeTree.getMaxCoord(); }

	NodeTree<Dim>& getNodeTree() { return nodeTree; }
	std::vector<float>& getVertexValues() { return vValues; }

	float getMaxAbsValue() const
	{
		float absMax = 0.0f;
		for(float v : vValues) absMax = glm::max(absMax, glm::abs(v));
		return absMax;
	}

	float getMinAbsValue() const
	{
		float absMin = 0.0f;
		for(float v : vValues) absMin = glm::min(absMin, glm::abs(v));
		return absMin;
	}

	float getMaxValue() const
	{
		float max = -INFINITY;
		for(float v : vValues) max = glm::max(max, v);
		return max;
	}

	float getMinValue() const
	{
		float min = INFINITY;
		for(float v : vValues) min = glm::min(min, v);
		return min;
	}

	// EXPORT FIELD AS IN SDFLIB
	struct BoundingBox
	{
		glm::vec3 min;
    	glm::vec3 max;
		template<class Archive>
		void serialize(Archive & archive)
		{
			archive(min, max); 
		}
	};

	struct OctreeNode
    {
        static constexpr uint32_t IS_LEAF_MASK = 1 << 31;
        static constexpr uint32_t MARK_MASK = 1 << 30;
        static constexpr uint32_t CHILDREN_INDEX_MASK = ~(IS_LEAF_MASK | MARK_MASK);
        union
        {
            uint32_t childrenIndex;
            float value;
        };

        inline void setValues(bool isLeaf, uint32_t index)
        {
            childrenIndex = (index & CHILDREN_INDEX_MASK) | 
                            ((isLeaf) ? IS_LEAF_MASK : 0);
        }

        template <class Archive>
        void serialize( Archive & ar )
        {
            ar(childrenIndex);
        }
    };

	enum SdfFormat
    {
        GRID,
        TRILINEAR_OCTREE,
        TRICUBIC_OCTREE,
        EXACT_OCTREE,
        NONE
    };
	template<class Archive>
    void save(Archive & archive) const
    { 
		if(Dim != 3) return;
		SdfFormat format = SdfFormat::TRILINEAR_OCTREE;
		BoundingBox mBox = {nodeTree.getMinCoord(), nodeTree.getMaxCoord()};
		int mStartGridSize = 1;
		uint32_t mMaxDepth = nodeTree.getMaxDepth();
		bool mSdfOnlyAtSurface = false;
		float mValueRange = getMaxAbsValue();
		float mMinBorderValue = 0.0f;
		std::vector<OctreeNode> mOctreeData(1);
		using Node = NodeTree<Dim>::InternalNode; 
		const std::vector<Node>& octree = nodeTree.octreeData;

		std::function<void(const Node&, uint32_t)> processNode;
		processNode = [&](const Node& node, uint32_t nodeIndex)
		{
			if(!node.isLeaf())
			{
				const uint32_t startIndex = mOctreeData.size();
				mOctreeData.resize(mOctreeData.size() + 8);
				mOctreeData[nodeIndex].setValues(false, startIndex);
				for(uint32_t i=0; i < 8; i++)
				{
					processNode(octree[node.getChildrenIndex() + i], startIndex + i);
				}
			}
			else
			{
				const uint32_t startIndex = mOctreeData.size();
				mOctreeData.resize(mOctreeData.size() + 8);
				mOctreeData[nodeIndex].setValues(true, startIndex);
				for(uint32_t i=0; i < 8; i++)
				{
					mOctreeData[startIndex + i].value = vValues[node.controlPointsIdx[i]];
				}
			}
		};

		processNode(octree[0], 0);
		archive(format);
        archive(mBox, mStartGridSize, mMaxDepth, mSdfOnlyAtSurface, mValueRange, mMinBorderValue, mOctreeData);
    }

private:
	NodeTree<Dim> nodeTree;
	std::vector<float> vValues;
};

template<uint32_t Dim>
class CubicNodeTree : public ScalarField<Dim>
{
public:
	using vec = ScalarField<Dim>::vec;

	CubicNodeTree(NodeTree<Dim>&& nodeTree, std::vector<std::array<float, BicubicInterpolation::NumBasis>>&& verticesValues) :
		nodeTree(nodeTree), vValues(verticesValues) {}

	CubicNodeTree(NodeTree<Dim>&& nodeTree, CubicNodeTree<Dim>& srcTree) :
		nodeTree(nodeTree) 
	{
		copyValues(srcTree);
	}

	float eval(vec point) const
	{
		using Inter = BicubicInterpolation;
		std::optional<typename NodeTree<Dim>::Node> node;
		nodeTree.getNode(point, node);
		if(!node)
		{
			std::cout << "out of bounds" << std::endl;
			return 100.0f;
		}
		const vec nodeSize = node->maxCoord - node->minCoord;

		std::array<std::array<float, Inter::NumBasis>, Inter::NumControlPoints> weigths;
		Inter::eval(node->transformToLocalCoord(point), nodeSize, weigths);
		float res = 0.0f;
		for(uint32_t i = 0; i < Inter::NumControlPoints; i++)
		{
			for(uint32_t j = 0; j < Inter::NumBasis; j++)
			{
				res += vValues[node->controlPointsIdx[i]][j] * weigths[i][j];
			}
		}
		return res;
	}

	vec evalGrad(vec point) override
	{
		using Inter = BicubicInterpolation;
		std::optional<typename NodeTree<Dim>::Node> node;
		nodeTree.getNode(point, node);
		if(!node)
		{
			std::cout << "out of bounds" << std::endl;
			return vec(0.0f);
		}
		const vec nodeSize = node->maxCoord - node->minCoord;

		std::array<std::array<std::array<float, Inter::NumBasis>, Inter::NumControlPoints>, Dim> weigths;
		Inter::evalGrad(node->transformToLocalCoord(point), nodeSize, weigths);
		vec res;
		for(uint32_t i = 0; i < Dim; i++)
		{
			res[i] = 0.0f;
			for(uint32_t j = 0; j < Inter::NumControlPoints; j++)
			{
				for(uint32_t k = 0; k < Inter::NumBasis; k++)
				{
					res[i] += vValues[node->controlPointsIdx[j]][k] * weigths[i][j][k];
				}
			}
		}
		return res;
	}

	vec getMinCoord() { return nodeTree.getMinCoord(); }
	vec getMaxCoord() { return nodeTree.getMaxCoord(); }

	const NodeTree<Dim>& getNodeTree() const { return nodeTree; }
	const std::vector<std::array<float, BicubicInterpolation::NumBasis>>& getVerticesValues() const
	{
		return vValues;
	}

	float getMaxAbsValue() const
	{
		float absMax = 0.0f;
		for(const auto& bValues : vValues) absMax = glm::max(absMax, glm::abs(bValues[0]));
		return absMax;
	}
public:
	NodeTree<Dim> nodeTree;
	std::vector<std::array<float, BicubicInterpolation::NumBasis>> vValues;

	void copyValues(CubicNodeTree<Dim>& srcTree)
	{
		using Inter = BicubicInterpolation;
		vValues.resize(nodeTree.getNumVertices());
		for(uint32_t i=0; i < nodeTree.getNumVertices(); i++)
		{
			std::optional<typename NodeTree<Dim>::Node> node;
			vec vPos = nodeTree.getVertices()[i];
			srcTree.nodeTree.getNode(vPos, node);
			if(node)
			{
				std::array<std::array<std::array<float, Inter::NumBasis>, Inter::NumControlPoints>, Inter::NumBasis> weights;
				Inter::evalBasisValues(node->transformToLocalCoord(vPos), node->maxCoord - node->minCoord, weights);
				for(uint32_t j=0; j < Inter::NumBasis; j++)
				{
					vValues[i][j] = 0.0f;
					for(uint32_t k=0; k < Inter::NumControlPoints; k++)
					{
						for(uint32_t w=0; w < Inter::NumBasis; w++)
						{
							vValues[i][j] += weights[j][k][w] * srcTree.vValues[node->controlPointsIdx[k]][w];
						}
					}
				}
			}
		}
	}
};

#endif 


