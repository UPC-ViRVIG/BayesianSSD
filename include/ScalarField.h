#ifndef _SCALAR_FIELD_INCLUDE
#define _SCALAR_FIELD_INCLUDE


#include <vector>
#include <glm/glm.hpp>
#include "Quadtree.h"
#include "InterpolationMethod.h"
#include "Image.h"

template<uint32_t Dim>
struct VStruct { typedef std::array<float, Dim> type; };

template<>
struct VStruct<2> { typedef glm::vec2 type; };

template<>
struct VStruct<3> { typedef glm::vec3 type; };

template<uint32_t Dim>
class ScalarField
{
public:
	using vec = VStruct<Dim>::type;

	virtual float eval(vec point) = 0;
	virtual vec getMinCoord() = 0;
	virtual vec getMaxCoord() = 0;
private:
};

class BilinearQuadtree : public ScalarField<Quadtree::Dim>
{
public:
	BilinearQuadtree(Quadtree&& quadtree, std::vector<float>&& verticesValues) :
		quad(quadtree), vValues(verticesValues) {}

	float eval(vec point)
	{
		std::optional<Quadtree::Node> node;
		quad.getNode(point, node);
		if(!node) return 100.0f;

		std::array<float, 4> weigths;
		BilinearInterpolation::eval(node->transformToLocalCoord(point), weigths);
		float res = 0.0f;
		for(uint32_t i = 0; i < 4; i++)
		{
			res += vValues[node->controlPointsIdx[i]] * weigths[i];
		}
		return res;
	}

	vec getMinCoord() { return quad.getMinOctreeCoord(); }
	vec getMaxCoord() { return quad.getMaxOctreeCoord(); }

	Quadtree& getQuadtree() { return quad; }

	float getMaxAbsValue()
	{
		float absMax = 0.0f;
		for(float v : vValues) absMax = glm::max(absMax, glm::abs(v));
		return absMax;
	}

private:
	Quadtree quad;
	std::vector<float> vValues;
};

#endif 


