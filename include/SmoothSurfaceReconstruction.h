#ifndef SMOOTH_SURFACE_RECONSTRUCTION_H
#define SMOOTH_SURFACE_RECONSTRUCTION_H

#include <functional>
#include <memory>
#include "PointCloud.h"
#include "Quadtree.h"
#include "ScalarField.h"
#include "InterpolationMethod.h"

namespace SmoothSurfaceReconstruction
{
    struct Config
    {
        uint32_t maxDepth;
        float posWeight;
        float gradientWeight;
        float smoothWeight;
    };

    template<typename Solver>
    std::unique_ptr<BilinearQuadtree> computeWithQuad2D(const PointCloud& cloud, Config config)
    {
        Quadtree quad;
        quad.compute(cloud, config.maxDepth);

        // Get constraints
        struct ConstrainedUnknown
        {
            array<uint32_t, 4> vertIds;
            array<float, 4> weights;
            uint32_t numOperators;
        };

        std::map<int, ConstrainedUnknown> constraints;
        const auto& tJoinVertices = quad.getTJointVerticesIndex();
        for(uint32_t vertId : tJoinVertices)
        {
            ConstrainedUnknown c;
            std::array<Quadtree::Node, 4> nodes;
            uint32_t numNodes = quad.getAdjacentNodes(vertId, nodes);
            Quadtree::Node& node = nodes[0];
            for(uint32_t i=1; i < numNodes; i++)
            {
                if(node.depth > nodes[i].depth) node = nodes[i];
            }

            std::array<float, 4> nodeWeights;
            BilinearInterpolation::eval(node.transformToLocalCoord(quad.getVertices()[vertId]), nodeWeights);
            c.numOperators = 0;

            for(uint32_t i=0; i < 4; i++)
            {
                if(glm::abs(nodeWeights[i]) > 1e-8)
                {
                    c.vertIds[c.numOperators] = node.controlPointsIdx[i];
                    c.weights[c.numOperators++] = nodeWeights[i];
                }
            }

            constraints[vertId] = c;
        }

        const uint32_t numUnknows = quad.getNumVertices() - tJoinVertices.size();
        Solver solver(numUnknows);

        std::vector<uint32_t> vertIdToUnknownId(quad.getNumVertices());

        uint32_t tJoinIdx = 0;
        uint32_t nextUnknownId = 0;
        for(uint32_t i=0; i < quad.getNumVertices(); i++)
        {
            if(tJoinIdx < tJoinVertices.size() && tJoinVertices[tJoinIdx] == i) 
            {
                vertIdToUnknownId[i] = std::numeric_limits<uint32_t>::max();
                tJoinIdx++;
            }
            else vertIdToUnknownId[i] = nextUnknownId++;
        }

        std::function<void(uint32_t, float)> setUnkownValue;
        setUnkownValue = [&](uint32_t vertId, float value)
        {
            if(vertIdToUnknownId[vertId] == std::numeric_limits<uint32_t>::max())
            {
                const ConstrainedUnknown& constraint = constraints[vertId];
                for(uint32_t i=0; i < constraint.numOperators; i++)
                {
                    setUnkownValue(constraint.vertIds[i], value * constraint.weights[i]);
                }
            }
            else
            {
                solver.addTerm(vertIdToUnknownId[vertId], value);
            }
        };

        // Generate point equations
        for(uint32_t i = 0; i < cloud.size(); i++)
	    {
            glm::vec2 nPos = cloud.point(i);
            std::optional<Quadtree::Node> node;
            quad.getNode(nPos, node);
            if(!node) continue;
            std::array<float, 4> weights;
            const glm::vec2 nnPos = node->transformToLocalCoord(nPos);
            BilinearInterpolation::eval(nnPos, weights);

            for(uint32_t i = 0; i < 4; i++) 
                setUnkownValue(node->controlPointsIdx[i], config.posWeight * weights[i]);

            solver.addConstantTerm(0.0f);
            solver.endEquation();

            glm::vec2 nNorm = glm::normalize(cloud.normal(i));
            std::array<std::array<float, 4>, 2> gradWeights;
            BilinearInterpolation::evalGrad(nnPos, gradWeights);

            for(uint32_t j=0; j < 2; j++)
            {
                for(uint32_t i=0; i < 4; i++)
                {
                    setUnkownValue(node->controlPointsIdx[i], config.gradientWeight * gradWeights[j][i]);
                }

                solver.addConstantTerm(config.gradientWeight * nNorm[j]);
                solver.endEquation();
            }
        }

        // Generate Node equations
        const unsigned int numNodesAtMaxDepth = 1 << config.maxDepth;
        const glm::vec2 nodeSize = (quad.getMaxOctreeCoord() - quad.getMinOctreeCoord()) / static_cast<float>(numNodesAtMaxDepth);
        const std::array<glm::vec2, 2> unknownSides = {
            glm::vec2(1.0f, 0.0f),
            glm::vec2(0.0f, 1.0f)
        };
        for(uint32_t i = 0; i < quad.getNumVertices(); i++)
        {
            if(vertIdToUnknownId[i] == std::numeric_limits<uint32_t>::max())
                continue;

            uint32_t numValidSides = 0;
			for(glm::vec2 side : unknownSides)
			{
                std::optional<Quadtree::Node> node;
				bool valid = true;
				for(float sign : {-1.0f, 1.0f})
				{
                    const glm::vec2 coord = quad.getVertices()[i] + sign * nodeSize * side;
					quad.getNode(coord, node);
					if(!node) { valid = false; break; }
				}

				if(!valid) continue;
				numValidSides++;

				for(float sign : {-1.0f, 1.0f})
				{
                    const glm::vec2 coord = quad.getVertices()[i] + sign * nodeSize * side;
					quad.getNode(coord, node);
					std::array<float, 4> weights;
					BilinearInterpolation::eval(node->transformToLocalCoord(coord), weights);
					for(uint32_t i=0; i < 4; i++)
					{
						if(glm::abs(weights[i]) > 1e-8)
						{
							setUnkownValue(node->controlPointsIdx[i], config.smoothWeight * weights[i]);
						}
					}
				}
			}

			if(numValidSides > 0) 
            {
                setUnkownValue(i, -config.smoothWeight * 2.0f * static_cast<float>(numValidSides));
                solver.addConstantTerm(0.0f);
                solver.endEquation();
            }
        }

        std::vector<double> unknownsValues(numUnknows);
        solver.solve(unknownsValues);

        std::function<float(uint32_t)> getVertValue;
        getVertValue = [&](uint32_t vertId) -> float
        {
            if(vertIdToUnknownId[vertId] == std::numeric_limits<uint32_t>::max())
            {
                const ConstrainedUnknown& constraint = constraints[vertId];
                float res = 0.0f;
                for(uint32_t i=0; i < constraint.numOperators; i++)
                {
                    res += constraint.weights[i] * getVertValue(constraint.vertIds[i]);
                }
                return res;
            }
            else return static_cast<float>(unknownsValues[vertIdToUnknownId[vertId]]);
        };
        
        std::vector<float> verticesValues(quad.getNumVertices());
        for(uint32_t i=0; i < quad.getNumVertices(); i++)
        {
            verticesValues[i] = getVertValue(i);
        }

        return std::make_unique<BilinearQuadtree>(std::move(quad), std::move(verticesValues));
    }
}
#endif