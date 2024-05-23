#ifndef SMOOTH_SURFACE_RECONSTRUCTION_H
#define SMOOTH_SURFACE_RECONSTRUCTION_H

#include <functional>
#include <memory>
#include "NodeTree.h"
#include "ScalarField.h"
#include "InterpolationMethod.h"
#include "EigenSparseMatrix.h"
#include "EigenSolver.h"

namespace SmoothSurfaceReconstruction
{
    template<uint32_t Dim>
    struct Config
    {
        NodeTree<Dim>::Config nodeTreeConfig;
        float posWeight;
        float gradientWeight;
        float smoothWeight;
    };

    template<uint32_t Dim, typename Solver>
    std::unique_ptr<LinearNodeTree<Dim>> computeLinearNodeTree(const PointCloud<Dim>& cloud, Config<Dim> config)
    {
        using NodeTreeConfig = NodeTree<Dim>::Config;
        using Inter = MultivariateLinearInterpolation<Dim>;
        using Node = NodeTree<Dim>::Node;
        using vec = ScalarField<Dim>::vec;

        NodeTree<Dim> quad;
        quad.compute(cloud, config.nodeTreeConfig);

        // Get constraints
        struct ConstrainedUnknown
        {
            std::array<uint32_t, Inter::NumControlPoints> vertIds;
            std::array<float, Inter::NumControlPoints> weights;
            uint32_t numOperators;
        };

        std::map<int, ConstrainedUnknown> constraints;
        const auto& tJoinVertices = quad.getTJointVerticesIndex();
        for(uint32_t vertId : tJoinVertices)
        {
            ConstrainedUnknown c;
            std::array<Node, NodeTree<Dim>::NumAdjacentNodes> nodes;
            uint32_t numNodes = quad.getAdjacentNodes(vertId, nodes);
            Node& node = nodes[0];
            for(uint32_t i=1; i < numNodes; i++)
            {
                if(node.depth > nodes[i].depth) node = nodes[i];
            }

            std::array<float, Inter::NumControlPoints> nodeWeights;
            Inter::eval(node.transformToLocalCoord(quad.getVertices()[vertId]), nodeWeights);
            c.numOperators = 0;

            for(uint32_t i=0; i < Inter::NumControlPoints; i++)
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

        std::vector<uint32_t> vertIdToUnknownVertId(quad.getNumVertices());

        uint32_t tJoinIdx = 0;
        uint32_t nextUnknownId = 0;
        for(uint32_t i=0; i < quad.getNumVertices(); i++)
        {
            if(tJoinIdx < tJoinVertices.size() && tJoinVertices[tJoinIdx] == i) 
            {
                vertIdToUnknownVertId[i] = std::numeric_limits<uint32_t>::max();
                tJoinIdx++;
            }
            else vertIdToUnknownVertId[i] = nextUnknownId++;
        }

        std::function<void(uint32_t, float)> setUnkownValue;
        setUnkownValue = [&](uint32_t vertId, float value)
        {
            if(vertIdToUnknownVertId[vertId] == std::numeric_limits<uint32_t>::max())
            {
                const ConstrainedUnknown& constraint = constraints[vertId];
                for(uint32_t i=0; i < constraint.numOperators; i++)
                {
                    setUnkownValue(constraint.vertIds[i], value * constraint.weights[i]);
                }
            }
            else
            {
                solver.addTerm(vertIdToUnknownVertId[vertId], value);
            }
        };

        // Generate point equations
        for(uint32_t i = 0; i < cloud.size(); i++)
	    {
            vec nPos = cloud.point(i);
            std::optional<Node> node;
            quad.getNode(nPos, node);
            if(!node) continue;
            std::array<float, Inter::NumControlPoints> weights;
            const vec nnPos = node->transformToLocalCoord(nPos);
            Inter::eval(nnPos, weights);

            for(uint32_t i = 0; i < Inter::NumControlPoints; i++) 
                setUnkownValue(node->controlPointsIdx[i], config.posWeight * weights[i]);

            solver.addConstantTerm(0.0f);
            solver.endEquation();

            vec nNorm = glm::normalize(cloud.normal(i));
            std::array<std::array<float, Inter::NumControlPoints>, Dim> gradWeights;
            Inter::evalGrad(nnPos, gradWeights);

            for(uint32_t j=0; j < Dim; j++)
            {
                for(uint32_t i=0; i < Inter::NumControlPoints; i++)
                {
                    setUnkownValue(node->controlPointsIdx[i], config.gradientWeight * gradWeights[j][i]);
                }

                solver.addConstantTerm(config.gradientWeight * nNorm[j]);
                solver.endEquation();
            }
        }

        // Generate Node equations
        const unsigned int numNodesAtMaxDepth = 1 << config.nodeTreeConfig.maxDepth;
        const vec nodeSize = (quad.getMaxCoord() - quad.getMinCoord()) / static_cast<float>(numNodesAtMaxDepth);
        for(uint32_t i = 0; i < quad.getNumVertices(); i++)
        {
            if(vertIdToUnknownVertId[i] == std::numeric_limits<uint32_t>::max())
                continue;

            uint32_t numValidSides = 0;
            for(uint32_t d=0; d < Dim; d++)
            {
                vec side;
                for(uint32_t j=0; j < Dim; j++)
                    side[j] = (d == j) ? 1.0f : 0.0f;
                
                std::optional<Node> node;
				bool valid = true;
				for(float sign : {-1.0f, 1.0f})
				{
                    const vec coord = quad.getVertices()[i] + sign * nodeSize * side;
					quad.getNode(coord, node);
					if(!node) { valid = false; break; }
				}

				if(!valid) continue;
				numValidSides++;

				for(float sign : {-1.0f, 1.0f})
				{
                    const vec coord = quad.getVertices()[i] + sign * nodeSize * side;
					quad.getNode(coord, node);
					std::array<float, Inter::NumControlPoints> weights;
					Inter::eval(node->transformToLocalCoord(coord), weights);
					for(uint32_t i=0; i < Inter::NumControlPoints; i++)
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
            if(vertIdToUnknownVertId[vertId] == std::numeric_limits<uint32_t>::max())
            {
                const ConstrainedUnknown& constraint = constraints[vertId];
                float res = 0.0f;
                for(uint32_t i=0; i < constraint.numOperators; i++)
                {
                    res += constraint.weights[i] * getVertValue(constraint.vertIds[i]);
                }
                return res;
            }
            else return static_cast<float>(unknownsValues[vertIdToUnknownVertId[vertId]]);
        };
        
        std::vector<float> verticesValues(quad.getNumVertices());
        for(uint32_t i=0; i < quad.getNumVertices(); i++)
        {
            verticesValues[i] = getVertValue(i);
        }

        return std::make_unique<LinearNodeTree<Dim>>(std::move(quad), std::move(verticesValues));
    }

    template<uint32_t Dim>
    std::unique_ptr<CubicNodeTree<Dim>> compute2DCubicNodeTree(const PointCloud<Dim>& cloud, Config<Dim> config)
    {
        using NodeTreeConfig = NodeTree<Dim>::Config;
        using Inter = BicubicInterpolation;
        using Node = NodeTree<Dim>::Node;
        using vec = ScalarField<Dim>::vec;

        NodeTree<Dim> quad;
        quad.compute(cloud, config.nodeTreeConfig);

        // Get constraints
        struct ConstrainedUnknown
        {
            std::array<uint32_t, Inter::NumControlPoints> vertIds;
            std::array<std::array<std::array<float, Inter::NumBasis>, Inter::NumControlPoints>, Inter::NumBasis> weights;
        };

        std::map<int, ConstrainedUnknown> constraints;
        const auto& tJoinVertices = quad.getTJointVerticesIndex();
        for(uint32_t vertId : tJoinVertices)
        {
            ConstrainedUnknown c;
            std::array<Node, NodeTree<Dim>::NumAdjacentNodes> nodes;
            uint32_t numNodes = quad.getAdjacentNodes(vertId, nodes);
            Node& node = nodes[0];
            for(uint32_t i=1; i < numNodes; i++)
            {
                if(node.depth > nodes[i].depth) node = nodes[i];
            }

            c.vertIds = node.controlPointsIdx;
            const vec nodeSize = node.maxCoord  - node.minCoord;
            Inter::evalBasisValues(node.transformToLocalCoord(quad.getVertices()[vertId]), nodeSize, c.weights);
            constraints[vertId] = c;
        }

        const uint32_t numUnknows = Inter::NumBasis * (quad.getNumVertices() - tJoinVertices.size());
        EigenSparseMatrix pointsMatrix(numUnknows);
        EigenVector pointsVector;

        std::vector<uint32_t> vertIdToUnknownVertId(quad.getNumVertices());

        uint32_t tJoinIdx = 0;
        uint32_t nextUnknownVertId = 0;
        for(uint32_t i=0; i < quad.getNumVertices(); i++)
        {
            if(tJoinIdx < tJoinVertices.size() && tJoinVertices[tJoinIdx] == i) 
            {
                vertIdToUnknownVertId[i] = std::numeric_limits<uint32_t>::max();
                tJoinIdx++;
            }
            else vertIdToUnknownVertId[i] = nextUnknownVertId++;
        }

        auto getUnknownId = [&](uint32_t unknownVertId, uint32_t basisId) { return Inter::NumBasis * unknownVertId + basisId; };

        std::function<void(uint32_t, const std::array<float, Inter::NumBasis>&)> setUnkownValue;
        setUnkownValue = [&](uint32_t vertId, const std::array<float, Inter::NumBasis>& value)
        {
            if(vertIdToUnknownVertId[vertId] == std::numeric_limits<uint32_t>::max())
            {
                const ConstrainedUnknown& c = constraints[vertId];
                for(uint32_t i=0; i < Inter::NumControlPoints; i++)
                {
                    std::array<float, Inter::NumBasis> cValues;
                    for(uint32_t j=0; j < Inter::NumBasis; j++)
                    {
                        float res = 0.0f;
                        for(uint32_t k=0; k < Inter::NumBasis; k++)
                        {
                            res += (glm::abs(c.weights[k][i][j]) < 1e-8) ? 0.0f : c.weights[k][i][j] * value[k];
                        }
                        cValues[j] = res;
                    }
                    setUnkownValue(c.vertIds[i], cValues);
                }
            }
            else
            {
                for(uint32_t i=0; i < Inter::NumBasis; i++)
                {
                    if(value[i] == 0.0f) continue;
                    pointsMatrix.addTerm(Inter::NumBasis * vertIdToUnknownVertId[vertId] + i, value[i]);
                }
            }
        };

        // Generate point equations
        for(uint32_t i = 0; i < cloud.size(); i++)
	    {
            vec nPos = cloud.point(i);
            std::optional<Node> node;
            quad.getNode(nPos, node);
            if(!node) continue;
            const vec nodeSize = node->maxCoord - node->minCoord;
            const vec nnPos = node->transformToLocalCoord(nPos);

            std::array<std::array<float, Inter::NumBasis>, Inter::NumControlPoints> weights;
            Inter::eval(nnPos, nodeSize, weights);

            for(uint32_t j = 0; j < Inter::NumControlPoints; j++)
            {
                for(float& v : weights[j]) v *= config.posWeight;
                setUnkownValue(node->controlPointsIdx[j], weights[j]);
            }

            pointsVector.addTerm(0.0f);
            pointsMatrix.endEquation();

            vec nNorm = glm::normalize(cloud.normal(i));
            std::array<std::array<std::array<float, Inter::NumBasis>, Inter::NumControlPoints>, Dim> gradWeights;
            Inter::evalGrad(nnPos, nodeSize, gradWeights);

            for(uint32_t j=0; j < Dim; j++)
            {
                for(uint32_t i=0; i < Inter::NumControlPoints; i++)
                {
                    for(float& v : gradWeights[j][i]) v *= config.gradientWeight;
                    setUnkownValue(node->controlPointsIdx[i], gradWeights[j][i]);
                }

                pointsVector.addTerm(config.gradientWeight * nNorm[j]);
                pointsMatrix.endEquation();
            }
        }

        // Generate Node equations
        // const unsigned int numNodesAtMaxDepth = 1 << config.nodeTreeConfig.maxDepth;
        // const vec nodeSize = 0.5f * (quad.getMaxCoord() - quad.getMinCoord()) / static_cast<float>(numNodesAtMaxDepth);
        // for(uint32_t i = 0; i < quad.getNumVertices(); i++)
        // {
        //     if(vertIdToUnknownVertId[i] == std::numeric_limits<uint32_t>::max())
        //         continue;

        //     uint32_t numValidSides = 0;
        //     for(uint32_t d=0; d < Dim; d++)
        //     {
        //         vec side;
        //         for(uint32_t j=0; j < Dim; j++)
        //             side[j] = (d == j) ? 1.0f : 0.0f;
                
        //         std::optional<Node> node;
		// 		bool valid = true;
		// 		for(float sign : {-1.0f, 1.0f})
		// 		{
        //             const vec coord = quad.getVertices()[i] + sign * nodeSize * side;
		// 			quad.getNode(coord, node);
		// 			if(!node) { valid = false; break; }
		// 		}

		// 		if(!valid) continue;
		// 		numValidSides++;

		// 		for(float sign : {-1.0f, 1.0f})
		// 		{
        //             const vec coord = quad.getVertices()[i] + sign * nodeSize * side;
		// 			quad.getNode(coord, node);
		// 			std::array<std::array<float, Inter::NumBasis>, Inter::NumControlPoints> weights;
		// 			Inter::eval(node->transformToLocalCoord(coord), weights);
		// 			for(uint32_t i=0; i < Inter::NumControlPoints; i++)
		// 			{
        //                 for(float& v : weights[i]) v *= config.smoothWeight;
        //                 setUnkownValue(node->controlPointsIdx[i], weights[i]);
		// 			}
		// 		}
		// 	}

		// 	if(numValidSides > 0) 
        //     {
        //         std::array<float, Inter::NumBasis> weights;
        //         weights.fill(0.0f);
        //         weights[0] = -config.smoothWeight * 2.0f * static_cast<float>(numValidSides);
        //         setUnkownValue(i, weights);
        //         solver.addConstantTerm(0.0f);
        //         solver.endEquation();
        //     }
        // }

        // Generate Node equations
        EigenSparseMatrix nodesMatrix(numUnknows, numUnknows);
        std::function<void(uint32_t, uint32_t, const std::array<float, Inter::NumBasis>&)> setUnkownTerm;
        setUnkownTerm = [&](uint32_t eqIdx, uint32_t vertId, const std::array<float, Inter::NumBasis>& value)
        {
            if(vertIdToUnknownVertId[vertId] == std::numeric_limits<uint32_t>::max())
            {
                const ConstrainedUnknown& c = constraints[vertId];
                for(uint32_t i=0; i < Inter::NumControlPoints; i++)
                {
                    std::array<float, Inter::NumBasis> cValues;
                    for(uint32_t j=0; j < Inter::NumBasis; j++)
                    {
                        float res = 0.0f;
                        for(uint32_t k=0; k < Inter::NumBasis; k++)
                        {
                            res += (glm::abs(c.weights[k][i][j]) < 1e-8) ? 0.0f : c.weights[k][i][j] * value[k];
                        }
                        cValues[j] = res;
                    }
                    setUnkownTerm(eqIdx, c.vertIds[i], cValues);
                }
            }
            else
            {
                for(uint32_t i=0; i < Inter::NumBasis; i++)
                {
                    if(value[i] == 0.0f) continue;
                    nodesMatrix.addTerm(eqIdx, Inter::NumBasis * vertIdToUnknownVertId[vertId] + i, value[i]);
                }
            }
        };

        std::array<std::array<std::array<std::array<float, 4>, 4>, 4>, 4> lapWeights;

        for(const Node& node : quad)
        {
            const vec nodeSize = node.maxCoord - node.minCoord;
            BicubicInterpolation::integrateLaplacian(nodeSize, lapWeights);
            for(auto& a1 : lapWeights)
                for(auto& a2 : a1)
                    for(auto& a3 : a2)
                        for(auto& v : a3)
                            v *= config.smoothWeight;
            for(uint32_t i=0; i < Inter::NumControlPoints; i++)
            {
                const uint32_t vertId = node.controlPointsIdx[i];
                if(vertIdToUnknownVertId[vertId] == std::numeric_limits<uint32_t>::max()) continue;
                for(uint32_t j=0; j < Inter::NumBasis; j++)
                {
                    const uint32_t eqIdx = vertIdToUnknownVertId[vertId] * Inter::NumBasis + j;
                    for(uint32_t k=0; k < Inter::NumControlPoints; k++)
                    {
                        setUnkownTerm(eqIdx, node.controlPointsIdx[k], lapWeights[i][j][k]);
                    }
                }                
            }
        }

        std::vector<double> unknownsValues(numUnknows);
        Eigen::SparseMatrix<double> pM = pointsMatrix.getMatrix();
        Eigen::VectorXd pb = pointsVector.getVector();
        Eigen::SparseMatrix<double> nM = nodesMatrix.getMatrix();
        auto A = pM.transpose() * pM + nM;
        auto b = pM.transpose() * pb;
        auto x = Eigen::Map<Eigen::VectorXd>(unknownsValues.data(), unknownsValues.size());
        EigenSolver::BiCGSTAB::solve(A, b, x);

        std::function<void(uint32_t, std::array<float, Inter::NumBasis>&)> getVertValue;
        getVertValue = [&](uint32_t vertId, std::array<float, Inter::NumBasis>& outValues)
        {
            if(vertIdToUnknownVertId[vertId] == std::numeric_limits<uint32_t>::max())
            {
                const ConstrainedUnknown& c = constraints[vertId];
                outValues.fill(0.0f);
                for(uint32_t i=0; i < Inter::NumControlPoints; i++)
                {
                    std::array<float, Inter::NumBasis> cValues;
                    getVertValue(c.vertIds[i], cValues);
                    for(uint32_t j=0; j < Inter::NumBasis; j++)
                    {
                        for(uint32_t k=0; k < Inter::NumBasis; k++)
                        {
                            outValues[j] += c.weights[j][i][k] * cValues[k];
                        }
                    }
                }
            }
            else 
            {
                for(uint32_t i=0; i < Inter::NumBasis; i++)
                {
                    outValues[i] = unknownsValues[Inter::NumBasis * vertIdToUnknownVertId[vertId] + i];
                }
            }
        };
        
        std::vector<std::array<float, Inter::NumBasis>> verticesValues(quad.getNumVertices());
        for(uint32_t i=0; i < quad.getNumVertices(); i++)
        {
            getVertValue(i, verticesValues[i]);
        }

        std::array<std::array<float, Inter::NumBasis>, Inter::NumControlPoints> nodeValues;
        float totalLaplacian = 0.0f;
        for(const Node& node : quad)
        {
            const vec nodeSize = node.maxCoord - node.minCoord;
            for(uint32_t i=0; i < Inter::NumControlPoints; i++)
            {
                for(uint32_t j=0; j < Inter::NumBasis; j++)
                {
                    nodeValues[i][j] = verticesValues[node.controlPointsIdx[i]][j];
                }
            }
            const float val = BicubicInterpolation::integrateLaplacian(nodeSize, nodeValues);
            // if(val < 0.0f)
            // {
            //     std::cout << "negative value " << val << std::endl;
            // }
            // if(val > 0.13617)
            // {
            //     // std::cout << numNodes << "big value " << val << std::endl;
            // }
            // std::cout << numNodes << "value " << val << std::endl;
            totalLaplacian += val;
        }

        std::cout << "Total laplacian: " << totalLaplacian << std::endl;

        return std::make_unique<CubicNodeTree<Dim>>(std::move(quad), std::move(verticesValues));
    }
}
#endif