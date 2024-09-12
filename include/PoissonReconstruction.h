#ifndef POISSON_RECONSTRUCTION_H
#define POISSON_RECONSTRUCTION_H

#include <functional>
#include <memory>
#include <optional>
#include "NodeTree.h"
#include "ScalarField.h"
#include "InterpolationMethod.h"
#include "EigenSparseMatrix.h"
#include "EigenSolver.h"
#include <Eigen/Eigenvalues>

namespace PoissonReconstruction
{

    template<uint32_t Dim>
    struct Config
    {
        // TODO: add configuration    
    };

    template<uint32_t Dim>
    std::unique_ptr<LinearNodeTree<Dim>> computeLinearNodeTree(NodeTree<Dim>&& quad, const PointCloud<Dim>& cloud, Config<Dim> config)
    {
        using NodeTreeConfig = NodeTree<Dim>::Config;
        using Inter = MultivariateLinearInterpolation<Dim>;
        using Node = NodeTree<Dim>::Node;
        using vec = ScalarField<Dim>::vec;

        // Calcualte vector field values
        EigenVector pointsVector;

        const unsigned int numNodesAtMaxDepth = 1 << quad.getMaxDepth();
        const vec minNodeSize = (quad.getMaxCoord() - quad.getMinCoord()) / static_cast<float>(numNodesAtMaxDepth);

        auto kernel = [&](vec x, vec y, float sigma)
        {
            auto pow = [](float val, uint32_t n)
            {
                float res = 1.0f;
                for(uint32_t i=0; i < n; i++) res *= val;
                return res;
            };

            float val = 1.0f;
            vec d = (x - y)/sigma;
            for(uint32_t i=0; i < Dim; i++)
            {
                const float a = glm::abs(d[i]);
                val *= int(a < 1.0f) * (3.0 * pow(a, 3) - 6.0 * pow(a, 2) + 4.0f) * (1.0/6.0) + int(a < 2.0f) * int(a > 1.0f) * ( -pow(a, 3) + 6.0 * pow(a, 2) - 12.0 * a + 8.0) * (1.0/6.0);
            }
            return val;
        };

        std::vector<float> pointDensity(cloud.size());

        for(uint32_t p=0; p < cloud.size(); p++)
        {
            float value = 0.0f;
            float sigma = 1.5f * minNodeSize.x;
            float invSigmaDim = sigma;
            for(uint32_t i=1; i < Dim; i++) invSigmaDim *= sigma;
            invSigmaDim = 1.0f / invSigmaDim;
            for(uint32_t i=0; i < cloud.size(); i++)
            {
                value += kernel(cloud.point(p), cloud.point(i), sigma) * invSigmaDim;
            }
            pointDensity[p] = value;
        }

        auto getTargetVector = [&](vec vPoint) 
        {
            vec value = vec(0.0f);

            float sigma = 1.5f * minNodeSize.x;
            float invSigmaDim = sigma;
            for(uint32_t i=1; i < Dim; i++) invSigmaDim *= sigma;
            invSigmaDim = 1.0f / invSigmaDim;
            for(uint32_t i=0; i < cloud.size(); i++)
            {
                value += kernel(vPoint, cloud.point(i), sigma) * invSigmaDim * glm::normalize(cloud.normal(i)) / pointDensity[i];
            }

            // float w = glm::abs(glm::length(vPoint - vec(0.5f, 0.5f)) - 0.2f);
            // w = glm::smoothstep(-0.1f, 0.0f, -w);

            // // return w * glm::normalize(vPoint - vec(0.5f, 0.5f));
            // vec a = w * glm::normalize(vPoint - vec(0.5f, 0.5f));
            // if(glm::any(glm::isnan(a)))
            // {
            //     return vec(0.0f);
            // }
            // return a;

            return value;
        };

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



        EigenSparseMatrix posMatrix(numUnknows);
        EigenSparseMatrix gradientMatrix(numUnknows);

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
                gradientMatrix.addTerm(vertIdToUnknownVertId[vertId], value);
            }
        };

        // Construct gradient matrix
        const float invNodeSize = 1.0f / minNodeSize.x;
        for(uint32_t i = 0; i < quad.getNumVertices(); i++)
        {
            // if(vertIdToUnknownVertId[i] == std::numeric_limits<uint32_t>::max())
            //     continue;
            vec targetVector = getTargetVector(quad.getVertices()[i]);
            for(uint32_t d=0; d < Dim; d++)
            {
                for(float sign1 : {-1.0f, 1.0f})
                // for(float sign1 : {1.0f})
                {
                    vec side;
                    for(uint32_t j=0; j < Dim; j++)
                        side[j] = (d == j) ? 1.0f : 0.0f;
                    
                    std::optional<Node> node;
                    bool valid = true;
                    // for(float sign : {-1.0f, 1.0f})
                    for(float sign : {sign1})
                    {
                        const vec coord = quad.getVertices()[i] + sign * minNodeSize * side;
                        quad.getNode(coord, node);
                        if(!node) { valid = false; break; }
                    }

                    if(!valid) continue;

                    setUnkownValue(i, -sign1 * invNodeSize);
                    // for(float sign : {-1.0f, 1.0f})
                    for(float sign : {sign1})
                    {
                        const vec coord = quad.getVertices()[i] + sign * minNodeSize * side;
                        quad.getNode(coord, node);
                        std::array<float, Inter::NumControlPoints> weights;
                        Inter::eval(node->transformToLocalCoord(coord), weights);
                        for(uint32_t j=0; j < Inter::NumControlPoints; j++)
                        {
                            if(glm::abs(weights[j]) > 1e-8)
                            {
                                setUnkownValue(node->controlPointsIdx[j], sign1 * invNodeSize * weights[j]);
                            }
                        }
                    }

                    gradientMatrix.endEquation();
                    pointsVector.addTerm(targetVector[d]);
                }
			}
        }

        std::function<void(uint32_t, float)> setUnkownValuePos;
        setUnkownValuePos = [&](uint32_t vertId, float value)
        {
            if(vertIdToUnknownVertId[vertId] == std::numeric_limits<uint32_t>::max())
            {
                const ConstrainedUnknown& constraint = constraints[vertId];
                for(uint32_t i=0; i < constraint.numOperators; i++)
                {
                    setUnkownValuePos(constraint.vertIds[i], value * constraint.weights[i]);
                }
            }
            else
            {
                posMatrix.addTerm(vertIdToUnknownVertId[vertId], value);
            }
        };

        for(uint32_t i = 0; i < cloud.size(); i++)
        {
            std::optional<Node> node;
            quad.getNode(cloud.point(i), node);
            if(node)
            {
                std::array<float, Inter::NumControlPoints> weights;
                Inter::eval(node->transformToLocalCoord(cloud.point(i)), weights);
                for(uint32_t j=0; j < Inter::NumControlPoints; j++)
                {
                    setUnkownValuePos(node->controlPointsIdx[j], weights[j]);
                }
                posMatrix.endEquation();
            }
        }

        std::vector<double> unknownsValues(numUnknows);
        Eigen::SparseMatrix<double> G = gradientMatrix.getMatrix();
        Eigen::SparseMatrix<double> P = posMatrix.getMatrix();
        Eigen::VectorXd v = pointsVector.getVector();
        Eigen::SparseMatrix<double> A = G.transpose() * G + P.transpose() * P;
        A += 1e-3 * Eigen::VectorXd::Ones(numUnknows).asDiagonal();
        auto b = G.transpose() * v;
        // auto A = G;
        // auto b = v;
        auto x = Eigen::Map<Eigen::VectorXd>(unknownsValues.data(), unknownsValues.size());
        EigenSolver::BiCGSTAB::solve(A, b, x);
        std::cout << "Final error: " << (A*x - b).squaredNorm() << std::endl;

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
}

#endif