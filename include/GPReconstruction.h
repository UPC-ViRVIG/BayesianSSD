#ifndef GP_RECONSTRUCTION_H
#define GP_RECONSTRUCTION_H

#include <memory>
#include "NodeTree.h"
#include "ScalarField.h"
#include "InterpolationMethod.h"
#include <Eigen/Core>

namespace GPReconstruction
{
    template<uint32_t Dim>
    std::unique_ptr<LinearNodeTree<Dim>> computeLinearNodeTree(NodeTree<Dim>&& quad, const PointCloud<Dim>& cloud, std::optional<LinearNodeTree<Dim>>& outCovariance = {})
    {
        using NodeTreeConfig = NodeTree<Dim>::Config;
        using Inter = MultivariateLinearInterpolation<Dim>;
        using Node = NodeTree<Dim>::Node;
        using vec = ScalarField<Dim>::vec;

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
        std::vector<uint32_t> unknownVertIdToVertId(numUnknows);

        uint32_t tJoinIdx = 0;
        uint32_t nextUnknownId = 0;
        for(uint32_t i=0; i < quad.getNumVertices(); i++)
        {
            if(tJoinIdx < tJoinVertices.size() && tJoinVertices[tJoinIdx] == i) 
            {
                vertIdToUnknownVertId[i] = std::numeric_limits<uint32_t>::max();
                tJoinIdx++;
            }
            else 
            {
                unknownVertIdToVertId[nextUnknownId] = i;
                vertIdToUnknownVertId[i] = nextUnknownId++;
            }
        }
        
        const float sigma = 20.0f;
        const float sigmaSq = sigma * sigma;
        const float length = 8.0f;
        const float lengthSq = length * length;
        auto kernelValue = [&](vec a, vec b)
        {
            vec d = a - b;
            return sigmaSq * glm::exp(-glm::dot(d, d) / (2.0f * lengthSq));
        };

        auto daKernelValue = [&](vec a, vec b, uint32_t dim)
        {
            vec d = a - b;
            return -d[dim] * sigmaSq * glm::exp(-glm::dot(d, d) / (2.0f * lengthSq)) / lengthSq;
        };

        auto dadbKernelValue = [&](vec a, vec b, uint32_t dim1, uint32_t dim2)
        {
            vec d = a - b;
            const float v = (dim1 == dim2) ? lengthSq : 0.0f;
            return (v - d[dim1] * d[dim2]) * sigmaSq * glm::exp(-glm::dot(d, d) / (2.0f * lengthSq)) / (lengthSq * lengthSq);
        };

        // Compute matrix k11
        const uint32_t numPoints = cloud.size();
        const uint32_t size1 = numPoints * (Dim + 1);
        // const uint32_t size1 = numPoints;
        Eigen::MatrixXd k11(size1, size1);

        for(uint32_t i = 0; i < size1; i++)
        {
            for(uint32_t j = 0; j < size1; j++)
            {
                const uint32_t imin = (i == j) ? i : glm::min(i, j);
                const uint32_t imax = (i == j) ? j : glm::max(i, j);

                if(imax < numPoints)
                {
                    k11(i, j) = kernelValue(cloud.point(i), cloud.point(j));
                }
                else if(imin < numPoints)
                {
                    k11(i, j) = daKernelValue(cloud.point((imax - numPoints) / Dim), 
                                            cloud.point(imin), 
                                            (imax - numPoints) % Dim);
                }
                else
                {
                    k11(i, j) = dadbKernelValue(cloud.point((i - numPoints) / Dim), 
                                            cloud.point((j - numPoints) / Dim), (i - numPoints) % Dim, (j - numPoints) % Dim);
                }
            }
        }

        // Compute matrix k22
        const uint32_t size2 = numUnknows;
        Eigen::MatrixXd k22(size2, size2);

        for(uint32_t i = 0; i < size2; i++)
        {
            for(uint32_t j = 0; j < size2; j++)
            {
                const uint32_t iVId = unknownVertIdToVertId[i];
                const uint32_t jVId = unknownVertIdToVertId[j];
                k22(i, j) = kernelValue(quad.getVertices()[iVId], quad.getVertices()[jVId]);
            }
        }

        // Compute matrix k12
        Eigen::MatrixXd k12(size1, size2);

        for(uint32_t i = 0; i < size1; i++)
        {
            for(uint32_t j = 0; j < size2; j++)
            {
                const uint32_t jVId = unknownVertIdToVertId[j];
                if(i < numPoints)
                {
                    k12(i, j) = kernelValue(cloud.point(i), quad.getVertices()[jVId]);
                }
                else
                {
                    k12(i, j) = daKernelValue(cloud.point((i - numPoints) / Dim), 
                                              quad.getVertices()[jVId],
                                              (i - numPoints) % Dim);
                }
            }
        }
        
        // Compute vector y
        Eigen::VectorXd y(size1);
        for(uint32_t i = 0; i < size1; i++)
        {
            if(i < numPoints) y(i) = 0.0;
            else y(i) = cloud.normal((i - numPoints) / Dim)[(i - numPoints) % Dim];
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(k11, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::VectorXd sv = svd.singularValues();
        uint32_t numZeros = 0;
        for(uint32_t i=0; i < sv.size(); i++)
        {
            if(sv(i) > 1e-9)
            {
                sv(i) = 1.0 / sv(i);
            }
            else
            {
                numZeros++;
                sv(i) = 0.0f;
            }
        }
        std::cout << "num zeros " << numZeros << std::endl;
        Eigen::MatrixXd invK11 = svd.matrixV() * sv.asDiagonal() * svd.matrixU().adjoint();

        Eigen::VectorXd mu = k12.transpose() * invK11 * y;
        Eigen::MatrixXd cov = k22 - k12.transpose() * invK11 * k12;

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
            else return static_cast<float>(mu(vertIdToUnknownVertId[vertId]));
        };

        std::function<float(uint32_t)> getVertCov;
        getVertCov = [&](uint32_t vertId) -> float
        {
            if(vertIdToUnknownVertId[vertId] == std::numeric_limits<uint32_t>::max())
            {
                const ConstrainedUnknown& constraint = constraints[vertId];
                float res = 0.0f;
                for(uint32_t i=0; i < constraint.numOperators; i++)
                {
                    res += constraint.weights[i] * getVertCov(constraint.vertIds[i]);
                }
                return res;
            }
            else
            {
                const uint32_t id = vertIdToUnknownVertId[vertId];
                return static_cast<float>(cov(id, id));
            }
        };

        std::vector<float> verticesValues(quad.getNumVertices());
        std::vector<float> verticesCov(quad.getNumVertices());
        uint32_t varNeg = 0;
        for(uint32_t i=0; i < quad.getNumVertices(); i++)
        {
            verticesValues[i] = getVertValue(i);
            verticesCov[i] = getVertCov(i);
            if (verticesCov[i] < 0.0f) varNeg++;
        }

        std::cout << "Negative variances: " << varNeg << std::endl;

        outCovariance = std::optional<LinearNodeTree<Dim>>(LinearNodeTree<Dim>(NodeTree<Dim>(quad), std::move(verticesCov)));

        return std::make_unique<LinearNodeTree<Dim>>(std::move(quad), std::move(verticesValues));
    }
};

#endif