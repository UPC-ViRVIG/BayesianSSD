#ifndef SMOOTH_SURFACE_RECONSTRUCTION_H
#define SMOOTH_SURFACE_RECONSTRUCTION_H

#include <functional>
#include <memory>
#include "NodeTree.h"
#include "ScalarField.h"
#include "InterpolationMethod.h"
#include "EigenSparseMatrix.h"
#include "EigenSolver.h"
#include <Eigen/Eigenvalues>
#include "Timer.h"

#define M_PI 3.14159265359

#include <fstream>
template <typename T>
void write_array_to_file(const std::vector<T>& arr, const std::string& filename) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for writing.");
  }

  // Write the size of the array as the first element
  uint32_t size = arr.size();
  file.write(reinterpret_cast<const char*>(&size), sizeof(size));

  // Write each element of the array
  for (const T& element : arr) {
    file.write(reinterpret_cast<const char*>(&element), sizeof(element));
  }

  file.close();
}

namespace SmoothSurfaceReconstruction
{
    enum Algorithm
    {
        VAR,
        BAYESIAN,
        GP   
    };

    template<uint32_t Dim>
    struct Config
    {
        float posWeight;
        float gradientWeight;
        float smoothWeight;
        Algorithm algorithm;
        bool computeVariance;
    };

    template<uint32_t Dim>
    std::unique_ptr<LinearNodeTree<Dim>> computeLinearNodeTree(NodeTree<Dim>&& quad, const PointCloud<Dim>& cloud, Config<Dim> config,
                                                               std::optional<LinearNodeTree<Dim>>& outCovariance)
    {
        std::optional<Eigen::MatrixXd> invCovMat;
        std::optional<Eigen::MatrixXd> covMat;
        std::optional<Eigen::SparseMatrix<double>> outP;
        std::optional<Eigen::SparseMatrix<double>> outN;
        std::optional<Eigen::SparseMatrix<double>> outS;
        std::optional<Eigen::VectorXd> outW;
        return computeLinearNodeTree(std::move(quad), cloud, config, outCovariance, invCovMat, covMat, outP, outN, outS, outW);
    }

    template<uint32_t Dim>
    std::unique_ptr<LinearNodeTree<Dim>> computeLinearNodeTree(NodeTree<Dim>&& quad, const PointCloud<Dim>& cloud, Config<Dim> config,
                                                               std::optional<LinearNodeTree<Dim>>& outCovariance,
                                                               std::optional<Eigen::MatrixXd>& invCovMat,
                                                               std::optional<Eigen::MatrixXd>& covMat,
                                                               std::optional<Eigen::SparseMatrix<double>>& outP,
                                                               std::optional<Eigen::SparseMatrix<double>>& outN,
                                                               std::optional<Eigen::SparseMatrix<double>>& outS,
                                                               std::optional<Eigen::VectorXd>& outW)
    {
        using NodeTreeConfig = NodeTree<Dim>::Config;
        using Inter = MultivariateLinearInterpolation<Dim>;
        using Node = NodeTree<Dim>::Node;
        using vec = ScalarField<Dim>::vec;

        Timer timer;
        timer.start();

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

        std::function<void(EigenSparseMatrix& mat, uint32_t, float)> setUnkownValue;
        setUnkownValue = [&](EigenSparseMatrix& mat, uint32_t vertId, float value)
        {
            if(vertIdToUnknownVertId[vertId] == std::numeric_limits<uint32_t>::max())
            {
                const ConstrainedUnknown& constraint = constraints[vertId];
                for(uint32_t i=0; i < constraint.numOperators; i++)
                {
                    setUnkownValue(mat, constraint.vertIds[i], value * constraint.weights[i]);
                }
            }
            else
            {
                mat.addTerm(vertIdToUnknownVertId[vertId], value);
            }
        };

        EigenSparseMatrix pointsMatrix(numUnknows);
        EigenSparseMatrix pointsCovMatrix(cloud.size());
        EigenSparseMatrix gradientMatrix(numUnknows);
        EigenVector gradientVector;

        // Generate point equations
        for(uint32_t i = 0; i < cloud.size(); i++)
	    {
            vec nPos = cloud.point(i);
            std::optional<Node> node;
            quad.getNode(nPos, node);
            if(!node) continue;
            std::array<float, Inter::NumControlPoints> weights;
            const vec nnPos = node->transformToLocalCoord(nPos);
            const vec nodeSize = node->maxCoord - node->minCoord;
            Inter::eval(nnPos, weights);

            for(uint32_t i = 0; i < Inter::NumControlPoints; i++)
                setUnkownValue(pointsMatrix, node->controlPointsIdx[i], weights[i]);

            pointsCovMatrix.addTerm(i, cloud.variance(i));
            pointsCovMatrix.endEquation();
            pointsMatrix.endEquation();

            vec nNorm = glm::normalize(cloud.normal(i));
            std::array<std::array<float, Inter::NumControlPoints>, Dim> gradWeights;
            Inter::evalGrad(nnPos, nodeSize, gradWeights);

            for(uint32_t j=0; j < Dim; j++)
            {
                for(uint32_t i=0; i < Inter::NumControlPoints; i++)
                {
                    setUnkownValue(gradientMatrix, node->controlPointsIdx[i], gradWeights[j][i]);
                }

                gradientVector.addTerm(nNorm[j]);
                gradientMatrix.endEquation();
            }
        }

        // Generate Node equations
        constexpr bool extrapolateLaplacian = false;
        EigenSparseMatrix smoothMatrix(numUnknows);
        const unsigned int numNodesAtMaxDepth = 1 << quad.getMaxDepth();
        const vec nodeSize = (quad.getMaxCoord() - quad.getMinCoord()) / static_cast<float>(numNodesAtMaxDepth) - vec(1e-6);
        const float invSizeSq = 1.0f / (nodeSize[0] * nodeSize[0]);
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
                float validSign = 0.0f;
				for(float sign : {-1.0f, 1.0f})
				{
                    const vec coord = quad.getVertices()[i] + sign * nodeSize * side;
					quad.getNode(coord, node);
					if(!node) { 
                        valid = false;
                        validSign = -sign; // The valid sign is the opposite one
                        break;
                    }
				}

                // All the sides are valid
				// if(!valid) continue;
                if(valid)
                {
                    for(float sign : {-1.0f, 1.0f})
                    {
                        const vec coord = quad.getVertices()[i] + sign * nodeSize * side;
                        quad.getNode(coord, node);
                        std::array<float, Inter::NumControlPoints> weights;
                        Inter::eval(node->transformToLocalCoord(coord), weights);
                        for(uint32_t j=0; j < Inter::NumControlPoints; j++)
                        {
                            if(glm::abs(weights[j]) > 1e-8)
                            {
                                setUnkownValue(smoothMatrix, node->controlPointsIdx[j], invSizeSq * weights[j]);
                            }
                        }
                    }
                    numValidSides++;
                }
                else if(extrapolateLaplacian)
                {
                    for(uint32_t j = 1; j <= 2; j++)
                    {
                        const vec coord = quad.getVertices()[i] + (validSign * static_cast<float>(j)) * nodeSize * side;
                        quad.getNode(coord, node);
                        std::array<float, Inter::NumControlPoints> weights;
                        Inter::eval(node->transformToLocalCoord(coord), weights);
                        float w = (j == 1) ? 2.0f : -1.0f;
                        for(uint32_t j=0; j < Inter::NumControlPoints; j++)
                        {
                            if(glm::abs(weights[j]) > 1e-8)
                            {
                                setUnkownValue(smoothMatrix, node->controlPointsIdx[j], invSizeSq * w * weights[j]);
                            }
                        }
                    }
                    numValidSides++;
                }
			}

			if(numValidSides > 0) 
            {
                setUnkownValue(smoothMatrix, i, -2.0f * invSizeSq * static_cast<float>(numValidSides));
                smoothMatrix.endEquation();
            }
        }

        Eigen::SparseMatrix<double> P = pointsMatrix.getMatrix();
        Eigen::MatrixXd covP = Eigen::MatrixXd(pointsCovMatrix.getMatrix());
        Eigen::SparseMatrix<double> N = gradientMatrix.getMatrix();
        Eigen::SparseMatrix<double> S = smoothMatrix.getMatrix();
        float invVarSmoothing = config.smoothWeight * config.smoothWeight;
        float invVarGradient = config.gradientWeight * config.gradientWeight;
        Eigen::VectorXd b = N.transpose() * invVarGradient * gradientVector.getVector();

        std::cout << "Time setting the problem: " << timer.getElapsedSeconds() << std::endl;

        if(config.computeVariance)
        {
            timer.start();

            Eigen::MatrixXd mP = Eigen::MatrixXd(P);
            Eigen::MatrixXd mS = Eigen::MatrixXd(S);
            Eigen::MatrixXd mN = Eigen::MatrixXd(N);
            Eigen::MatrixXd SVDmat;

            switch(config.algorithm)
            {
                case VAR:
                    SVDmat = mP.transpose() * covP.inverse() * mP + invVarSmoothing * mS.transpose() * mS + invVarGradient * mN.transpose() * mN;
                    break;
                case BAYESIAN:
                    SVDmat = mP.transpose() * covP.inverse() * mP + invVarSmoothing * mS.transpose() * mS + invVarGradient * mN.transpose() * mN;
                    break;
                case GP:
                    // SVDmat = invVarSmoothing * mS.transpose() * mS;
                    SVDmat = mP.transpose() * covP.inverse() * mP + invVarSmoothing * mS.transpose() * mS + invVarGradient * mN.transpose() * mN;
                    break;
            }

            invCovMat = std::optional<Eigen::MatrixXd>(SVDmat);

            Eigen::JacobiSVD<Eigen::MatrixXd> svd(SVDmat, Eigen::ComputeThinU | Eigen::ComputeThinV);
            Eigen::VectorXd sv = svd.singularValues();
            uint32_t numZeros = 0;
            for(uint32_t i=0; i < sv.size(); i++)
            {
                if(sv(i) > 1e-9)
                {
                    sv(i) = 1.0f / sv(i);
                }
                else
                {
                    numZeros++;
                    sv(i) = 0.0f;
                }
            }
            Eigen::MatrixXd invSVDmat = svd.matrixV() * sv.asDiagonal() * svd.matrixU().adjoint();

            Eigen::MatrixXd CovX;
            switch(config.algorithm)
            {
                case VAR:
                    CovX = invSVDmat * mP.transpose() * covP.inverse() * mP * invSVDmat.transpose() + invSVDmat * mN.transpose() * invVarGradient * mN * invSVDmat.transpose();
                    break;
                case BAYESIAN:
                    CovX = invSVDmat;
                    break;
                case GP:
                    CovX = invSVDmat - invSVDmat * mP.transpose() * covP.inverse() * mP * invSVDmat.transpose() + invSVDmat * mN.transpose() * invVarGradient * mN * invSVDmat.transpose();
                    break;
            }
            covMat = std::optional<Eigen::MatrixXd>(CovX);

            std::cout << "Time computing covariance: " << timer.getElapsedSeconds() << std::endl;
        }
        
        timer.start();

        Eigen::SparseMatrix<double> A = P.transpose() * covP.inverse() * P + invVarGradient * N.transpose() * N + invVarSmoothing * S.transpose() * S;
        std::vector<double> unknownsValues(numUnknows);
        auto x = Eigen::Map<Eigen::VectorXd>(unknownsValues.data(), unknownsValues.size());

        EigenSolver::BiCGSTAB::solve(A, b, x);

        std::cout << "Time solving problem: " << timer.getElapsedSeconds() << std::endl;

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
                return static_cast<float>(covMat.value()(id, id));
            }
        };
        
        std::vector<float> verticesValues(quad.getNumVertices());        
        for(uint32_t i=0; i < quad.getNumVertices(); i++)
        {
            verticesValues[i] = getVertValue(i);
        }

        if(config.computeVariance)
        {
            std::vector<float> verticesCov(config.computeVariance ? quad.getNumVertices() : 0);
            for(uint32_t i=0; i < quad.getNumVertices(); i++)
            {
                verticesCov[i] = getVertCov(i);
            }
            outCovariance = std::optional<LinearNodeTree<Dim>>(LinearNodeTree<Dim>(NodeTree<Dim>(quad), std::move(verticesCov)));
        }

        outP = std::optional<Eigen::SparseMatrix<double>>(std::move(P));
        outN = std::optional<Eigen::SparseMatrix<double>>(std::move(N));
        outS = std::optional<Eigen::SparseMatrix<double>>(std::move(S));
        outW = std::optional<Eigen::MatrixXd>(x);

        return std::make_unique<LinearNodeTree<Dim>>(std::move(quad), std::move(verticesValues));
    }

    template<uint32_t Dim>
    double evaulatePosteriorFunc(const PointCloud<Dim>& cloud, Config<Dim> config,
                                Eigen::SparseMatrix<double>& P, Eigen::SparseMatrix<double>& N, Eigen::SparseMatrix<double>& S,
                                Eigen::VectorXd& w)
    {
        EigenSparseMatrix pointsCovMatrix(cloud.size());

        // Position Gaussian
        Eigen::VectorXd Pw = P * w;
        double gv = 0.0;
        double det = 1.0;
        double power = 1.0;

        for(uint32_t i = 0; i < cloud.getPoints().size(); i++)
        {
            gv += Pw(i) * Pw(i) / cloud.variance(i);
            // det *= cloud.variance(i);
            // power *= 2.0 * M_PI;
        }

        // double positionValue = 1.0 / (glm::sqrt(power * det)) * glm::exp(-0.5 * gv);
        double positionValue = -0.5 * gv;

        // Normals Gaussian
        Eigen::VectorXd Nw = N * w;

        gv = 0.0;
        det = 1.0;
        power = 1.0;
        double invVarianceGradient = config.gradientWeight  * config.gradientWeight;
        double varianceGradient = 1.0 / invVarianceGradient;
        for(uint32_t i = 0; i < cloud.getPoints().size(); i++)
        {
            for(uint32_t j = 0; j < 2; j++)
            {
                gv += invVarianceGradient * (Nw(2 * i + j) - cloud.normal(i)[j]) * (Nw(2 * i + j) - cloud.normal(i)[j]);
                // det *= varianceGradient;
                // power *= 2.0 * M_PI;
            }
        }

        // double gradientValue = 1.0 / (glm::sqrt(power * det)) * glm::exp(-0.5 * gv);
        double gradientValue = -0.5 * gv;

        // Smoothness Gaussian
        Eigen::VectorXd Sw = S * w;

        gv = 0.0;
        det = 1.0;
        power = 1.0;
        double invSmoothnessGradient = config.smoothWeight  * config.smoothWeight;
        double varianceSmoothness = 1.0 / invSmoothnessGradient;
        for(uint32_t i = 0; i < Sw.rows(); i++)
        {
            gv += invSmoothnessGradient * Sw(i) * Sw(i);
            // det *= varianceSmoothness;
            // power *= 2.0 * M_PI;
        }

        // double smoothnessValue = 1.0 / (glm::sqrt(power * det)) * glm::exp(-0.5 * gv);
        double smoothnessValue = -0.5 * gv;

        return glm::exp(positionValue + gradientValue + smoothnessValue);
    }

    template<uint32_t Dim>
    std::unique_ptr<CubicNodeTree<Dim>> compute2DCubicNodeTree(NodeTree<Dim>&& quad, const PointCloud<Dim>& cloud, Config<Dim> config,
                                                               std::optional<std::reference_wrapper<std::vector<float>>> outNodeTreeVertexEnergyValues = std::optional<std::reference_wrapper<std::vector<float>>>(),
                                                               std::optional<std::reference_wrapper<std::vector<float>>> outEigenValues = std::optional<std::reference_wrapper<std::vector<float>>>())
    {
        using NodeTreeConfig = NodeTree<Dim>::Config;
        using Inter = BicubicInterpolation;
        using Node = NodeTree<Dim>::Node;
        using vec = ScalarField<Dim>::vec;

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
        auto A = 2.0f * pM.transpose() * pM + nM;
        auto b = 2.0f * pM.transpose() * pb;
        for (double& v : unknownsValues) v = 0.0;
        auto x = Eigen::Map<Eigen::VectorXd>(unknownsValues.data(), unknownsValues.size());
        std::cout << (A*x - b).squaredNorm() << std::endl;
        // std::vector<float> solverIterValues;
        // std::vector<float> solverIterSteps;
        // EigenSolver::GD::solve(A, b, x, &solverIterValues, &solverIterSteps);
        // write_array_to_file(solverIterValues, "solverIterValues.bin");
        // write_array_to_file(solverIterSteps, "solverIterSteps.bin");

        EigenSolver::BiCGSTAB::solve(A, b, x);
        // EigenSolver::CG::solve(A, b, x);

        std::cout << "Final error: " << (A*x - b).squaredNorm() << std::endl;


        // if(outNodeTreeVertexEnergyValues || outEigenValues)
        if(false)
        {
            Eigen::SelfAdjointEigenSolver<Eigen::SparseMatrix<double>> eigenSolver;
            eigenSolver.compute(A, Eigen::DecompositionOptions::ComputeEigenvectors);
            Eigen::VectorXd eigenValues = eigenSolver.eigenvalues();
            auto eigenVectors = eigenSolver.eigenvectors();

            if(outEigenValues)
            {
                outEigenValues->get().resize(eigenValues.size());
                for(uint32_t i=0; i < eigenValues.size(); i++) // Copy array
                {
                    outEigenValues->get()[i] = eigenValues[i];
                }
            }

            // std::vector<uint32_t> eigenIndices(eigenValues.size());
            // for(uint32_t i=0; i < eigenIndices.size(); i++)
            // {
            //     eigenIndices[i] = i;
            // }

            // std::sort(eigenIndices.begin(), eigenIndices.end(), [&](const uint32_t& a, const uint32_t& b) -> bool {
            //     return eigenValues[a] > eigenValues[b];
            // });
            
            // std::vector<float> unknownsEnergy(numUnknows, 0.0f);

            // for(uint32_t i=0; i < eigenValues.size(); i++)
            // {
            //     const uint32_t eIdx = eigenIndices[i];
            //     for(uint32_t ei=0; ei < numUnknows; ei++)
            //     {
            //         unknownsEnergy[ei] += glm::abs(eigenVectors.col(eIdx)(ei)) * glm::abs(eigenValues[eIdx]);
            //     }
            // }
            
            // std::vector<float>& treeVertexEnergy = outNodeTreeVertexEnergyValues.value();
            // treeVertexEnergy = std::vector<float>(Inter::NumBasis * quad.getNumVertices(), 0.0f);
            // for(uint32_t i = 0; i < quad.getNumVertices(); i++)
            // {
            //     if(vertIdToUnknownVertId[i] == std::numeric_limits<uint32_t>::max()) continue;
            //     for(uint32_t j = 0; j < Inter::NumBasis; j++)
            //     {
            //         treeVertexEnergy[Inter::NumBasis * i + j] = unknownsEnergy[Inter::NumBasis * vertIdToUnknownVertId[i] + j];
            //     }
            // }

            // Invert eigen values
            
            Eigen::MatrixXd same = eigenVectors * eigenVectors.transpose();
            std::cout << "Identity: " << (same - Eigen::MatrixXd::Identity(numUnknows, numUnknows)).squaredNorm() << std::endl;

            Eigen::VectorXd zb = eigenVectors.transpose() * b;
            std::cout << "Norm Z: " << (eigenVectors * zb - b).squaredNorm() << std::endl;
            uint32_t notInverted = 0;
            for(uint32_t i=0; i < eigenValues.rows(); i++)
            {
                if(glm::abs(eigenValues[i]) < 1e20)
                {
                    zb(i) = zb(i) / eigenValues(i);
                }
                else
                {
                    eigenValues(i) = 0.0;
                    notInverted++;
                }
            }

            std::cout << "not inverted " << notInverted << std::endl;
            x = eigenVectors * zb;
            std::cout << "Final error: " << (A*x - b).squaredNorm() << std::endl;


        // double minEigenValue = glm::abs(eigenValues[0]); 
            // double maxEigenValue = glm::abs(eigenValues[0]); 
            // for(uint32_t i=1; i < eigenValues.rows(); i++)
            // {
            //     minEigenValue = glm::min(minEigenValue, glm::abs(eigenValues[i]));
            //     maxEigenValue = glm::max(maxEigenValue, glm::abs(eigenValues[i]));
            // }

            // std::cout << "conditoned number: " << maxEigenValue / minEigenValue << std::endl;

            // auto x = Eigen::Map<Eigen::VectorXd>(unknownsValues.data(), unknownsValues.size());
            // EigenSolver::BiCGSTAB::solve(A, b, x);
        }

        
        // Compute all vertex values
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

        return std::make_unique<CubicNodeTree<Dim>>(std::move(quad), std::move(verticesValues));
    }


    template<uint32_t Dim>
    float computeCubicNodeLoss(const CubicNodeTree<Dim>& tree, const PointCloud<Dim>& cloud,  Config<Dim> config)
    {
        using NodeTreeConfig = NodeTree<Dim>::Config;
        using Inter = BicubicInterpolation;
        using Node = NodeTree<Dim>::Node;
        using vec = ScalarField<Dim>::vec;

        // Compute position and gradient error
        float posError = 0.0f;
        float gradError = 0.0f;
        for(uint32_t i = 0; i < cloud.size(); i++)
        {
            vec nPos = cloud.point(i);
            const float e = tree.eval(nPos);
            posError += e*e;

            vec nNorm = glm::normalize(cloud.normal(i));
            vec n = tree.evalGrad(nPos);
            gradError += glm::dot(nNorm, n);
        }

        // Compute laplacian error
        std::array<std::array<float, Inter::NumBasis>, Inter::NumControlPoints> nodeValues;
        float totalLaplacian = 0.0f;
        for(const Node& node : tree.getNodeTree())
        {
            const vec nodeSize = node.maxCoord - node.minCoord;
            for(uint32_t i=0; i < Inter::NumControlPoints; i++)
            {
                for(uint32_t j=0; j < Inter::NumBasis; j++)
                {
                    nodeValues[i][j] = tree.getVerticesValues()[node.controlPointsIdx[i]][j];
                }
            }
            const float val = BicubicInterpolation::integrateLaplacian(nodeSize, nodeValues);
            totalLaplacian += val;
        }

        float lossError = config.posWeight * posError + config.gradientWeight * gradError + config.smoothWeight * totalLaplacian;

        std::cout << "Total Laplacian: " << totalLaplacian << std::endl;
        std::cout << "Loss: " << lossError << std::endl;

        return lossError;
    }
}
#endif