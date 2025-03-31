#ifndef SMOOTH_SURFACE_RECONSTRUCTION_H
#define SMOOTH_SURFACE_RECONSTRUCTION_H

#include <functional>
#include <memory>
#include <random>
#include "NodeTree.h"
#include "ScalarField.h"
#include "InterpolationMethod.h"
#include "EigenSparseMatrix.h"
#include "EigenSolver.h"
#include "EigenDecompositionLaplacian.h"
#include <Eigen/Eigenvalues>
#include "Timer.h"
#ifdef LOW_RANK_SVD_AVAILABLE
extern "C"
{
#include "low_rank_svd_algorithms_gsl.h"
#undef min
#undef max
}
#endif

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

// Function to write an Eigen matrix to a binary file
void writeMatrixToFile(const Eigen::MatrixXd& matrix, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        // Write dimensions
        int rows = matrix.rows();
        int cols = matrix.cols();
        file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(int));

        // Write data
        file.write(reinterpret_cast<const char*>(matrix.data()), matrix.size() * sizeof(double));
        file.close();
        std::cout << "Matrix written to " << filename << std::endl;
    } else {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
    }
}

namespace SmoothSurfaceReconstruction
{
    enum Algorithm
    {
        VAR,
        BAYESIAN,
        GP   
    };

    enum InverseAlgorithm
    {
        FULL,
        BASE_RED,
        BASE_RED_QR,
        LOW_RANK
    };

    template<uint32_t Dim>
    struct Config
    {
        float posWeight;
        float gradientWeight;
        float gradientXYWeight;
        float smoothWeight;
        Algorithm algorithm;
        bool computeVariance;
        InverseAlgorithm invAlgorithm;
        uint32_t invRedMatRank;
    };

    template<uint32_t Dim>
    std::unique_ptr<LinearNodeTree<Dim>> computeLinearNodeTree(NodeTree<Dim>&& quad, const PointCloud<Dim>& cloud, Config<Dim> config,
                                                               std::optional<LinearNodeTree<Dim>>& outCovariance, std::vector<glm::vec3>& vertices)
    {
        std::optional<Eigen::MatrixXd> invCovMat;
        std::optional<Eigen::MatrixXd> covMat;
        std::optional<Eigen::SparseMatrix<double>> outP;
        std::optional<Eigen::SparseMatrix<double>> outN;
        std::optional<Eigen::SparseMatrix<double>> outS;
        std::optional<Eigen::VectorXd> outW;
        return computeLinearNodeTree(std::move(quad), cloud, config, outCovariance, invCovMat, covMat, outP, outN, outS, outW, vertices);
    }
 
    template<uint32_t Dim>
    std::unique_ptr<LinearNodeTree<Dim>> computeLinearNodeTree(NodeTree<Dim>&& quad, const PointCloud<Dim>& cloud, Config<Dim> config,
                                                               std::optional<LinearNodeTree<Dim>>& outCovariance,
                                                               std::optional<Eigen::MatrixXd>& invCovMat,
                                                               std::optional<Eigen::MatrixXd>& covMat,
                                                               std::optional<Eigen::SparseMatrix<double>>& outP,
                                                               std::optional<Eigen::SparseMatrix<double>>& outN,
                                                               std::optional<Eigen::SparseMatrix<double>>& outS,
                                                               std::optional<Eigen::VectorXd>& outW,
                                                               std::vector<glm::vec3>& vertices)
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

        std::function<void(EigenSparseMatrix& mat, uint32_t, uint32_t, float)> setUnkownTerm;
        setUnkownTerm = [&](EigenSparseMatrix& mat, uint32_t vertId1, uint32_t vertId2, float value)
        {
            if(vertIdToUnknownVertId[vertId1] == std::numeric_limits<uint32_t>::max())
            {
                const ConstrainedUnknown& constraint = constraints[vertId1];
                for(uint32_t i=0; i < constraint.numOperators; i++)
                {
                    setUnkownTerm(mat, constraint.vertIds[i], vertId2, value * constraint.weights[i]);
                }
            }
            else
            {
                if(vertIdToUnknownVertId[vertId2] == std::numeric_limits<uint32_t>::max())
                {
                    const ConstrainedUnknown& constraint = constraints[vertId2];
                    for(uint32_t i=0; i < constraint.numOperators; i++)
                    {
                        setUnkownTerm(mat, vertId1, constraint.vertIds[i], value * constraint.weights[i]);
                    }
                }
                else
                {
                    mat.addTerm(vertIdToUnknownVertId[vertId1], vertIdToUnknownVertId[vertId2], value);
                }
            }
        };

        EigenSparseMatrix pointsMatrix(numUnknows);
        EigenVector pointsCovVector;
        EigenSparseMatrix gradientMatrix(numUnknows);
        EigenVector gradientVector;
        EigenSparseMatrix gradientInvCovMatrix(Dim * cloud.size());
        float invVarGradient = config.gradientWeight * config.gradientWeight;
        float invVarXYGradient = config.gradientXYWeight * config.gradientXYWeight;
        // Generate point equations
        for(uint32_t p = 0; p < cloud.size(); p++)
	    {
            vec nPos = cloud.point(p);
            std::optional<Node> node;
            quad.getNode(nPos, node);
            if(!node) continue;
            std::array<float, Inter::NumControlPoints> weights;
            const vec nnPos = node->transformToLocalCoord(nPos);
            const vec nodeSize = node->maxCoord - node->minCoord;

            // Points position
            Inter::eval(nnPos, weights);

            for(uint32_t i = 0; i < Inter::NumControlPoints; i++)
                setUnkownValue(pointsMatrix, node->controlPointsIdx[i], weights[i]);

            pointsCovVector.addTerm(cloud.variance(p));
            pointsMatrix.endEquation();

            // Points gradient
            vec nNorm = glm::normalize(cloud.normal(p));
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

            const std::tuple<Eigen::Matrix<double, Dim, Dim>, Eigen::Vector<double, Dim>>& nData = cloud.normalInvCovarianceDes(p);
            Eigen::Vector<double, Dim> normInvCov = std::get<1>(nData);
            normInvCov(Dim-1) = normInvCov(Dim-1) * invVarGradient;
            for(uint32_t j=0; j < Dim-1; j++) normInvCov(j) = normInvCov(j) * invVarXYGradient;
            Eigen::Matrix<double, Dim, Dim> covMat = std::get<0>(nData).transpose() * normInvCov.asDiagonal() * std::get<0>(nData);
            // auto covMat1 = cloud.normalInvCovariance(p);
            for(uint32_t j=0; j < Dim; j++)
            {
                for(uint32_t i=0; i < Dim; i++)
                {
                    gradientInvCovMatrix.addTerm(Dim*p + j, Dim*p + i, covMat(j, i));
                }
            }
        }

        // Generate Node equations
        // constexpr bool extrapolateLaplacian = false;
        EigenSparseMatrix smoothMatrix1(numUnknows);
        EigenVector smoothFactor1;
        EigenSparseMatrix smoothMatrix2(numUnknows);
        EigenVector smoothFactor2;

        const unsigned int numNodesAtMaxDepth = 1 << quad.getMaxDepth();
        const vec quadSize = quad.getMaxCoord() - quad.getMinCoord();
        const vec nodeSize = quadSize / static_cast<float>(numNodesAtMaxDepth) - vec(1e-6);
        const float invSizeSq = 1.0f / (nodeSize[0] * nodeSize[0]);
        double totalVolume = 1.0;
        for(uint32_t d=0; d < Dim; d++) totalVolume *= quadSize[d];
        const double invTotalVolume = 1.0 / totalVolume; 
        // TODO: the area does not sum, there is a margin added bewteen nodes of different size
        for(uint32_t i = 0; i < quad.getNumVertices(); i++)
        {
            if(vertIdToUnknownVertId[i] == std::numeric_limits<uint32_t>::max())
                continue;

            std::array<Node, NodeTree<Dim>::NumAdjacentNodes> nodes;
            uint32_t numNodes = quad.getAdjacentNodes(i, nodes);
            std::array<std::array<float, 2>, Dim> dirMaxSizes;
            std::array<std::array<uint32_t, 2>, Dim> dirMinDepth;
            dirMaxSizes.fill({0.0f, 0.0f});
            dirMinDepth.fill({0, 0});
            for(uint32_t j=0; j < numNodes; j++)
            {
                const vec size = nodes[j].maxCoord - nodes[j].minCoord;
                const vec center = 0.5f * (nodes[j].maxCoord + nodes[j].minCoord);
                for(uint32_t d=0; d < Dim; d++)
                {
                    const uint32_t sId = center[d] < quad.getVertices()[i][d] ? 0 : 1;
                    float& v = dirMaxSizes[d][sId];
                    if(v < size[d])
                    {
                        v = size[d];
                        dirMinDepth[d][sId] = nodes[j].depth;
                    }
                }
            }

            for(uint32_t d=0; d < Dim; d++)
            {
                // uint32_t numValidSides = 0;
                vec side;
                double inteArea = 1.0;
                for(uint32_t j=0; j < Dim; j++)
                {
                    side[j] = (d == j) ? 1.0f : 0.0f;
                    if(d != j) inteArea *= (0.5 * dirMaxSizes[j][0] + 0.5 * dirMaxSizes[j][1]);
                }                    
                
                std::optional<Node> node;
                const bool valid = dirMinDepth[d][0] > 0 && dirMinDepth[d][1] > 0;

                if(valid)
                {
                    // Calculate weights
                    const double cw = 2.0f * inteArea * invTotalVolume;
                    double minPartWeight = 0.5 * glm::pow(2.0, static_cast<int>(dirMinDepth[d][1])-static_cast<int>(dirMinDepth[d][0]));
                    double maxPartWeight = 0.5 * glm::pow(2.0, static_cast<int>(dirMinDepth[d][0])-static_cast<int>(dirMinDepth[d][1]));

                    std::array<float, Inter::NumControlPoints> weights;
                    // -1
                    vec coord = quad.getVertices()[i] - dirMaxSizes[d][0] * side;
                    quad.getNode(coord, node);
                    if(!node)
                    {
                        continue;
                    }
                    Inter::eval(node->transformToLocalCoord(coord), weights);
                    const double w11 = 0.5 / dirMaxSizes[d][0];
                    const double w12 = maxPartWeight / dirMaxSizes[d][1];
                    for(uint32_t j=0; j < Inter::NumControlPoints; j++)
                    {
                        if(glm::abs(weights[j]) > 1e-8)
                        {
                            setUnkownValue(smoothMatrix1, node->controlPointsIdx[j], w11 * weights[j]);
                            setUnkownValue(smoothMatrix2, node->controlPointsIdx[j], w12 * weights[j]);
                        }
                    }

                    // 1
                    coord = quad.getVertices()[i] + dirMaxSizes[d][1] * side;
                    quad.getNode(coord, node);
                    if(!node)
                    {
                        continue;
                        // vec tVec = (coord - quad.getMinCoord()) / (quad.getMaxCoord() - quad.getMinCoord());
                        // std::cout << tVec.x << " " << tVec.y << " " << tVec.z << std::endl;
                    }
                    Inter::eval(node->transformToLocalCoord(coord), weights);
                    const double w21 = minPartWeight / dirMaxSizes[d][0];
                    const double w22 = 0.5 / dirMaxSizes[d][1];
                    for(uint32_t j=0; j < Inter::NumControlPoints; j++)
                    {
                        if(glm::abs(weights[j]) > 1e-8)
                        {
                            setUnkownValue(smoothMatrix1, node->controlPointsIdx[j], w21 * weights[j]);
                            setUnkownValue(smoothMatrix2, node->controlPointsIdx[j], w22 * weights[j]);
                        }
                    }

                    setUnkownValue(smoothMatrix1, i, -(w11+w21));
                    smoothFactor1.addTerm(2.0 * cw / dirMaxSizes[d][0]);
                    smoothMatrix1.endEquation();
                    setUnkownValue(smoothMatrix2, i, -(w12+w22));
                    smoothFactor2.addTerm(2.0 * cw / dirMaxSizes[d][1]);
                    smoothMatrix2.endEquation();
                }
			}
        }

        EigenSparseMatrix smoothMatrix3(numUnknows);
        std::array<std::array<float, Inter::NumControlPoints>, Inter::NumControlPoints> sGradWeights;
        Inter::evalSecondGradIntegGrad(sGradWeights);
        for(const Node& n : quad)
        {
            const double factor = invTotalVolume * Inter::factorSecondGradIntegGrad(n.maxCoord - n.minCoord);
            for(uint32_t i=0; i < Inter::NumControlPoints; i++)
            {
                for(uint32_t j=0; j < Inter::NumControlPoints; j++)
                {
                    if(glm::abs(sGradWeights[j][i]) > 1e-8)
                    {
                        setUnkownTerm(smoothMatrix3, n.controlPointsIdx[i], n.controlPointsIdx[j], factor * sGradWeights[j][i]);
                    }
                }
            }
        }

        Eigen::SparseMatrix<double> P = pointsMatrix.getMatrix();
        pointsMatrix = EigenSparseMatrix(0, 0); // Free memory
        auto covP = pointsCovVector.getVector();

        Eigen::SparseMatrix<double> N = gradientMatrix.getMatrix();
        gradientMatrix = EigenSparseMatrix(0, 0); // Free memory
        Eigen::SparseMatrix<double> iCovN = gradientInvCovMatrix.getMatrix();
        gradientInvCovMatrix = EigenSparseMatrix(0, 0); // Free memory

        Eigen::SparseMatrix<double> S1 = smoothMatrix1.getMatrix();
        smoothMatrix1 = EigenSparseMatrix(0, 0); // Free memory
        auto fS1 = smoothFactor1.getVector();
        Eigen::SparseMatrix<double> S2 = smoothMatrix2.getMatrix();
        smoothMatrix2 = EigenSparseMatrix(0, 0); // Free memory
        auto fS2 = smoothFactor2.getVector();
        Eigen::SparseMatrix<double> S3 = smoothMatrix3.getMatrix();
        smoothMatrix3 = EigenSparseMatrix(0, 0); // Free memory
        Eigen::SparseMatrix<double> dS = S1.transpose() * fS1.asDiagonal() * S1 + S2.transpose() * fS2.asDiagonal() * S2 + S3;
        S1 = Eigen::SparseMatrix<double>(); S2 = Eigen::SparseMatrix<double>(); S3 = Eigen::SparseMatrix<double>();
        float invVarSmoothing = config.smoothWeight * config.smoothWeight;

        invVarGradient = 1.0;
        Eigen::VectorXd b = invVarGradient * N.transpose() * iCovN * gradientVector.getVector();
        Eigen::SparseMatrix<double> A = P.transpose() * covP.asDiagonal().inverse() * P + invVarGradient * N.transpose() * iCovN * N + invVarSmoothing * dS;
        // Eigen::SparseMatrix<double> A = P.transpose() * covP.asDiagonal().inverse() * P + invVarGradient * N.transpose() * N + invVarSmoothing * (S.transpose() * S);
        std::cout << "Time setting the problem: " << timer.getElapsedSeconds() << std::endl;

        std::vector<double> unknownsValues(numUnknows, 1.0f);

        std::cout << "Matrix: " << A.rows() << " x " << A.cols() << std::endl;
        std::cout << "Sparsity: " << A.nonZeros() << " / " << A.cols() * A.rows() << std::endl;
        
        timer.start();

        unknownsValues = std::vector<double>(numUnknows, 0.0f);
        auto x = Eigen::Map<Eigen::VectorXd>(unknownsValues.data(), unknownsValues.size());

        // EigenSolver::BiCGSTAB::solve(A, b, x);
        EigenSolver::CG::solve(A, b, x);

        std::cout << "Time solving problem: " << timer.getElapsedSeconds() << std::endl;

        // std::cout << "Smooth value" << (dS * x).sum() << std::endl;
        Eigen::VectorXd Px = P * x;
        Eigen::VectorXd Nx = N * x;
        double xPx = 0.0; 
        for(uint32_t i=0; i < Px.rows(); i++)
        {
            const double v = Px(i);
            xPx += glm::sqrt(v * v / covP(i));
        }
        xPx = xPx / static_cast<double>(Px.rows());
        Eigen::Vector<double, Dim> xNxD = Eigen::Vector<double, Dim>::Zero();
        double meanMag = 0.0f;
        for(uint32_t i=0; i < Nx.rows(); i+=Dim)
        {
            Eigen::Vector<double, Dim> g;
            for(uint32_t d=0; d < Dim; d++) g(d) = Nx(i+d);
            meanMag += g.squaredNorm();
            auto& icovT = cloud.normalInvCovarianceDes(i/3);
            g = std::get<0>(icovT) * g;
            for(uint32_t d=0; d < Dim; d++) 
            {
                const double sq = invVarGradient * std::get<1>(icovT)(d) * g(d) * g(d);
                xNxD(d) += glm::sqrt(sq);
            }
        }
        std::cout << meanMag / static_cast<double>(Nx.rows()/Dim) << std::endl;
        std::cout << xPx << std::endl;
        if(Dim == 3)
        {
            std::cout << (0.5 * (xNxD(0) + xNxD(1)) / static_cast<double>(Nx.rows()/Dim)) << std::endl;
            std::cout << xNxD(2) / static_cast<double>(Nx.rows()/Dim) << std::endl;
        }
        if(Dim == 2)
        {
            std::cout << xNxD(0) / static_cast<double>(Nx.rows()/Dim) << std::endl;
            std::cout << xNxD(1) / static_cast<double>(Nx.rows()/Dim) << std::endl;
        }
        std::cout << invVarSmoothing * x.transpose() * dS * x << std::endl;
        std::cout << "Time solving problem: " << timer.getElapsedSeconds() << std::endl;

        P = Eigen::SparseMatrix<double>();
        N = Eigen::SparseMatrix<double>();
        dS = Eigen::SparseMatrix<double>();

        timer.start();

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

        Eigen::VectorXd CovX;
        if(config.computeVariance)
        {
            timer.start();
            using RowMMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
            RowMMatrixXd invSVDmat;
            Eigen::MatrixXd conv;
            const uint32_t K=config.invRedMatRank;
            switch(config.invAlgorithm)
            {
                case FULL:
                    //break; // CHANGED
                case BASE_RED:
                case BASE_RED_QR:
                    {
                        Eigen::MatrixXd SVDmat;
                        Eigen::BDCSVD<Eigen::MatrixXd> svd;
                        if(config.invAlgorithm == BASE_RED || config.invAlgorithm == BASE_RED_QR)
                        {
                            std::vector<glm::ivec3> gridPoints(numUnknows);
                            vec bbSize = quad.getMaxCoord() - quad.getMinCoord();
                            for(uint32_t i=0; i < numUnknows; i++)
                            {
                                const uint32_t vId = unknownVertIdToVertId[i];
                                vec p = glm::round(static_cast<float>(numNodesAtMaxDepth) * (quad.getVertices()[vId] - quad.getMinCoord()) / bbSize);
                                gridPoints[i] = glm::ivec3(0);
                                for(uint32_t d=0; d < Dim; d++) gridPoints[i][d] = static_cast<int>(p[d]);
                            }
                            if(Dim == 2)
                            {
                                conv = EigenDecompositionLaplacian::getMatrix(glm::ivec3(numNodesAtMaxDepth+1, numNodesAtMaxDepth+1, 1), gridPoints, K);
                            }
                            else
                            {
                                conv = EigenDecompositionLaplacian::getMatrix(glm::ivec3(numNodesAtMaxDepth+1), gridPoints, K);

                                // std::random_device rd; // Obtain a random seed from the hardware
                                // std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
                                // std::normal_distribution<> dis(0.0, 1.0); // Define the normal distribution
                                // conv = Eigen::MatrixXd(numUnknows, K);
                                // for(uint32_t i=0; i < numUnknows; i++)
                                // {
                                //     for(uint32_t j=0; j < K; j++)
                                //     {
                                //         conv(i, j) = dis(gen);
                                //     }
                                // }
                                // conv = A * conv;
                            }

                            // std::cout << "QR" << std::endl;
                            if(config.invAlgorithm == BASE_RED_QR)
                            {
                                std::cout << "QR" << std::endl;
                                Eigen::HouseholderQR<Eigen::MatrixXd> qr(conv);
                                auto thinQ = Eigen::MatrixXd::Identity(conv.rows(), conv.cols());
                                Eigen::MatrixXd q = qr.householderQ();
                                conv = qr.householderQ() * thinQ;
                            }
                            // writeMatrixToFile(thinQ, "conv.bin");

                            // RowMMatrixXd convRM = conv;
                            // gsl_matrix *glsConv = gsl_matrix_calloc(convRM.rows(), convRM.cols());
                            // std::memcpy(glsConv->data, convRM.data(), convRM.rows() * convRM.cols() * sizeof(double));

                            // gsl_matrix *glsThinQ = gsl_matrix_calloc(convRM.rows(), convRM.cols());

                            // timer.start();
                            // std::cout << "Start QR" << std::endl;
                            // QR_factorization_getQ(glsConv, glsThinQ);
                            // std::cout << "QR facto: " << timer.getElapsedSeconds() << std::endl;

                            // conv = Eigen::Map<RowMMatrixXd>(glsThinQ->data, glsThinQ->size1, glsThinQ->size2);
                            // writeMatrixToFile(conv, "conv.bin");

                            Eigen::MatrixXd SVDredMat = conv.transpose() * (A * conv);
                            std::cout << "Red: " << timer.getElapsedSeconds() << std::endl;
                            svd = Eigen::BDCSVD<Eigen::MatrixXd>(SVDredMat, Eigen::ComputeThinU | Eigen::ComputeThinV);
                            std::cout << "SVD: " << timer.getElapsedSeconds() << std::endl;
                        }
                        else
                        {
                            SVDmat = Eigen::MatrixXd(A);
                            svd = Eigen::BDCSVD<Eigen::MatrixXd>(SVDmat, Eigen::ComputeThinU | Eigen::ComputeThinV);
                        }
                        
                        Eigen::VectorXd sv = svd.singularValues();
                        
                        if(false && config.invAlgorithm == FULL)
                        {
                            std::vector<uint32_t> svIndices(sv.size());
                            for(uint32_t i=0; i < sv.size(); i++) svIndices[i] = i;
                            std::sort(svIndices.begin(), svIndices.end(), [&](uint32_t id1, uint32_t id2)
                            {
                                return sv(id1) < sv(id2);
                            });

                            for(uint32_t i=0; i < K; i++)
                            {
                                if(glm::abs(sv(svIndices[i])) > 1e-8)
                                {
                                    sv(svIndices[i]) = 1.0 / sv(svIndices[i]);
                                }
                                else
                                {
                                    sv(svIndices[i]) = 0.0;
                                }
                            }
                            for(uint32_t i=K; i < sv.size(); i++) sv(svIndices[i]) = 0.0;
                        }
                        else
                        {
                            for(uint32_t i=0; i < sv.size(); i++)
                            {
                                if(glm::abs(sv(i)) > 1e-8)
                                {
                                    sv(i) = 1.0 / sv(i);
                                }
                                else
                                {
                                    sv(i) = 0.0;
                                }
                            }
                        }                        

                        if(config.invAlgorithm == BASE_RED || config.invAlgorithm == BASE_RED_QR)
                        {
                            invSVDmat = svd.matrixV() * sv.asDiagonal() * svd.matrixU().adjoint();
                        }
                        else
                        {
                            invSVDmat = svd.matrixV() * sv.asDiagonal() * svd.matrixU().adjoint();
                        }
                    }
                    break;
                case LOW_RANK:
#ifdef LOW_RANK_SVD_AVAILABLE                
                    {
                        Eigen::MatrixXd SVDred(A.transpose());
                        gsl_matrix svdmat;
                        svdmat.size1 = SVDred.rows();
                        svdmat.size2 = SVDred.cols();
                        svdmat.tda = SVDred.rows();
                        svdmat.data = SVDred.data();
                        svdmat.owner = 0;
                        svdmat.block = NULL;

                        gsl_matrix *gU, *gS, *gV;
                        timer.start();
                        randomized_low_rank_svd1(&svdmat, K, &gU, &gS, &gV);
                        std::cout << timer.getElapsedSeconds() << std::endl;

                        for(uint32_t i=0; i < K; i++)
                        {
                            double value = gsl_matrix_get(gS, i, i);
                            if(glm::abs(value) > 1e-8)
                            {
                                gsl_matrix_set(gS, i, i, 1.0 / value);
                            }
                            else
                            {
                                gsl_matrix_set(gS, i, i, 0.0);
                            }
                        }

                        using RowMMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
                        auto eU = Eigen::Map<RowMMatrixXd>(gU->data, gU->size1, gU->size2);
                        auto eS = Eigen::Map<RowMMatrixXd>(gS->data, gS->size1, gS->size2);
                        auto eV = Eigen::Map<RowMMatrixXd>(gV->data, gV->size1, gV->size2);

                        invSVDmat = eV * eS.diagonal() * eU.transpose();
                    }
#else
                    std::cout << "LowRankSVD library is not included" << std::endl;
#endif
                    break;
            }

            CovX = Eigen::VectorXd::Zero(numUnknows);
            switch(config.algorithm)
            {
                case VAR:
                    {
                        std::cout << "S" << std::endl;
                        auto sA = P.transpose() * covP.asDiagonal().inverse() * P + invVarGradient * N.transpose() * iCovN * N;
                        if(config.invAlgorithm == BASE_RED || config.invAlgorithm == BASE_RED_QR)
                        {
                            std::cout << "M1" << std::endl;
                            timer.start();
                            RowMMatrixXd CAC = conv.transpose() * sA * conv;
                            CAC = invSVDmat * (CAC * invSVDmat.transpose());
                            Eigen::MatrixXd svdC = CAC * conv.transpose();
                            std::cout << timer.getElapsedSeconds() << std::endl;
                            std::cout << "M2" << std::endl;
                            timer.start();
                            for(uint32_t i=0; i < numUnknows; i++) // Compute only the diagonal
                            {
                                CovX(i) = conv.row(i).dot(svdC.col(i));
                            }
                            std::cout << timer.getElapsedSeconds() << std::endl;
                        }
                        else
                        {
                            std::cout << "M1" << std::endl;
                            timer.start();
                            Eigen::MatrixXd sASVD = sA * invSVDmat.transpose();
                            std::cout << timer.getElapsedSeconds() << std::endl;
                            std::cout << "M2" << std::endl;
                            timer.start();
                            for(uint32_t i=0; i < numUnknows; i++) // Compute only the diagonal
                            {
                                CovX(i) = invSVDmat.row(i).dot(sASVD.col(i));
                            }
                            std::cout << timer.getElapsedSeconds() << std::endl;
                        }
                    }
                    break;
                case BAYESIAN:
                    if(config.invAlgorithm == BASE_RED || config.invAlgorithm == BASE_RED_QR)
                    {
                        std::cout << "Final part: " << timer.getElapsedSeconds() << std::endl;
                        Eigen::MatrixXd svdC = invSVDmat * conv.transpose();
                        for(uint32_t i=0; i < numUnknows; i++) // Compute only the diagonal
                        {
                            CovX(i) = conv.row(i).dot(svdC.col(i));
                        }
                        std::cout << "Time computing covariance: " << timer.getElapsedSeconds() << std::endl;
                        // Eigen::MatrixXd invMat = conv * svdC;
                        // writeMatrixToFile(invMat, "invSVDmat.bin");
                    }
                    else
                    {
                        // Eigen::MatrixXd resInv(numUnknows, numUnknows);
                        // auto identityMat = Eigen::MatrixXd::Identity(numUnknows, numUnknows);
                        // EigenSolver::CG::solve(A, identityMat, resInv);
                        // CovX = resInv.diagonal();

                        CovX = invSVDmat.diagonal();
                        std::cout << "Time computing covariance: " << timer.getElapsedSeconds() << std::endl;
                        // writeMatrixToFile(invSVDmat, "invSVDmat.bin");
                        // Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> solver;
                        // solver.compute(A);

                        // Eigen::VectorXd evres(numUnknows);
                        // Eigen::VectorXd evb1 = CovX;
                        // for(uint32_t i=0; i < numUnknows; i++)
                        // {
                        //     evb1(i) = 1.0;
                        //     evres = solver.solve(evb1);
                        //     CovX(i) = evres(i);
                        //     evb1(i) = 0.0;
                        //     if(i % 2000 == 0) std::cout << i << std::endl;
                        // }

                        break;

                        // std::vector<uint32_t> unknownsToCompute;
                        // std::vector<uint32_t> nodesToCompute;
                        // std::function<void(uint32_t)> calculateVertex;
                        // calculateVertex = [&](uint32_t vertId)
                        // {
                        //     if(vertIdToUnknownVertId[vertId] == std::numeric_limits<uint32_t>::max())
                        //     {
                        //         const ConstrainedUnknown& constraint = constraints[vertId];
                        //         float res = 0.0f;
                        //         for(uint32_t i=0; i < constraint.numOperators; i++)
                        //         {
                        //             calculateVertex(constraint.vertIds[i]);
                        //         }
                        //     }
                        //     else unknownsToCompute.push_back(vertIdToUnknownVertId[vertId]);
                        // };

                        // std::cout << "Serach nodes contaning the isosurface" << std::endl;
                        // // float minDist = 0.005 * glm::length(quad.getMaxCoord() - quad.getMinCoord());
                        // for(const Node& n : quad)
                        // {
                        //     // bool allInside = true;
                        //     // bool allOutside = true;
                        //     vec size = quad.getMaxCoord() - quad.getMinCoord();
                        //     vec oCoord = (0.5f * (n.maxCoord + n.minCoord) - quad.getMinCoord()) / size;
                        //     bool hasMinDist = oCoord.y < 0.55f && oCoord.y > 0.45f;
                        //     // for(uint32_t c=0; c < Inter::NumControlPoints; c++)
                        //     // {
                        //     //     const float val = getVertValue(n.controlPointsIdx[c]);
                        //     //     // allInside = allInside && val < 1e-8;
                        //     //     // allOutside = allOutside && val > -1e-8;
                        //     //     hasMinDist = hasMinDist || glm::abs(val) < minDist;
                        //     // }
                        //     // if(!allInside && !allOutside)
                        //     if(hasMinDist)
                        //     {
                        //         for(uint32_t c=0; c < Inter::NumControlPoints; c++)
                        //         {
                        //             calculateVertex(n.controlPointsIdx[c]);
                        //         }
                        //     }
                        // }
                        // std::cout << "Generate Sparse Matrix" << std::endl;
                        // std::sort(unknownsToCompute.begin(), unknownsToCompute.end());
                        // auto endIt = std::unique(unknownsToCompute.begin(), unknownsToCompute.end());
                        // unknownsToCompute.resize(std::distance(unknownsToCompute.begin(), endIt));

                        // std::vector<Eigen::Triplet<double>> matrixTriplets;
                        // for(uint32_t i=0; i < unknownsToCompute.size(); i++)
                        // {
                        //     matrixTriplets.push_back(Eigen::Triplet<double>(unknownsToCompute[i], i, 1.0));
                        // }
                        // Eigen::SparseMatrix<double> evb(numUnknows, unknownsToCompute.size());
                        // std::cout << "setFromTriplets" << std::endl;
                        // evb.setFromTriplets(matrixTriplets.begin(), matrixTriplets.end());
                        // std::cout << "Vertex found " << unknownsToCompute.size() << std::endl;
                        // Eigen::VectorXd evres(numUnknows);
                        // Eigen::VectorXd evb1 = CovX;
                        // for(uint32_t i=0; i < unknownsToCompute.size(); i++)
                        // {
                        //     evb1(i) = 1.0;
                        //     evres = solver.solve(evb1);
                        //     CovX(unknownsToCompute[i]) = evres(i);
                        //     evb1(i) = 0.0;
                        // }
                    }
                    break;
                case GP:
                    // CovX = invSVDmat - invSVDmat * mP.transpose() * mCovP.inverse() * mP * invSVDmat.transpose() + invSVDmat * mN.transpose() * invVarGradient * mN * invSVDmat.transpose();
                    break;
            }
            // covMat = std::optional<Eigen::MatrixXd>(CovX);
        }

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
                return static_cast<float>(CovX(id));
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

        std::cout << "Create node linear tree: " << timer.getElapsedSeconds() << std::endl;

        // outP = std::optional<Eigen::SparseMatrix<double>>(std::move(P));
        // outN = std::optional<Eigen::SparseMatrix<double>>(std::move(N));
        // outS = std::optional<Eigen::SparseMatrix<double>>(std::move(S));
        // outW = std::optional<Eigen::MatrixXd>(x);

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