#ifndef EIGEN_SPARSE_MATRIX
#define EIGEN_SPARSE_MATRIX

#include <iostream>
#include <vector>
#include <Eigen/Sparse>

class EigenVector
{
public:
    EigenVector() {}

    void addTerm(float value)
    {
        vector.push_back(value);
    }

    auto getVector()
    {
        return Eigen::Map<Eigen::VectorXd>(vector.data(), vector.size());
    }

private:
    std::vector<double> vector;
};

class EigenSparseMatrix
{
public:
    EigenSparseMatrix(uint32_t numUnknowns) : nUnknowns(numUnknowns), eqIdx(0) {}
    EigenSparseMatrix(uint32_t numEq, uint32_t numUnknowns) : nUnknowns(numUnknowns), eqIdx(numEq) {}

    void addTerm(uint32_t unknownId, float coeff)
    {
        matrixTriplets.push_back(Eigen::Triplet<double>(eqIdx, unknownId, static_cast<double>(coeff)));
    }

    void addTerm(uint32_t eqIdx, uint32_t unknownId, float coeff)
    {
        this->eqIdx = glm::max(this->eqIdx, eqIdx+1);
        matrixTriplets.push_back(Eigen::Triplet<double>(eqIdx, unknownId, static_cast<double>(coeff)));
    }

    void addTerms(std::initializer_list<std::tuple<uint32_t, float>> listcoeff)
    {
        for(auto& term : listcoeff)
        {
            addTerm(std::get<0>(term), std::get<1>(term));
        }
    }

    void endEquation() { eqIdx++; }

    Eigen::SparseMatrix<double> getMatrix()
    {
        const uint32_t nEquations = eqIdx;
        Eigen::SparseMatrix<double> A(nEquations, nUnknowns);
        A.setFromTriplets(matrixTriplets.begin(), matrixTriplets.end());
        return A;
    }

private:
    std::vector<Eigen::Triplet<double>> matrixTriplets;
    uint32_t nUnknowns;
    uint32_t eqIdx;
};

#endif