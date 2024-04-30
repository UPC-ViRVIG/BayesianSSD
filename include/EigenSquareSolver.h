#ifndef EIGEN_SQUARE_SOLVER
#define EIGEN_SQUARE_SOLVER

#include <iostream>
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

class EigenSquareSolver
{
public:
    EigenSquareSolver(uint32_t numUnknowns) : nUnknowns(numUnknowns), eqIdx(0) {}

    void addTerm(uint32_t unknownId, float coeff)
    {
        matrixTriplets.push_back(Eigen::Triplet<double>(eqIdx, unknownId, static_cast<double>(coeff)));
    }

    void addTerms(std::initializer_list<std::tuple<uint32_t, float>> listcoeff)
    {
        for(auto& term : listcoeff)
        {
            addTerm(std::get<0>(term), std::get<1>(term));
        }
    }

    void addConstantTerm(float coeff)
    {
        if(constantTerms.size() > eqIdx)
        {
            std::cerr << "equation constant already added" << std::endl;
            return;
        }
        constantTerms.push_back(coeff);
    }

    void endEquation() { eqIdx++; }

    void solve(std::vector<double>& output)
    {
        const uint32_t nEquations = eqIdx;
        Eigen::SparseMatrix<double> A(nEquations, nUnknowns), AtA;
	    Eigen::VectorXd b(nEquations), Atb(nUnknowns);

        output.resize(nUnknowns);
        auto x = Eigen::Map<Eigen::VectorXd>(output.data(), output.size());

        A.setFromTriplets(matrixTriplets.begin(), matrixTriplets.end());
        for(uint32_t i=0; i < nEquations; i++) b[i] = constantTerms[i];

        AtA = A.transpose() * A;
        Atb = A.transpose() * b;

        Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
        solver.compute(AtA);
        x = solver.solve(Atb);
    }

private:
    std::vector<Eigen::Triplet<double>> matrixTriplets;
    std::vector<double> constantTerms;
    uint32_t nUnknowns;
    uint32_t eqIdx;
};

#endif