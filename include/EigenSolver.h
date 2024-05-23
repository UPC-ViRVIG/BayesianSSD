#ifndef EIGEN_SOLVER
#define EIGEN_SOLVER

#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

namespace EigenSolver
{
    struct BiCGSTAB 
    {
        template<typename TA, typename TB, typename TR>
        static void solve(TA& matrixA, TB& vectorB, TR& result)
        {
            Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
            solver.compute(matrixA);
            result = solver.solve(vectorB);
        }
    };
};

#endif