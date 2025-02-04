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
            Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::IncompleteLUT<double>> solver;
            solver.compute(matrixA);
            result = solver.solve(vectorB);
            std::cout << "Num Iterations: " << solver.iterations() << std::endl;
        }
    };

    struct CG 
    {
        template<typename TA, typename TB, typename TR>
        static void solve(TA& matrixA, TB& vectorB, TR& result)
        {
            Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> solver;
            solver.compute(matrixA);
            result = solver.solveWithGuess(vectorB, result);
            std::cout << "Num Iterations: " << solver.iterations() << std::endl;
        }

        template<typename TA, typename TB, typename TR>
        static void solve(TA& mat, TB& rhs, TR& x,
                          std::vector<float>* systemValuePerIteration,
                          std::vector<float>* stepSizePerIteration)
        {
            double tol_error = 1e-8;
            uint32_t iters = 0;
            double tol = tol_error;
            uint32_t maxIters = 1000;
            uint32_t n = mat.cols();

            Eigen::VectorXd residual = rhs - mat * x; //initial residual

            double rhsNorm2 = rhs.squaredNorm();

            const double considerAsZero = (std::numeric_limits<double>::min)();
            double threshold = glm::max(double(tol*tol*rhsNorm2),considerAsZero);
            double residualNorm2 = residual.squaredNorm();
            if (residualNorm2 < threshold)
            {
                iters = 0;
                tol_error = sqrt(residualNorm2 / rhsNorm2);
                return;
            }

            Eigen::VectorXd p(n);
            p = residual;      // initial search direction

            Eigen::VectorXd z(n), tmp(n);
            double absNew = residual.dot(p);  // the square of the absolute value of r scaled by invM
            uint32_t i = 0;
            while(i < maxIters)
            {
                tmp.noalias() = mat * p;                    // the bottleneck of the algorithm

                double alpha = absNew / p.dot(tmp);         // the amount we travel on dir
                x += alpha * p;                             // update solution
                residual -= alpha * tmp;                    // update residual
                
                residualNorm2 = residual.squaredNorm();
                if(residualNorm2 < threshold)
                break;
                
                z = residual;                // approximately solve for "A z = residual"

                double absOld = absNew;
                absNew = residual.dot(z);     // update the absolute value of r
                double beta = absNew / absOld;              // calculate the Gram-Schmidt value used to create the new search direction
                p = z + beta * p;                           // update search direction
                i++;
            }
            tol_error = glm::sqrt(residualNorm2 / rhsNorm2);
            iters = i;
        }
    };

    struct GD
    {
        template<typename TA, typename TB, typename TR>
        static void solve(TA& A, TB& b, TR& result,
                          std::vector<float>* systemValuePerIteration = nullptr,
                          std::vector<float>* stepSizePerIteration = nullptr)
        {
            if(systemValuePerIteration) systemValuePerIteration->clear();
            if(stepSizePerIteration) stepSizePerIteration->clear();

            Eigen::VectorXd r, Ar;
            double phi, rr;
            r = b - A * result;
            for(uint32_t i = 0; i < 1000000; i++)
            {
                Ar = A * r;
                rr = r.dot(r);
                phi = rr / r.dot(Ar);
                result = result + phi * r;
                if(rr < 1e-8) break;
                r = r - phi * Ar;

                // Iterations statistic
                if(systemValuePerIteration && i % 10 == 0) 
                {
                    systemValuePerIteration->push_back(static_cast<float>((A * result - b).squaredNorm()));
                }
                if(stepSizePerIteration && i % 10 == 0)
                {
                    stepSizePerIteration->push_back(static_cast<float>(phi));
                }
            }
        }
    };
};

#endif