#ifndef EIGEN_SOLVER
#define EIGEN_SOLVER

#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#ifdef PETSC_AVAILABLE
#include <petscksp.h>
#endif 


namespace EigenSolver
{
    struct BiCGSTAB 
    {
        template<typename TA, typename TB, typename TR>
        static void solve(TA& matrixA, TB& vectorB, TR& result)
        {
            Eigen::BiCGSTAB<TA, Eigen::IncompleteLUT<double>> solver;
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
            Eigen::ConjugateGradient<TA, Eigen::Lower | Eigen::Upper> solver;
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

#ifdef PETSC_AVAILABLE
        static void initPetsc()
        {
            char ** args;
            int argc = 0;
            // PetscFunctionBeginUser;
            PetscCallVoid(PetscInitialize(&argc, &args, NULL, NULL));
        }

        static void closePetsc()
        {
            PetscCallVoid(PetscFinalize());
        } 

        template<typename TA, typename TB, typename TR>
        static void solveGPU(TA& emat, TB& eb, TR& ex)
        {
            initPetsc();
            if(sizeof(PetscScalar) != 8)
            {
                std::cout << "Petsc does not uses doubles" << std::endl;
            } 

            Mat A; Vec b; Vec x;
            // Set vector x
            PetscCallVoid(VecCreate(PETSC_COMM_SELF, &x));
            PetscCallVoid(VecSetType(x, VECCUDA));
            PetscCallVoid(VecSetSize(x, ex.rows(), ex.rows()));
            PetscCallVoid(VecSetFromOptions(x));

            PetscScalar* xArrayPtr;
            PetscCallVoid(VecGetArray(x, &xArrayPtr));
            PetscCallVoid(PetscArraycpy(xArrayPtr, ex.data(), ex.rows() * sizeof(PetscScalar)));

            // Set vector b
            PetscCallVoid(VecCreate(PETSC_COMM_SELF, &b));
            PetscCallVoid(VecSetType(b, VECCUDA));
            PetscCallVoid(VecSetSize(b, eb.rows(), eb.rows()));
            PetscCallVoid(VecSetFromOptions(b));

            PetscScalar* bArrayPtr;
            PetscCallVoid(VecGetArray(b, &bArrayPtr));
            PetscCallVoid(PetscArraycpy(bArrayPtr, eb.data(), eb.rows() * sizeof(PetscScalar)));
            
            // Set mat A
            PetscCallVoid(MatCreateSeqAIJ(PETSC_COMM_SELF, emat.rows(), emat.cols(), 27, NULL, &A));
            PetscCallVoid(MatSetType(A, MATAIJCUSPARSE));
            PetscCallVoid(MatSetFromOptions(A));
            
            for (int k=0; k<emat.outerSize(); ++k) 
            {
                for (Eigen::SparseMatrix<double>::InnerIterator it(emat, k); it; ++it) 
                {
                    PetscInt row = it.row();    // row index
                    PetscInt col = it.col();    // col index
                    PetscScalar value = it.value();  // the value

                    PetscCallVoid(MatSetValues(A, 1, &row, 1, &col, &value, INSERT_VALUES));
                }
            }

            PetscCallVoid(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
            PetscCallVoid(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

            // Call solver            
            KSP kspSolver;
            PetscCallVoid(KSPCreate(PETSC_COMM_SELF, &kspSolver));
            PetscCallVoid(KSPSetType(kspSolver, KSPCG));
            PetscCallVoid(KSPSetOperators(kspSolver, A, A));
            PetscCallVoid(KSPSetFromOptions(kspSolver));
            PetscCallVoid(KSPSolve(kspSolver, b, x));
            PetscInt numIter;
            KSPGetIterationNumber(kspSolver, &numIter);
            std::cout << "Num Iterations: " << numIter << std::endl;

            KSPDestroy(&kspSolver);

            // Get result


            // Free matrices
            // TODO

            closePetsc();
        }
#endif 
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