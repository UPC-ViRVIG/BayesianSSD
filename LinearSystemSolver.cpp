#include "LinearSystemSolver.h"
#include <Eigen/IterativeLinearSolvers>


LinearSystemSolver::LinearSystemSolver()
{
	A = NULL;
	b = NULL;
}


void LinearSystemSolver::setMatrix(Eigen::SparseMatrix<double> *matrix)
{
	A = matrix;
}

void LinearSystemSolver::setIndependentVector(Eigen::VectorXd *independentVector)
{
	b = independentVector;
}

bool LinearSystemSolver::solve(Eigen::VectorXd &x, bool bUseGuess)
{
	if(A == NULL || b == NULL)
	{
		errorMessage = "System parameters not initialized";
		return false;
	}
	if(A->rows() == A->cols())
	{
		Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> cg;
		cg.setTolerance(1e-5);
		cg.compute(*A);
		Eigen::ComputationInfo info = cg.info();
		if(info!=Eigen::Success)
		{
			errorMessage = "Decomposition failed";
			return false;
		}
		if(bUseGuess)
			x = cg.solveWithGuess(*b, x);
		else
			x = cg.solve(*b);
		info = cg.info();
		if(info!=Eigen::Success)
		{
			errorMessage = "Solving failed";
			return false;
		}
		numIterations = cg.iterations();
	}
	else
	{
		Eigen::SparseMatrix<double> AtA;
		Eigen::VectorXd Atb;
	
		AtA = A->transpose() * (*A);
		Atb = A->transpose() * (*b);
		
		Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> cg;
		cg.setTolerance(1e-5);
		cg.compute(AtA);
		Eigen::ComputationInfo info = cg.info();
		if(info!=Eigen::Success)
		{
			errorMessage = "Decomposition failed";
			return false;
		}
		if(bUseGuess)
			x = cg.solveWithGuess(Atb, x);
		else
			x = cg.solve(Atb);
		info = cg.info();
		if(info!=Eigen::Success)
		{
			errorMessage = "Solving failed";
			return false;
		}
		numIterations = cg.iterations();
	}
	errorValue = ((*A)*x - (*b)).norm();
	relativeErrorValue = ((*A)*x - (*b)).norm() / (*b).norm();
	errorMessage = "";
	
	return true;
}

string LinearSystemSolver::log() const
{
	return errorMessage;
}

double LinearSystemSolver::error()
{
	if(A == NULL || b == NULL)
	{
		errorMessage.assign("System parameters not initialized");
		return 0.0;
	}

	return errorValue;
}

double LinearSystemSolver::relativeError()
{
	if(A == NULL || b == NULL)
	{
		errorMessage = "System parameters not initialized";
		return 0.0;
	}

	return relativeErrorValue;
}

int LinearSystemSolver::iterations()
{
	if(A == NULL || b == NULL)
	{
		errorMessage = "System parameters not initialized";
		return 0.0;
	}

	return numIterations;
}





