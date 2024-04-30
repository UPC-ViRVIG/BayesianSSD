#ifndef _LINEAR_SYSTEM_SOLVER_INCLUDE
#define _LINEAR_SYSTEM_SOLVER_INCLUDE


#include <Eigen/Sparse>
#include <string>


using namespace std;


class LinearSystemSolver
{

public:
	LinearSystemSolver();

	void setMatrix(Eigen::SparseMatrix<double> *matrix);
	void setIndependentVector(Eigen::VectorXd *independentVector);
	
	bool solve(Eigen::VectorXd &x, bool bUseGuess=false);

	string log() const;
	double error();
	double relativeError();
	int iterations();
	
private:
	Eigen::SparseMatrix<double> *A;
	Eigen::VectorXd *b;
	string errorMessage;
	double errorValue, relativeErrorValue;
	int numIterations;

};


#endif



