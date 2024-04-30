#include <iostream>
#include <cmath>
#include "BiharmonicReconstruction.h"
#include "timing.h"
#include "AppParams.h"
#include "LinearSystemSolver.h"


BiharmonicReconstruction::BiharmonicReconstruction()
{
	pW = gW = sW = 1.0f;
}


void BiharmonicReconstruction::setWeights(double pointEqWeight, double gradientEqWeight, double smoothnessEqWeight)
{
	pW = pointEqWeight;
	gW = gradientEqWeight;
	sW = smoothnessEqWeight;
}

void BiharmonicReconstruction::compute(const PointCloud &cloud, ScalarField &field)
{
	unsigned int nEquations, nUnknowns;
	
	if(AppParams::instance()->bLogging)
		cout << "Preparing the system" << endl;
	long lastTime = getTimeMilliseconds();
	
	nEquations = 3 * cloud.size() + (field.width() - 2) * (field.height() - 2);
	nUnknowns = field.width() * field.height();
	
	if(AppParams::instance()->bLogging)
		cout << nEquations << " equations and " << nUnknowns << " unknowns" << endl;

	Eigen::SparseMatrix<double> A(nEquations, nUnknowns), AtA;
	Eigen::VectorXd b(nEquations), Atb, x;
	vector<Eigen::Triplet<double>> triplets;
	unsigned int eqIndex = 0;

	for(unsigned int i=0; i<cloud.size(); i++)
	{
		addPointEquation(eqIndex, cloud.point(i), field, triplets, b);
		eqIndex++;
	}
	for(unsigned int i=0; i<cloud.size(); i++)
	{
		addGradientEquations(eqIndex, cloud.point(i), cloud.normal(i), field, triplets, b);
		eqIndex += 2;
	}
	for(unsigned int j=1; j<(field.height()-1); j++)
		for(unsigned int i=1; i<(field.width()-1); i++)
		{
			addSmoothnessEquation(eqIndex, field, i, j, triplets, b);
			eqIndex++;
		}

	if(AppParams::instance()->bLogging)
		cout << "Total equations added: " << eqIndex << endl;
	
	A.setFromTriplets(triplets.begin(), triplets.end());

	if(AppParams::instance()->bLogging)
		cout << "Building A & b in " << (getTimeMilliseconds() - lastTime) << " ms" << endl;

	LinearSystemSolver solver;
	
	if(AppParams::instance()->bLogging)
		cout << "Solving the system" << endl;
	lastTime = getTimeMilliseconds();

	solver.setMatrix(&A);
	solver.setIndependentVector(&b);
	if(!solver.solve(x))
	{
		if(AppParams::instance()->bLogging)
			cout << "Error: " << solver.log() << endl;
	}
	else
	{
		if(AppParams::instance()->bLogging)
		{
			cout << "The relative error is: " << solver.relativeError() << endl;
			cout << "Error: " << solver.error() << endl;
			cout << "Iterations: " << solver.iterations() << endl;
		}
		for(unsigned int j=0, pos=0; j<field.height(); j++)
			for(unsigned int i=0; i<field.width(); i++, pos++)
				field(i, j) = x(pos);
	}

	if(AppParams::instance()->bLogging)
		cout << "Solved in " << (getTimeMilliseconds() - lastTime) << " ms" << endl;
}

void BiharmonicReconstruction::computeComponentWise(const PointCloud &cloud, ScalarField &field)
{
	unsigned int nEquations, nUnknowns;
	
	cout << "Preparing the system" << endl;
	long lastTime = getTimeMilliseconds();
	
	nEquations = 3 * cloud.size() + 2 * field.width() * field.height() - 2 * field.width() - 2 * field.height();
	nUnknowns = field.width() * field.height();
	
	cout << nEquations << " equations and " << nUnknowns << " unknowns" << endl;

	Eigen::SparseMatrix<double> A(nEquations, nUnknowns), AtA;
	Eigen::VectorXd b(nEquations), Atb, x;
	vector<Eigen::Triplet<double>> triplets;
	unsigned int eqIndex = 0;

	for(unsigned int i=0; i<cloud.size(); i++)
	{
		addPointEquation(eqIndex, cloud.point(i), field, triplets, b);
		eqIndex++;
	}
	for(unsigned int i=0; i<cloud.size(); i++)
	{
		addGradientEquations(eqIndex, cloud.point(i), cloud.normal(i), field, triplets, b);
		eqIndex += 2;
	}
	for(unsigned int j=0; j<field.height(); j++)
		for(unsigned int i=1; i<(field.width()-1); i++)
	{
		addHorizontalBoundarySmoothnessEquation(eqIndex, field, i, j, triplets, b);
		eqIndex++;
	}
	for(unsigned int j=1; j<(field.height()-1); j++)
		for(unsigned int i=0; i<field.width(); i++)
	{
		addVerticalBoundarySmoothnessEquation(eqIndex, field, i, j, triplets, b);
		eqIndex++;
	}
	
	cout << "Total equations added: " << eqIndex << endl;
	
	A.setFromTriplets(triplets.begin(), triplets.end());

	cout << "Building A & b in " << (getTimeMilliseconds() - lastTime) << " ms" << endl;

	LinearSystemSolver solver;
	
	if(AppParams::instance()->bLogging)
		cout << "Solving the system" << endl;
	lastTime = getTimeMilliseconds();

	solver.setMatrix(&A);
	solver.setIndependentVector(&b);
	if(!solver.solve(x))
	{
		if(AppParams::instance()->bLogging)
			cout << "Error: " << solver.log() << endl;
	}
	else
	{
		if(AppParams::instance()->bLogging)
		{
			cout << "The relative error is: " << solver.relativeError() << endl;
			cout << "Error: " << solver.error() << endl;
			cout << "Iterations: " << solver.iterations() << endl;
		}
		for(unsigned int j=0, pos=0; j<field.height(); j++)
			for(unsigned int i=0; i<field.width(); i++, pos++)
				field(i, j) = x(pos);
	}

	if(AppParams::instance()->bLogging)
		cout << "Solved in " << (getTimeMilliseconds() - lastTime) << " ms" << endl;
}

void BiharmonicReconstruction::computeBilaplacian(const PointCloud &cloud, ScalarField &field)
{
	unsigned int nEquations, nUnknowns;
	
	cout << "Preparing the system" << endl;
	long lastTime = getTimeMilliseconds();
	
	nEquations = 3 * cloud.size() + (field.width() - 2) * (field.height() - 2) + (field.width() - 4) * (field.height() - 4);
	nUnknowns = field.width() * field.height();
	
	cout << nEquations << " equations and " << nUnknowns << " unknowns" << endl;

	Eigen::SparseMatrix<double> A(nEquations, nUnknowns), AtA;
	Eigen::VectorXd b(nEquations), Atb, x;
	vector<Eigen::Triplet<double>> triplets;
	unsigned int eqIndex = 0;

	for(unsigned int i=0; i<cloud.size(); i++)
	{
		addPointEquation(eqIndex, cloud.point(i), field, triplets, b);
		eqIndex++;
	}
	for(unsigned int i=0; i<cloud.size(); i++)
	{
		addGradientEquations(eqIndex, cloud.point(i), cloud.normal(i), field, triplets, b);
		eqIndex += 2;
	}
	for(unsigned int j=1; j<(field.height()-1); j++)
		for(unsigned int i=1; i<(field.width()-1); i++)
		{
			addSmoothnessEquation(eqIndex, field, i, j, triplets, b);
			eqIndex++;
		}
	for(unsigned int j=2; j<(field.height()-2); j++)
		for(unsigned int i=2; i<(field.width()-2); i++)
		{
			addBilaplacianSmoothnessEquation(eqIndex, field, i, j, triplets, b);
			eqIndex++;
		}
	
	cout << "Total equations added: " << eqIndex << endl;
	
	A.setFromTriplets(triplets.begin(), triplets.end());

	cout << "Building A & b in " << (getTimeMilliseconds() - lastTime) << " ms" << endl;

	LinearSystemSolver solver;
	
	if(AppParams::instance()->bLogging)
		cout << "Solving the system" << endl;
	lastTime = getTimeMilliseconds();

	solver.setMatrix(&A);
	solver.setIndependentVector(&b);
	if(!solver.solve(x))
	{
		if(AppParams::instance()->bLogging)
			cout << "Error: " << solver.log() << endl;
	}
	else
	{
		if(AppParams::instance()->bLogging)
		{
			cout << "The relative error is: " << solver.relativeError() << endl;
			cout << "Error: " << solver.error() << endl;
			cout << "Iterations: " << solver.iterations() << endl;
		}
		for(unsigned int j=0, pos=0; j<field.height(); j++)
			for(unsigned int i=0; i<field.width(); i++, pos++)
				field(i, j) = x(pos);
	}

	if(AppParams::instance()->bLogging)
		cout << "Solved in " << (getTimeMilliseconds() - lastTime) << " ms" << endl;
}

void BiharmonicReconstruction::computeNoGradient(const PointCloud &cloud, ScalarField &field)
{
	unsigned int nEquations, nUnknowns;
	
	cout << "Preparing the system" << endl;
	long lastTime = getTimeMilliseconds();
	
	nEquations = 3 * cloud.size() + 2 * field.width() * field.height() - 2 * field.width() - 2 * field.height();
	nUnknowns = field.width() * field.height();
	
	cout << nEquations << " equations and " << nUnknowns << " unknowns" << endl;

	Eigen::SparseMatrix<double> A(nEquations, nUnknowns), AtA;
	Eigen::VectorXd b(nEquations), Atb, x;
	vector<Eigen::Triplet<double>> triplets;
	unsigned int eqIndex = 0;

	for(unsigned int i=0; i<cloud.size(); i++)
	{
		addPointEquation(eqIndex, cloud.point(i), field, triplets, b);
		eqIndex++;
		addPointEquation(eqIndex, cloud.point(i) + (1.0f / field.width()) * cloud.normal(i), field, triplets, b, 1.0f);
		eqIndex++;
		addPointEquation(eqIndex, cloud.point(i) - (1.0f / field.height()) * cloud.normal(i), field, triplets, b, -1.0f);
		eqIndex++;
	}
	for(unsigned int j=0; j<field.height(); j++)
		for(unsigned int i=1; i<(field.width()-1); i++)
	{
		addHorizontalBoundarySmoothnessEquation(eqIndex, field, i, j, triplets, b);
		eqIndex++;
	}
	for(unsigned int j=1; j<(field.height()-1); j++)
		for(unsigned int i=0; i<field.width(); i++)
	{
		addVerticalBoundarySmoothnessEquation(eqIndex, field, i, j, triplets, b);
		eqIndex++;
	}
	
	cout << "Total equations added: " << eqIndex << endl;
	
	A.setFromTriplets(triplets.begin(), triplets.end());

	cout << "Building A & b in " << (getTimeMilliseconds() - lastTime) << " ms" << endl;

	LinearSystemSolver solver;
	
	if(AppParams::instance()->bLogging)
		cout << "Solving the system" << endl;
	lastTime = getTimeMilliseconds();

	solver.setMatrix(&A);
	solver.setIndependentVector(&b);
	if(!solver.solve(x))
	{
		if(AppParams::instance()->bLogging)
			cout << "Error: " << solver.log() << endl;
	}
	else
	{
		if(AppParams::instance()->bLogging)
		{
			cout << "The relative error is: " << solver.relativeError() << endl;
			cout << "Error: " << solver.error() << endl;
			cout << "Iterations: " << solver.iterations() << endl;
		}
		for(unsigned int j=0, pos=0; j<field.height(); j++)
			for(unsigned int i=0; i<field.width(); i++, pos++)
				field(i, j) = x(pos);
	}

	if(AppParams::instance()->bLogging)
		cout << "Solved in " << (getTimeMilliseconds() - lastTime) << " ms" << endl;
}

void BiharmonicReconstruction::addPointEquation(unsigned int eqIndex, const glm::vec2 &P, const ScalarField &field, vector<Eigen::Triplet<double>> &triplets, Eigen::VectorXd &b, float value)
{
	unsigned int i, j;
	float x, y;
	
	i = (unsigned int)floor(P.x * (field.width() - 1));
	i = glm::max(0u, glm::min(i, field.width()-2));
	j = (unsigned int)floor(P.y * (field.height() - 1));
	j = glm::max(0u, glm::min(j, field.height()-2));
	
	x = P.x * (field.width() - 1) - float(i);
	y = P.y * (field.height() - 1) - float(j);
	
	triplets.push_back(Eigen::Triplet<double>(eqIndex, unknownIndex(field, i+1, j+1), pW*x*y));
	triplets.push_back(Eigen::Triplet<double>(eqIndex, unknownIndex(field, i, j+1), pW*(1.0f-x)*y));
	triplets.push_back(Eigen::Triplet<double>(eqIndex, unknownIndex(field, i+1, j), pW*x*(1.0f-y)));
	triplets.push_back(Eigen::Triplet<double>(eqIndex, unknownIndex(field, i, j), pW*(1.0f-x)*(1.0f-y)));
	
	b(eqIndex) = value;
}

void BiharmonicReconstruction::addGradientEquations(unsigned int eqIndex, const glm::vec2 &P, const glm::vec2 &N, const ScalarField &field, vector<Eigen::Triplet<double>> &triplets, Eigen::VectorXd &b)
{
	unsigned int i, j;
	float x, y;
	
	i = (unsigned int)floor(P.x * (field.width() - 1));
	i = glm::max(0u, glm::min(i, field.width()-2));
	j = (unsigned int)floor(P.y * (field.height() - 1));
	j = glm::max(0u, glm::min(j, field.height()-2));
	
	x = P.x * (field.width() - 1) - float(i);
	y = P.y * (field.height() - 1) - float(j);
	
	triplets.push_back(Eigen::Triplet<double>(eqIndex, unknownIndex(field, i+1, j+1), gW*y));
	triplets.push_back(Eigen::Triplet<double>(eqIndex, unknownIndex(field, i, j+1), -gW*y));
	triplets.push_back(Eigen::Triplet<double>(eqIndex, unknownIndex(field, i+1, j), gW*(1.0f-y)));
	triplets.push_back(Eigen::Triplet<double>(eqIndex, unknownIndex(field, i, j), -gW*(1.0f-y)));
	
	b(eqIndex) = gW*N.x;

	triplets.push_back(Eigen::Triplet<double>(eqIndex+1, unknownIndex(field, i+1, j+1), gW*x));
	triplets.push_back(Eigen::Triplet<double>(eqIndex+1, unknownIndex(field, i, j+1), gW*(1.0f-x)));
	triplets.push_back(Eigen::Triplet<double>(eqIndex+1, unknownIndex(field, i+1, j), -gW*x));
	triplets.push_back(Eigen::Triplet<double>(eqIndex+1, unknownIndex(field, i, j), -gW*(1.0f-x)));
	
	b(eqIndex+1) = gW*N.y;
}

void BiharmonicReconstruction::addSmoothnessEquation(unsigned int eqIndex, const ScalarField &field, unsigned int i, unsigned int j, vector<Eigen::Triplet<double>> &triplets, Eigen::VectorXd &b)
{
	triplets.push_back(Eigen::Triplet<double>(eqIndex, unknownIndex(field, i, j), -sW*4.0f));

	triplets.push_back(Eigen::Triplet<double>(eqIndex, unknownIndex(field, i+1, j), sW*1.0f));
	triplets.push_back(Eigen::Triplet<double>(eqIndex, unknownIndex(field, i-1, j), sW*1.0f));
	triplets.push_back(Eigen::Triplet<double>(eqIndex, unknownIndex(field, i, j+1), sW*1.0f));
	triplets.push_back(Eigen::Triplet<double>(eqIndex, unknownIndex(field, i, j-1), sW*1.0f));
	
	b(eqIndex) = 0.0f;
}

void BiharmonicReconstruction::addBilaplacianSmoothnessEquation(unsigned int eqIndex, const ScalarField &field, unsigned int i, unsigned int j, vector<Eigen::Triplet<double>> &triplets, Eigen::VectorXd &b)
{
	triplets.push_back(Eigen::Triplet<double>(eqIndex, unknownIndex(field, i, j), sW*20.0f/16.0f));

	triplets.push_back(Eigen::Triplet<double>(eqIndex, unknownIndex(field, i+1, j), -sW*8.0f/16.0f));
	triplets.push_back(Eigen::Triplet<double>(eqIndex, unknownIndex(field, i-1, j), -sW*8.0f/16.0f));
	triplets.push_back(Eigen::Triplet<double>(eqIndex, unknownIndex(field, i, j+1), -sW*8.0f/16.0f));
	triplets.push_back(Eigen::Triplet<double>(eqIndex, unknownIndex(field, i, j-1), -sW*8.0f/16.0f));

	triplets.push_back(Eigen::Triplet<double>(eqIndex, unknownIndex(field, i+1, j+1), sW*2.0f/16.0f));
	triplets.push_back(Eigen::Triplet<double>(eqIndex, unknownIndex(field, i+1, j-1), sW*2.0f/16.0f));
	triplets.push_back(Eigen::Triplet<double>(eqIndex, unknownIndex(field, i-1, j+1), sW*2.0f/16.0f));
	triplets.push_back(Eigen::Triplet<double>(eqIndex, unknownIndex(field, i-1, j-1), sW*2.0f/16.0f));

	triplets.push_back(Eigen::Triplet<double>(eqIndex, unknownIndex(field, i+2, j), sW*1.0f/16.0f));
	triplets.push_back(Eigen::Triplet<double>(eqIndex, unknownIndex(field, i-2, j), sW*1.0f/16.0f));
	triplets.push_back(Eigen::Triplet<double>(eqIndex, unknownIndex(field, i, j+2), sW*1.0f/16.0f));
	triplets.push_back(Eigen::Triplet<double>(eqIndex, unknownIndex(field, i, j-2), sW*1.0f/16.0f));
	
	b(eqIndex) = 0.0f;
}

void BiharmonicReconstruction::addHorizontalBoundarySmoothnessEquation(unsigned int eqIndex, const ScalarField &field, unsigned int i, unsigned int j, vector<Eigen::Triplet<double>> &triplets, Eigen::VectorXd &b)
{
	triplets.push_back(Eigen::Triplet<double>(eqIndex, unknownIndex(field, i, j), -sW*2.0f));

	triplets.push_back(Eigen::Triplet<double>(eqIndex, unknownIndex(field, i+1, j), sW*1.0f));
	triplets.push_back(Eigen::Triplet<double>(eqIndex, unknownIndex(field, i-1, j), sW*1.0f));
	
	b(eqIndex) = 0.0f;
}

void BiharmonicReconstruction::addVerticalBoundarySmoothnessEquation(unsigned int eqIndex, const ScalarField &field, unsigned int i, unsigned int j, vector<Eigen::Triplet<double>> &triplets, Eigen::VectorXd &b)
{
	triplets.push_back(Eigen::Triplet<double>(eqIndex, unknownIndex(field, i, j), -sW*2.0f));

	triplets.push_back(Eigen::Triplet<double>(eqIndex, unknownIndex(field, i, j+1), sW*1.0f));
	triplets.push_back(Eigen::Triplet<double>(eqIndex, unknownIndex(field, i, j-1), sW*1.0f));
	
	b(eqIndex) = 0.0f;
}

unsigned int BiharmonicReconstruction::unknownIndex(const ScalarField &field, unsigned int i, unsigned int j) const
{
	return j * field.width() + i;
}


void BiharmonicReconstruction::computeGD(const PointCloud &cloud, ScalarField &field)
{
	unsigned int nEquations, nUnknowns;
	
	cout << "Preparing the system" << endl;
	long lastTime = getTimeMilliseconds();
	
	nEquations = 3 * cloud.size() + 2 * field.width() * field.height() - 2 * field.width() - 2 * field.height();
	nUnknowns = field.width() * field.height();
	
	cout << nEquations << " equations and " << nUnknowns << " unknowns" << endl;

	Eigen::SparseMatrix<double> A(nEquations, nUnknowns), AtA;
	Eigen::VectorXd b(nEquations), Atb, x(nUnknowns);
	vector<Eigen::Triplet<double>> triplets;
	unsigned int eqIndex = 0;

	for(unsigned int i=0; i<cloud.size(); i++)
	{
		addPointEquation(eqIndex, cloud.point(i), field, triplets, b);
		eqIndex++;
	}
	for(unsigned int i=0; i<cloud.size(); i++)
	{
		addGradientEquations(eqIndex, cloud.point(i), cloud.normal(i), field, triplets, b);
		eqIndex += 2;
	}
	for(unsigned int j=0; j<field.height(); j++)
		for(unsigned int i=1; i<(field.width()-1); i++)
	{
		addHorizontalBoundarySmoothnessEquation(eqIndex, field, i, j, triplets, b);
		eqIndex++;
	}
	for(unsigned int j=1; j<(field.height()-1); j++)
		for(unsigned int i=0; i<field.width(); i++)
	{
		addVerticalBoundarySmoothnessEquation(eqIndex, field, i, j, triplets, b);
		eqIndex++;
	}
	
	cout << "Total equations added: " << eqIndex << endl;
	
	A.setFromTriplets(triplets.begin(), triplets.end());

	cout << "Building A & b in " << (getTimeMilliseconds() - lastTime) << " ms" << endl;

	cout << "Computing normal equation" << endl;
	lastTime = getTimeMilliseconds();
	
	AtA = A.transpose() * A;
	Atb = A.transpose() * b;
	
	cout << "Computed AtA & Atb in " << (getTimeMilliseconds() - lastTime) << " ms" << endl;

	cout << "Solving least squares" << endl;
	lastTime = getTimeMilliseconds();

	Eigen::VectorXd r, AtAr;
	float phi;
	
	// Method: Gradient descent (Wikipedia)
	r = Atb - AtA * x;
	for(unsigned int i=0; i<100000; i++)
	{
		AtAr = AtA * r;
		phi = (r.dot(r)) / (r.dot(AtAr));
		x = x + phi * r;
		r = r - phi * AtAr;
	}
	
	cout << "Least squares solved in " << (getTimeMilliseconds() - lastTime) << " ms" << endl;
	
	double relative_error = (A*x - b).norm() / b.norm();
	cout << "The relative error is: " << relative_error << endl;
		
	for(unsigned int j=0, pos=0; j<field.height(); j++)
		for(unsigned int i=0; i<field.width(); i++, pos++)
			field(i, j) = x(pos);
}

void BiharmonicReconstruction::computeNoGradientGD(const PointCloud &cloud, ScalarField &field)
{
	unsigned int nEquations, nUnknowns;
	
	cout << "Preparing the system" << endl;
	long lastTime = getTimeMilliseconds();
	
	nEquations = 3 * cloud.size() + 2 * field.width() * field.height() - 2 * field.width() - 2 * field.height();
	nUnknowns = field.width() * field.height();
	
	cout << nEquations << " equations and " << nUnknowns << " unknowns" << endl;

	Eigen::SparseMatrix<double> A(nEquations, nUnknowns), AtA;
	Eigen::VectorXd b(nEquations), Atb, x(nUnknowns);
	vector<Eigen::Triplet<double>> triplets;
	unsigned int eqIndex = 0;

	for(unsigned int i=0; i<cloud.size(); i++)
	{
		addPointEquation(eqIndex, cloud.point(i), field, triplets, b);
		eqIndex++;
		addPointEquation(eqIndex, cloud.point(i) + (1.0f / field.width()) * cloud.normal(i), field, triplets, b, 1.0f);
		eqIndex++;
		addPointEquation(eqIndex, cloud.point(i) - (1.0f / field.height()) * cloud.normal(i), field, triplets, b, -1.0f);
		eqIndex++;
	}
	for(unsigned int j=0; j<field.height(); j++)
		for(unsigned int i=1; i<(field.width()-1); i++)
	{
		addHorizontalBoundarySmoothnessEquation(eqIndex, field, i, j, triplets, b);
		eqIndex++;
	}
	for(unsigned int j=1; j<(field.height()-1); j++)
		for(unsigned int i=0; i<field.width(); i++)
	{
		addVerticalBoundarySmoothnessEquation(eqIndex, field, i, j, triplets, b);
		eqIndex++;
	}
	
	cout << "Total equations added: " << eqIndex << endl;
	
	A.setFromTriplets(triplets.begin(), triplets.end());

	cout << "Building A & b in " << (getTimeMilliseconds() - lastTime) << " ms" << endl;

	cout << "Computing normal equation" << endl;
	lastTime = getTimeMilliseconds();
	
	AtA = A.transpose() * A;
	Atb = A.transpose() * b;
	
	Eigen::VectorXd r, AtAr;
	float phi;
	
	cout << "Computed AtA & Atb in " << (getTimeMilliseconds() - lastTime) << " ms" << endl;

	cout << "Solving least squares" << endl;
	lastTime = getTimeMilliseconds();

	// Method: Gradient descent (Wikipedia)
	r = Atb - AtA * x;
	for(unsigned int i=0; i<100000; i++)
	{
		AtAr = AtA * r;
		phi = (r.dot(r)) / (r.dot(AtAr));
		x = x + phi * r;
		r = r - phi * AtAr;
	}
	
	cout << "Least squares solved in " << (getTimeMilliseconds() - lastTime) << " ms" << endl;
	
	double relative_error = (A*x - b).norm() / b.norm();
	cout << "The relative error is: " << relative_error << endl;
		
	for(unsigned int j=0, pos=0; j<field.height(); j++)
		for(unsigned int i=0; i<field.width(); i++, pos++)
			field(i, j) = x(pos);
}








