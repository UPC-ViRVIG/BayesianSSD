#ifndef _BIHARMONIC_RECONSTRUCTION_INCLUDE
#define _BIHARMONIC_RECONSTRUCTION_INCLUDE


#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include "PointCloud.h"
#include "ScalarField.h"


class BiharmonicReconstruction
{
public:
	BiharmonicReconstruction();

	void setWeights(double pointEqWeight, double gradientEqWeight, double smoothnessEqWeight);
	void compute(const PointCloud &cloud, ScalarField &field);
	void computeComponentWise(const PointCloud &cloud, ScalarField &field);
	void computeBilaplacian(const PointCloud &cloud, ScalarField &field);
	void computeNoGradient(const PointCloud &cloud, ScalarField &field);

	void computeGD(const PointCloud &cloud, ScalarField &field);
	void computeNoGradientGD(const PointCloud &cloud, ScalarField &field);

private:
	void addPointEquation(unsigned int eqIndex, const glm::vec2 &P, const ScalarField &field, vector<Eigen::Triplet<double>> &triplets, Eigen::VectorXd &b, float value = 0.0f);
	void addGradientEquations(unsigned int eqIndex, const glm::vec2 &P, const glm::vec2 &N, const ScalarField &field, vector<Eigen::Triplet<double>> &triplets, Eigen::VectorXd &b);
	void addSmoothnessEquation(unsigned int eqIndex, const ScalarField &field, unsigned int i, unsigned int j, vector<Eigen::Triplet<double>> &triplets, Eigen::VectorXd &b);
	void addBilaplacianSmoothnessEquation(unsigned int eqIndex, const ScalarField &field, unsigned int i, unsigned int j, vector<Eigen::Triplet<double>> &triplets, Eigen::VectorXd &b);

	void addHorizontalBoundarySmoothnessEquation(unsigned int eqIndex, const ScalarField &field, unsigned int i, unsigned int j, vector<Eigen::Triplet<double>> &triplets, Eigen::VectorXd &b);
	void addVerticalBoundarySmoothnessEquation(unsigned int eqIndex, const ScalarField &field, unsigned int i, unsigned int j, vector<Eigen::Triplet<double>> &triplets, Eigen::VectorXd &b);
	
	unsigned int unknownIndex(const ScalarField &field, unsigned int i, unsigned int j) const;

private:
	double pW, gW, sW;

};


#endif 


