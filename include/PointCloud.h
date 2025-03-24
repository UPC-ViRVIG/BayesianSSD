#ifndef _POINT_CLOUD_INCLUDE
#define _POINT_CLOUD_INCLUDE


#include <vector>
#include <string>
#include <fstream>
#include <happly.h>
#include "Vector.h"
#include <nanoflann.hpp>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/IterativeLinearSolvers>
#include <glm/gtc/quaternion.hpp>

template<uint32_t Dim>
class PointCloud
{
public:
	using vec = VStruct<Dim>::type;
	bool readFromFile(const std::string &filename, bool readVariance = false) { return false; }
	bool writeToFile(const std::string& filename) { return false; }
	
	uint32_t size() const { return points.size(); }
	size_t kdtree_get_point_count() const { return points.size(); } // Required by nanoflann

	// Point properties
	vec &point(uint32_t index) { return points[index]; }
	const vec &point(uint32_t index) const { return points[index]; }
	float kdtree_get_pt(const size_t idx, const size_t dim) const { return points[idx][dim]; } // Required by nanoflann
	const std::vector<vec>& getPoints() const { return points; }
	std::vector<vec>& getPoints() { return points; }
	const float variance(uint32_t index) const { return pointsVar[index]; }
	const std::vector<float>& getVariances() const { return pointsVar; }
	std::vector<float>& getVariances() { return pointsVar; }

	
	vec &normal(uint32_t index) { return normals[index]; }
	const vec &normal(uint32_t index) const { return normals[index]; }
	const Eigen::Matrix<double, Dim, Dim>& normalInvCovariance(uint32_t index) const { return normalsInvCov[index]; }
	const std::tuple<Eigen::Matrix<double, Dim, Dim>, Eigen::Vector<double, Dim>>& normalInvCovarianceDes(uint32_t index) const { return normalsInvCovDes[index]; }
	void computeNormals(uint32_t numNear, double stdByDistance = 0, double pointsCorrelation=0.0);
	void fillNormalsData(double normalStd);


	template <class BBOX>
	bool kdtree_get_bbox(BBOX &bb) const { return false; } // Required by nanoflann
private:
	std::vector<vec> points, normals;
	std::vector<float> pointsVar;
	std::vector<Eigen::Matrix<double, Dim, Dim>> normalsInvCov;
	std::vector<std::tuple<Eigen::Matrix<double, Dim, Dim>, Eigen::Vector<double, Dim>>> normalsInvCovDes;
};

template<>
bool PointCloud<2>::readFromFile(const std::string &filename, bool readVariance)
{
	std::ifstream fin;
	unsigned int n;
	vec v;
	
	fin.open(filename.c_str());
	if(!fin.is_open())
		return false;
	fin >> n;
	points.resize(n);
	normals.resize(n);
	pointsVar.resize(n, 1.0f);
	normalsInvCov.resize(n);
	normalsInvCovDes.resize(n);
	for(unsigned int i=0; i<n; i++)
	{
		fin >> v.x >> v.y;
		points[i] = v;
		fin >> v.x >> v.y;
		v = glm::normalize(v);
		normals[i] = v;
		if(readVariance)
		{
			float val; fin >> val;
			pointsVar[i] = val;
		}
	}
	fin.close();	
	
	return true;
}

template<>
bool PointCloud<2>::writeToFile(const std::string& filename) 
{
	std::ofstream fout;
	fout.open(filename, std::ofstream::trunc);
	if(!fout.is_open()) return false;
	fout << size() << std::endl;
	for(uint32_t i = 0; i < size(); i++)
	{
		fout << point(i)[0] << " " << point(i)[1] << " ";
		fout << normal(i)[0] << " " << normal(i)[1] << " ";
		fout << variance(i) << " ";
		auto& nvar = std::get<1>(normalsInvCovDes[i]);
		fout << 1.0/nvar(0) << " " << 1.0/nvar(1) << std::endl;
	}
	fout.close();
}

template<>
bool PointCloud<3>::writeToFile(const std::string& filename) 
{
	constexpr uint32_t Dim = 3;
	happly::PLYData plyOut;

	// Write pos
	std::vector<float> tmp(size());
	plyOut.addElement("vertex", size());
	for(uint32_t i=0; i < size(); i++) tmp[i] = points[i][0];
	plyOut.getElement("vertex").addProperty("x", tmp);
	for(uint32_t i=0; i < size(); i++) tmp[i] = points[i][1];
	plyOut.getElement("vertex").addProperty("y", tmp);
	for(uint32_t i=0; i < size(); i++) tmp[i] = points[i][2];
	plyOut.getElement("vertex").addProperty("z", tmp);

	for(uint32_t i=0; i < size(); i++) tmp[i] = normals[i][0];
	plyOut.getElement("vertex").addProperty("nx", tmp);
	for(uint32_t i=0; i < size(); i++) tmp[i] = normals[i][1];
	plyOut.getElement("vertex").addProperty("ny", tmp);
	for(uint32_t i=0; i < size(); i++) tmp[i] = normals[i][2];
	plyOut.getElement("vertex").addProperty("nz", tmp);

	// Write points std
	std::vector<float> noiseStd(size());
	std::transform(pointsVar.begin(), pointsVar.end(), noiseStd.begin(), [](float a) { return glm::sqrt(a); });
	plyOut.getElement("vertex").addProperty("noise_std", noiseStd);

	// Write normals std
	std::array<std::vector<float>, 3> normalsNoise;
	normalsNoise.fill(std::vector<float>(size()));
	for(uint32_t i=0; i < size(); i++)
	{
		auto invVar = std::get<1>(normalsInvCovDes[i]);
		normalsNoise[0][i] = glm::sqrt(1.0 / invVar(0));
		normalsNoise[1][i] = glm::sqrt(1.0 / invVar(1));
		normalsNoise[2][i] = glm::sqrt(1.0 / invVar(2));
	}
	plyOut.getElement("vertex").addProperty("normal_noise_x", normalsNoise[0]);
	plyOut.getElement("vertex").addProperty("normal_noise_y", normalsNoise[1]);
	plyOut.getElement("vertex").addProperty("normal_noise_z", normalsNoise[2]);
	
	plyOut.write(filename);

	return true;
}

template<>
bool PointCloud<3>::readFromFile(const std::string &filename, bool readVariance)
{
	happly::PLYData plyIn(filename);

	// Read pos
	std::vector<float> xProp = plyIn.getElement("vertex").getProperty<float>("x");
    std::vector<float> yProp = plyIn.getElement("vertex").getProperty<float>("y");
    std::vector<float> zProp = plyIn.getElement("vertex").getProperty<float>("z");

	// Read normal
	bool hasNormals = plyIn.getElement("vertex").hasProperty("nx") | 
					  plyIn.getElement("vertex").hasProperty("ny") | 
					  plyIn.getElement("vertex").hasProperty("nz");
					  
	std::vector<float> nxProp = hasNormals ? plyIn.getElement("vertex").getProperty<float>("nx") : std::vector<float>();
    std::vector<float> nyProp = hasNormals ? plyIn.getElement("vertex").getProperty<float>("ny") : std::vector<float>();
    std::vector<float> nzProp = hasNormals ? plyIn.getElement("vertex").getProperty<float>("nz") : std::vector<float>();

	// Read point variance
	if(plyIn.getElement("vertex").hasProperty("noise_std"))
	{
		pointsVar = plyIn.getElement("vertex").getProperty<float>("noise_std");
	}
	else
	{
		std::cout << "Point cloud does not have variance" << std::endl;
		pointsVar.resize(xProp.size(), 1.0f);
	}

	points.resize(xProp.size());
	normals.resize(xProp.size());
	normalsInvCov.resize(xProp.size());
	normalsInvCovDes.resize(xProp.size());

	for(uint32_t i=0; i < xProp.size(); i++)
	{
		points[i] = glm::vec3(xProp[i], yProp[i], zProp[i]);
		if(hasNormals) normals[i] = glm::normalize(glm::vec3(nxProp[i], nyProp[i], nzProp[i]));
		pointsVar[i] = pointsVar[i] * pointsVar[i]; // Pass std to var
	}
	
	return true;
}

template<uint32_t Dim>
void PointCloud<Dim>::fillNormalsData(double normalStd) { }

template<>
void PointCloud<2>::fillNormalsData(double normalStd)
{
	const uint32_t Dim = 2;
	for(uint32_t i=0; i < size(); i++)
	{
		const vec n = normal(i);
		vec nTan = vec(-n.y, n.x);
		Eigen::Matrix<double, Dim, Dim> R;
		Eigen::Vector<double, Dim> invVar;
		R(0, 0) = nTan.x; R(0, 1) = nTan.y;
		R(1, 0) = n.x; R(1, 1) = n.y;
		invVar(0) = 1.0 / normalStd*normalStd;
		double sumVar = normalStd*normalStd;

		auto cdf = [](float x, float mu, float std)
		{
			return 0.5f * (1.0f + glm::abs(erf((x - mu) / (glm::sqrt(2.0f) * std))));
		};

		double sumStd = glm::sqrt(sumVar);
		double lastCdfValue = 0.5;
		double varz = 0.0;
		for(uint32_t j=0; j < 15; j++)
		{
			double s = 9.0 * j/15.0;
			s = s * s * sumStd;
			double f = 9.0 * (j+1.0)/15.0;
			f = f * f * sumStd;
			const double x = 0.5 * s + 0.5 * f;
			double vcdf = cdf(f, 0.0, sumStd);
			const double v = 1.0 - glm::sqrt(glm::max(1 - x*x, 0.0));
			varz += 2.0 * (vcdf - lastCdfValue) * v * v;
			lastCdfValue = vcdf;
		}
		invVar(1) = 1.0 / varz;
		normalsInvCov[i] = R.transpose() * invVar.asDiagonal() * R;
		normalsInvCovDes[i] = std::make_tuple(R, invVar);
	}
}

template<uint32_t Dim>
void PointCloud<Dim>::computeNormals(uint32_t numNear, double varByDistancePer, double pointsCorrelation)
{
	bool useCovariances = pointsCorrelation > 0.0;
	using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud<Dim>>, PointCloud<Dim>, Dim /* dim */>;
	my_kd_tree_t kdtree(Dim /* dim */, *this, {10 /* max depth */});

	// const size_t numNear = (Dim == 2) ? 17 : 30;
	// const size_t numNear = (Dim == 2) ? 3 : 20;
	std::vector<uint32_t> nearIndexCache(numNear);
	std::vector<float> nearDistSqrCache(numNear);

	double meanNeighbourDistance = 0.0;
	for(uint32_t i=0; i < size(); i++)
	{
		// Get nearest points
		uint32_t numResults = kdtree.knnSearch(reinterpret_cast<const float*>(&points[i]), numNear, nearIndexCache.data(), nearDistSqrCache.data());
		vec c = vec(0.0f);
		const float invNumResults = 1.0f / static_cast<float>(numResults);
		for(uint32_t j=0; j < numResults; j++)
		{
			for(uint32_t d=0; d < Dim; d++) 
				c[d] += invNumResults * points[nearIndexCache[j]][d];
		}
		
		double meanDistance = 0.0;
		for(uint32_t j=0; j < numResults; j++)
		{
			meanDistance += glm::length(points[nearIndexCache[j]] - c) * invNumResults;
		}
		meanNeighbourDistance += meanDistance / static_cast<double>(size());
	}

	auto cdf = [](float x, float mu, float std)
	{
		return 0.5f * (1.0f + glm::abs(erf((x - mu) / (glm::sqrt(2.0f) * std))));
	};
	
	Eigen::LDLT<Eigen::MatrixXd> solver;
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, Dim, Dim>> eigenSolver;

	Eigen::Matrix<double, Dim, Dim> nCovMat;
	uint32_t ignoredNormals = 0;
	Eigen::MatrixXd pCovMat(numNear, numNear); 
	Eigen::MatrixXd P(numNear, Dim); 
	Eigen::MatrixXd invCovP(numNear, Dim);
	Eigen::VectorXd minDistPair(numNear);
	#pragma omp parallel for firstprivate(nCovMat, pCovMat, P, invCovP, minDistPair, nearIndexCache, nearDistSqrCache) private(solver, eigenSolver)
	for(uint32_t i=0; i < size(); i++)
	{
		// Get nearest points
		uint32_t numResults = kdtree.knnSearch(reinterpret_cast<const float*>(&points[i]), numNear, nearIndexCache.data(), nearDistSqrCache.data());

		// Compute mean distance
		vec c = vec(0.0f);
		double meanDistance = 0.0;
		const float invNumResults = 1.0f / static_cast<float>(numResults);
		for(uint32_t j=0; j < numResults; j++)
		{
			meanDistance += glm::length(points[nearIndexCache[j]] - points[i]) * invNumResults;
			for(uint32_t d=0; d < Dim; d++) 
				c[d] += invNumResults * points[nearIndexCache[j]][d];
		}

		const double meanSqDistance = meanNeighbourDistance * meanNeighbourDistance;
		
		pCovMat = Eigen::MatrixXd::Zero(numNear, numNear);
		double meanMinPairDistance = 0.0;
		if(useCovariances)
		{
			minDistPair = INFINITY * Eigen::VectorXd::Ones(numNear);
			for(uint32_t j=0; j < numResults; j++)
			{
				pCovMat(j, j) = 0.0;
				for(uint32_t k=j+1; k < numResults; k++)
				{
					const vec p = points[nearIndexCache[j]] - points[nearIndexCache[k]];
					pCovMat(j, k) = glm::dot(p, p);
					pCovMat(k, j) = pCovMat(j, k);
					const double dist = static_cast<double>(glm::sqrt(glm::dot(p, p)));
					minDistPair(j) = glm::min(minDistPair(j), dist);
					minDistPair(k) = glm::min(minDistPair(k), dist);
				}
			}
			meanMinPairDistance = minDistPair.sum() / static_cast<double>(numResults-1);
		}

		// Compute covariance matrix
		nCovMat = Eigen::Matrix<double, Dim, Dim>::Zero();
		P = Eigen::MatrixXd::Zero(numNear, Dim);
		const double v1 = variance(i);
		for(uint32_t j=0; j < numResults; j++)
		{
			const double v2 = variance(nearIndexCache[j]);
			const vec p = points[nearIndexCache[j]] - c;
			double sqDist = glm::dot(p, p) / meanSqDistance;
			const double pVar = (1.0 + varByDistancePer * (sqDist-1.0))  * (v1 + v2);
			if(useCovariances)
			{
				for(uint32_t d = 0; d < Dim; d++) P(j, d) = p[d];
				pCovMat(j, j) = pVar;
			}
			else
			{
				const double invVar = 1.0 / pVar;
				for(uint32_t di = 0; di < Dim; di++)
				{
					for(uint32_t dj = 0; dj < Dim; dj++)
					{
						nCovMat(di, dj) += invVar * static_cast<double>(p[di] * p[dj]);
					}
				}
			}
		}

		// Compute covariancies between points
		if(useCovariances)
		{
			// const double sqMeanMinPairDistance = meanMinPairDistance * meanMinPairDistance;
			const double sqMeanMinPairDistance = 2 * meanDistance * meanDistance * invNumResults;
			for(uint32_t j=0; j < numResults; j++)
			{
				const double vj = pCovMat(j, j);
				for(uint32_t k=j+1; k < numResults; k++)
				{
					// const vec p = points[nearIndexCache[j]] - points[nearIndexCache[k]];
					// const double sqDist = glm::dot(p, p);
					const double sqDist = pCovMat(j, k);
					const double vk = pCovMat(k, k);
					const double cov = pointsCorrelation * glm::sqrt(vj * vk) * glm::exp(-sqDist/(2 * sqMeanMinPairDistance));
					pCovMat(j, k) = cov; pCovMat(k, j) = cov;
				}
			}
		}

		// Compute inv cov
		if(useCovariances)
		{
			solver.compute(pCovMat);
			invCovP = solver.solve(P);
			nCovMat = P.transpose() * invCovP;
		}

		// Compute best normal
		eigenSolver.compute(nCovMat, Eigen::DecompositionOptions::ComputeEigenvectors);
		Eigen::Matrix<double, Dim, 1> eigenValues = eigenSolver.eigenvalues();

		uint32_t minIndex = 0;
		double minValue = glm::abs(eigenValues(0));
		for(uint32_t d=1; d < Dim; d++)
		{
			if(glm::abs(eigenValues(d)) < minValue)
			{
				minValue = glm::abs(eigenValues(d)); minIndex = d;
			}
		}

		Eigen::Matrix<double, Dim, Dim> eigenVectors = eigenSolver.eigenvectors();
		auto minVec = eigenVectors.col(minIndex);
		vec newNormal;
		for(uint32_t d=0; d < Dim; d++) newNormal[d] = static_cast<float>(minVec(d));
		newNormal = glm::normalize(newNormal);

		// Flip normals
		if(glm::dot(newNormal, normals[i]) >= 0.0) normals[i] = newNormal;
		else normals[i] = -newNormal;

		// Compute covariance
		double sumVar = 0.0;
		for(uint32_t d=1; d < Dim; d++)
		{
			eigenValues((minIndex + d) % Dim) -= eigenValues(minIndex);
			sumVar += 1.0 / eigenValues((minIndex + d) % Dim);
		}
		

		double sumStd = glm::sqrt(sumVar);
		double lastCdfValue = 0.5;
		double varz = 0.0;
		for(uint32_t j=0; j < 15; j++)
		{
			double s = 9.0 * j/15.0;
			s = s * s * sumStd;
			double f = 9.0 * (j+1.0)/15.0;
			f = f * f * sumStd;
			const double x = 0.5 * s + 0.5 * f;
			double vcdf = cdf(f, 0.0, sumStd);
			const double v = 1.0 - glm::sqrt(glm::max(1 - x*x, 0.0));
			varz += 2.0 * (vcdf - lastCdfValue) * v * v;
			lastCdfValue = vcdf;
		}

		eigenValues(minIndex) = 1.0 / varz;
		normalsInvCov[i] = eigenVectors * eigenValues.asDiagonal() * eigenVectors.transpose();
		Eigen::Matrix<double, Dim, Dim> oEigenVectors;
		Eigen::Vector<double, Dim> oEigenValues;
		for(uint32_t d=0; d < Dim; d++) {
			oEigenVectors.col(d) = eigenVectors.col((minIndex+d+1)%Dim);
			oEigenValues(d) = eigenValues((minIndex+d+1)%Dim);
		}
		normalsInvCovDes[i] = std::make_tuple(oEigenVectors.transpose(), oEigenValues);
		normalsInvCov[i] = oEigenVectors * oEigenValues.asDiagonal() * oEigenVectors.transpose();
		// if(sumVar > 0.99) // Ignore this normal
		// {
		// 	// if(ignoredNormals < 50) std::cout << sumVar << std::endl;
		// 	normalsInvCov[i] = Eigen::Matrix<double, Dim, Dim>::Zero();
		// 	normalsInvCovDes[i] = std::make_tuple(Eigen::Matrix<double, Dim, Dim>::Zero(), Eigen::Vector<double, Dim>::Zero());
		// 	ignoredNormals++;
		// }
		// else
		// sumVar = glm::min(sumVar, 0.99);
		// const double sumVar4 = glm::min(4 * sumVar, 1.0);
		// double a = glm::max((1.0 - glm::sqrt(1-sumVar4)) / (sumVar4), 0.503);
		// eigenValues(minIndex) = 1.0 / ((2.*a-1.)*sumVar);
		// if(sumVar > 0.99)
		// {
		// 	eigenValues(minIndex) = 1.0 / sumVar;
		// }
	}
	std::cout << "Num ignored normals " << ignoredNormals << std::endl;
}

#endif


