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

	
	vec &normal(uint32_t index) { return normals[index]; }
	const vec &normal(uint32_t index) const { return normals[index]; }
	const Eigen::Matrix<double, Dim, Dim>& normalInvCovariance(uint32_t index) const { return normalsInvCov[index]; }
	const std::tuple<Eigen::Matrix<double, Dim, Dim>, Eigen::Vector<double, Dim>>& normalInvCovarianceDes(uint32_t index) const { return normalsInvCovDes[index]; }
	void computeNormals(float stdByDistance = 0, bool useCovariances=false);


	template <class BBOX>
	bool kdtree_get_bbox(BBOX &bb) const { return false; } // Required by nanoflann
private:
	std::vector<vec> points, normals;
	std::vector<float> pointsVar;
	std::vector<Eigen::Matrix<double, Dim, Dim>> normalsInvCov;
	std::vector<std::tuple<Eigen::Matrix<double, Dim, Dim>, Eigen::Vector<double, Dim>>> normalsInvCovDes;
	std::vector<vec> normalsInvVar;
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
		vec invVar = normalsInvVar[i];

		normalsNoise[0][i] = glm::sqrt(1.0 / invVar.x);
		normalsNoise[1][i] = glm::sqrt(1.0 / invVar.y);
		normalsNoise[2][i] = glm::sqrt(1.0 / invVar.z);
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
	normalsInvVar.resize(xProp.size());
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
void PointCloud<Dim>::computeNormals(float stdByDistance, bool useCovariances)
{
	using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud<Dim>>, PointCloud<Dim>, Dim /* dim */>;
	my_kd_tree_t kdtree(Dim /* dim */, *this, {10 /* max depth */});

	const size_t numNear = 20;
	std::vector<uint32_t> nearIndexCache(numNear);
	std::vector<float> nearDistSqrCache(numNear);
	
	Eigen::Matrix<double, Dim, Dim> nCovMat;
	uint32_t ignoredNormals = 0;
	const double varByDistance = stdByDistance * stdByDistance;
	Eigen::Matrix<double, numNear, numNear> pCovMat; 
	Eigen::Matrix<double, numNear, Dim> P; 
	for(uint32_t i=0; i < size(); i++)
	{
		const vec c = points[i];
		// Get nearest points
		uint32_t numResults = kdtree.knnSearch(reinterpret_cast<const float*>(&c), numNear, nearIndexCache.data(), nearDistSqrCache.data());

		// Compute covariance matrix
		nCovMat = Eigen::Matrix<double, Dim, Dim>::Zero();
		pCovMat = Eigen::Matrix<double, numNear, numNear>::Zero();
		P = Eigen::Matrix<double, numNear, Dim>::Zero();
		const double v1 = variance(i);
		for(uint32_t j=0; j < numResults; j++)
		{
			if(nearIndexCache[j] == i) continue;
			const double v2 = variance(nearIndexCache[j]);
			const vec p = points[nearIndexCache[j]] - c;
			const double distVar = varByDistance * glm::dot(p, p);
			const double cov = v1 * v2 * glm::sqrt(1.0/((distVar + v1) * (distVar + v2)));
			if(useCovariances)
			{
				for(uint32_t d = 0; d < Dim; d++) P(j, d) = p[d];
				pCovMat(j, j) = v1 + v2 + distVar - 2.0 * cov;
			}
			else
			{
				const double invVar = 1.0 / (v1 + v2 + distVar - 2.0 * cov);
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
			for(uint32_t j=0; j < numResults; j++)
			{
				const double vj = pCovMat(j, j);
				for(uint32_t k=0; k < numResults; k++)
				{
					if(j == k) continue;
					const vec p = points[nearIndexCache[j]] - points[nearIndexCache[k]];
					const double distVar = varByDistance * glm::dot(p, p);
					const double vk = pCovMat(k, k);
					const double cov = vj * vk * glm::sqrt(1.0/((distVar + vj) * (distVar + vk)));
					pCovMat(j, k) = cov;
				}
			}
		}

		// Compute inv cov
		if(useCovariances)
		{
			Eigen::BDCSVD<Eigen::MatrixXd> svd(pCovMat, Eigen::ComputeThinU | Eigen::ComputeThinV);
			Eigen::VectorXd sv = svd.singularValues();
			for(uint32_t i=0; i < sv.size(); i++)
			{
				if(glm::abs(sv(i)) > 1e-8) sv(i) = 1.0 / sv(i);
				else sv(i) = 0.0;
			}
			Eigen::Matrix<double, numNear, numNear> pInvCovMat = svd.matrixV() * sv.asDiagonal() * svd.matrixU().adjoint();
			nCovMat = P.transpose() * pInvCovMat * P;
		}

		// Compute best normal
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, Dim, Dim>> eigenSolver;
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
			sumVar += 1.0 / eigenValues((minIndex + d) % Dim);
		}
		if(sumVar > 0.99) // Ignore this normal
		{
			if(ignoredNormals < 50) std::cout << sumVar << std::endl;
			normalsInvCov[i] = Eigen::Matrix<double, Dim, Dim>::Zero();
			ignoredNormals++;
		}
		else
		{
			const double sumVar4 = glm::min(4 * sumVar, 1.0);
			double a = glm::max((1.0 - glm::sqrt(1-sumVar4)) / (sumVar4), 0.503);
			eigenValues(minIndex) = 1.0 / ((2.*a-1.)*sumVar);
			normalsInvCov[i] = eigenVectors * eigenValues.asDiagonal() * eigenVectors.transpose();
			normalsInvVar[i] = vec(eigenValues((minIndex+1)%Dim), eigenValues((minIndex+2)%Dim), eigenValues(minIndex));
			Eigen::Matrix<double, Dim, Dim> oEigenVectors;
			Eigen::Vector<double, Dim> oEigenValues;
			for(uint32_t d=0; d < Dim; d++) {
				oEigenVectors.col(d) = eigenVectors.col((minIndex+d+1)%Dim);
				oEigenValues(d) = eigenValues((minIndex+d+1)%Dim);
			}
			normalsInvCovDes[i] = std::make_tuple(oEigenVectors.transpose(), oEigenValues);
		}
	}
	std::cout << "Num ignored normals " << ignoredNormals << std::endl;
}

#endif


