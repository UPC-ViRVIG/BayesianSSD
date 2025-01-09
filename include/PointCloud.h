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
	const vec& normalVarianceVec(uint32_t index) const { return normalsVar[index]; }
	void computeNormals();


	template <class BBOX>
	bool kdtree_get_bbox(BBOX &bb) const { return false; } // Required by nanoflann
private:
	std::vector<vec> points, normals;
	std::vector<float> pointsVar;
	std::vector<vec> normalsVar;
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
	normalsVar.resize(n);
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

	points.resize(xProp.size());
	normals.resize(xProp.size());
	pointsVar.resize(xProp.size(), 1.0f);
	normalsVar.resize(xProp.size());

	for(uint32_t i=0; i < xProp.size(); i++)
	{
		points[i] = glm::vec3(xProp[i], yProp[i], zProp[i]);
		if(hasNormals) normals[i] = glm::vec3(nxProp[i], nyProp[i], nzProp[i]);
	}
	
	return true;
}

template<uint32_t Dim>
void PointCloud<Dim>::computeNormals()
{
	using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud<Dim>>, PointCloud<Dim>, Dim /* dim */>;
	my_kd_tree_t kdtree(Dim /* dim */, *this, {10 /* max depth */});

	const size_t numNear = 5;
	std::vector<uint32_t> nearIndexCache(numNear);
	std::vector<float> nearDistSqrCache(numNear);
	
	Eigen::Matrix<float, Dim, Dim> nCovMat;
	for(uint32_t i=0; i < size(); i++)
	{
		const vec c = points[i];
		// Get nearest points
		uint32_t numResults = kdtree.knnSearch(reinterpret_cast<const float*>(&c), numNear, nearIndexCache.data(), nearDistSqrCache.data());

		// Compute covariance matrix
		nCovMat = Eigen::Matrix<float, Dim, Dim>::Zero();
		for(uint32_t j=0; j < numResults; j++)
		{
			const vec p = points[nearIndexCache[j]] - c;
			const float var = variance(i) + variance(nearIndexCache[j]);
			for(uint32_t di = 0; di < Dim; di++)
			{
				for(uint32_t dj = 0; dj < Dim; dj++)
				{
					nCovMat(di, dj) += var * p[di] * p[dj];
				}
			}
		}

		// Compute best normal
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, Dim, Dim>> eigenSolver;
		eigenSolver.compute(nCovMat, Eigen::DecompositionOptions::ComputeEigenvectors);
		Eigen::Vector3f eigenValues = eigenSolver.eigenvalues();

		uint32_t minIndex = 0;
		float minValue = glm::abs(eigenValues(0));
		for(uint32_t d=1; d < Dim; d++)
		{
			if(glm::abs(eigenValues(d)) < minValue)
			{
				minValue = glm::abs(eigenValues(d)); minIndex = d;
			}
		}

		auto minVec = eigenSolver.eigenvectors().col(minIndex);
		vec newNormal;
		for(uint32_t d=0; d < Dim; d++) newNormal[d] = minVec(d);
		newNormal = glm::normalize(newNormal);

		// Flip normals
		if(glm::dot(newNormal, normals[i]) >= 0.0) normals[i] = newNormal;
		else normals[i] = -newNormal;

		// TODO: compute variance
		for(uint32_t d=0; d < Dim; d++) normalsVar[i][d] = 1.0f;
	}
}

#endif


