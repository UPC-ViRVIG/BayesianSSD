#ifndef _POINT_CLOUD_INCLUDE
#define _POINT_CLOUD_INCLUDE


#include <vector>
#include <string>
#include <fstream>
#include <happly.h>
#include "Vector.h"

template<uint32_t Dim>
class PointCloud
{
public:
	using vec = VStruct<Dim>::type;
	bool readFromFile(const std::string &filename) { return false; }
	
	uint32_t size() const { return points.size(); }
	vec &point(uint32_t index) { return points[index]; }
	const vec &point(uint32_t index) const { return points[index]; }
	vec &normal(uint32_t index) { return normals[index]; }
	const vec &normal(uint32_t index) const { return normals[index]; }
	const std::vector<vec>& getPoints() const { return points; }
	std::vector<vec>& getPoints() { return points; }

private:
	std::vector<vec> points, normals;

};

template<>
bool PointCloud<2>::readFromFile(const std::string &filename)
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
	for(unsigned int i=0; i<n; i++)
	{
		fin >> v.x >> v.y;
		points[i] = v;
		fin >> v.x >> v.y;
		v = glm::normalize(v);
		normals[i] = v;
	}
	fin.close();
	
	return true;
}

template<>
bool PointCloud<3>::readFromFile(const std::string &filename)
{
	happly::PLYData plyIn(filename);

	// Read pos
	std::vector<float> xProp = plyIn.getElement("vertex").getProperty<float>("x");
    std::vector<float> yProp = plyIn.getElement("vertex").getProperty<float>("y");
    std::vector<float> zProp = plyIn.getElement("vertex").getProperty<float>("z");

	// Read normal
	std::vector<float> nxProp = plyIn.getElement("vertex").getProperty<float>("nx");
    std::vector<float> nyProp = plyIn.getElement("vertex").getProperty<float>("ny");
    std::vector<float> nzProp = plyIn.getElement("vertex").getProperty<float>("nz");

	points.resize(xProp.size());
	normals.resize(xProp.size());

	for(uint32_t i=0; i < xProp.size(); i++)
	{
		points[i] = glm::vec3(xProp[i], yProp[i], zProp[i]);
		normals[i] = glm::vec3(nxProp[i], nyProp[i], nzProp[i]);
	}
	
	return true;
}



#endif


