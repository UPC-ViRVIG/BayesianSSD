#include <fstream>
#include "PointCloud.h"


bool PointCloud::readFromFile(const string &filename)
{
	ifstream fin;
	unsigned int n;
	glm::vec2 v;
	
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

unsigned int PointCloud::size() const
{
	return points.size();
}

glm::vec2 &PointCloud::point(unsigned int index)
{
	return points[index];
}

const glm::vec2 &PointCloud::point(unsigned int index) const
{
	return points[index];
}

glm::vec2 &PointCloud::normal(unsigned int index)
{
	return normals[index];
}

const glm::vec2 &PointCloud::normal(unsigned int index) const
{
	return normals[index];
}


