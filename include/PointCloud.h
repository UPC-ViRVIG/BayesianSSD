#ifndef _POINT_CLOUD_INCLUDE
#define _POINT_CLOUD_INCLUDE


#include <vector>
#include <string>
#include <glm/glm.hpp>


template<uint32_t Dim>
class PointCloud
{
public:
	bool readFromFile(const std::string &filename);
	
	unsigned int size() const;
	vec &point(unsigned int index);
	const vec &point(unsigned int index) const;
	vec &normal(unsigned int index);
	const vec &normal(unsigned int index) const;
	const std::vector<vec>& getPoints() const { return points; }

private:
	std::vector<vec> points, normals;

};


#endif


