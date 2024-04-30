#ifndef _POINT_CLOUD_INCLUDE
#define _POINT_CLOUD_INCLUDE


#include <vector>
#include <string>
#include <glm/glm.hpp>


using namespace std;


class PointCloud
{
public:
	bool readFromFile(const string &filename);
	
	unsigned int size() const;
	glm::vec2 &point(unsigned int index);
	const glm::vec2 &point(unsigned int index) const;
	glm::vec2 &normal(unsigned int index);
	const glm::vec2 &normal(unsigned int index) const;
	const vector<glm::vec2>& getPoints() const { return points; }

private:
	vector<glm::vec2> points, normals;

};


#endif


