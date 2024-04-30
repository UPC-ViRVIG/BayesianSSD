#ifndef _QUADTREE_INCLUDE
#define _QUADTREE_INCLUDE


#include <vector>
#include <map>
#include <glm/glm.hpp>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include "PointCloud.h"
#include "ScalarField.h"


using namespace std;


class Quadtree;


struct ConstrainedUnknown
{
	array<unsigned int, 4> unknownIds;
	array<float, 4> weights;
	uint32_t numOperators;
};

struct Unknown
{	
	Unknown(glm::vec2 pos) : coord(pos), numVertices(1), 
							 isConstrained(false), constrainId(0) {}
	glm::vec2 coord;
	int numVertices;
	bool isConstrained;
	uint32_t constrainId;
};

struct QuadtreeNode
{
	~QuadtreeNode();
	
	void subdivide(int levels);
	
	bool isLeaf() const;
	QuadtreeNode *pointToCell(const glm::vec2 &P, uint32_t* outNodeDepth=nullptr);
	void collectCorners(Quadtree *qtree);
	float eval(glm::vec2 point, Quadtree* qtree);
	void eval(glm::vec2 point, array<float, 4>& outWeights);
	void evalGrad(glm::vec2 point, array<array<float, 4>, 2>& outWeights);

	void draw(Image &image) const;

public:
	glm::vec2 minCoords, maxCoords;
	vector<glm::vec2> points, normals;
	QuadtreeNode *children[4];
	// Unknown id for each of the node's corners (0-> (minx, miny), 1-> (maxx, miny), 2-> (minx, maxy), 3-> (maxx, maxy))
	int cornerUnknowns[4];
	
};


class Quadtree
{
public:
	Quadtree();
	~Quadtree();
	
	void setWeights(double pointEqWeight, double gradientEqWeight, double smoothnessEqWeight);
	void compute(const PointCloud &cloud, unsigned int levels, ScalarField &field);

	float eval(glm::vec2 point);
	
	void draw(Image &image);
	
private:
	QuadtreeNode *pointToCell(const glm::vec2 &P, uint32_t* outNodeDepth=nullptr);
	unsigned int pointToInteger(const glm::vec2 &P) const;

	void constrainTCorners();
	float getMaxAbsValue();

	void addPointEquation(unsigned int eqIndex, const glm::vec2 &P, vector<Eigen::Triplet<double>> &triplets, vector<float> &bCoeffs);
	void addGradientEquations(unsigned int eqIndex, const glm::vec2 &P, const glm::vec2 &N, vector<Eigen::Triplet<double>> &triplets, vector<float> &bCoeffs);

private:
	unsigned int nLevels, nUnknowns;
	QuadtreeNode *root;
	// Maps unique corner id to unknown id in the linear system
	map<int, int> cornerToUnknown;
	std::vector<Unknown> unknownsInfo;
	vector<ConstrainedUnknown> unknownConstraints;
	std::vector<float> unknownValues;


	//std::vector<std::vector<>>
	double pW, gW, sW;
	
	friend class QuadtreeNode;

};


#endif


