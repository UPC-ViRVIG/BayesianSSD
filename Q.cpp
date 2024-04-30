#include <iostream>
#include "Quadtree.h"
#include "timing.h"
#include "AppParams.h"


QuadtreeNode::~QuadtreeNode()
{
	for(unsigned int i=0; i<4; i++)
		delete children[i];
}

void QuadtreeNode::subdivide(int levels)
{
	if((levels > 0) && (points.size() > 0))
	{
		float sx, sy;
		unsigned int node;
		
		sx = (minCoords.x + maxCoords.x) / 2.0f;
		sy = (minCoords.y + maxCoords.y) / 2.0f;
		for(unsigned int i=0; i<4; i++)
		{
			children[i] = new QuadtreeNode();
			for(unsigned int j=0; j<4; j++)
				children[i]->children[j] = NULL;
		}
		for(unsigned int j=0; j<points.size(); j++)
		{
			node = 0;
			node += (points[j].x < sx)?0:1;
			node += (points[j].y < sy)?0:2;
			children[node]->points.push_back(points[j]);
			children[node]->normals.push_back(normals[j]);
		}
		children[0]->minCoords = minCoords;
		children[0]->maxCoords = glm::vec2(sx, sy);
		children[1]->minCoords = glm::vec2(sx, minCoords.y);
		children[1]->maxCoords = glm::vec2(maxCoords.x, sy);
		children[2]->minCoords = glm::vec2(minCoords.x, sy);
		children[2]->maxCoords = glm::vec2(sx, maxCoords.y);
		children[3]->minCoords = glm::vec2(sx, sy);
		children[3]->maxCoords = maxCoords;

		for(unsigned int i=0; i<4; i++)
			children[i]->subdivide(levels-1);
	}
}

bool QuadtreeNode::isLeaf() const
{
	return (children[0] == NULL) && (children[1] == NULL) && (children[2] == NULL) && (children[3] == NULL);
}

QuadtreeNode *QuadtreeNode::pointToCell(const glm::vec2 &P, uint32_t* outNodeDepth)
{
	if(isLeaf())
		return this;
	else
	{
		unsigned int node = 0;
		node += (P.x < children[0]->maxCoords.x)?0:1;
		node += (P.y < children[0]->maxCoords.y)?0:2;
		if(outNodeDepth != nullptr) (*outNodeDepth)++;
		return children[node]->pointToCell(P, outNodeDepth);
	}
}

float QuadtreeNode::eval(glm::vec2 point, Quadtree* qtree)
{
	glm::vec2 np = (point - minCoords) / (maxCoords - minCoords);
	const float dx1 = qtree->unknownValues[cornerUnknowns[0]] * (1.0f - np.x) + qtree->unknownValues[cornerUnknowns[1]] * np.x;
	const float dx2 = qtree->unknownValues[cornerUnknowns[2]] * (1.0f - np.x) + qtree->unknownValues[cornerUnknowns[3]] * np.x;
	return dx1 * (1.0f - np.y) + dx2 * np.y;
}

void QuadtreeNode::eval(glm::vec2 point, array<float, 4>& outWeights)
{
	glm::vec2 np = (point - minCoords) / (maxCoords - minCoords);
	outWeights[0] = (1.0f - np.x) * (1.0f - np.y);
	outWeights[1] = np.x * (1.0f - np.y);
	outWeights[2] = (1.0f - np.x) * np.y;
	outWeights[3] = np.x * np.y;
}

void QuadtreeNode::evalGrad(glm::vec2 point, array<array<float, 4>, 2>& outWeights)
{
	glm::vec2 np = (point - minCoords) / (maxCoords - minCoords);
	// In X
	outWeights[0][0] = -(1.0f - np.y);
	outWeights[0][1] = (1.0f - np.y);
	outWeights[0][2] = -np.y;
	outWeights[0][3] = np.y;

	// In Y
	outWeights[1][0] = -(1.0f - np.x);
	outWeights[1][1] = -np.x;
	outWeights[1][2] = (1.0f - np.x);
	outWeights[1][3] = np.x;
}

void QuadtreeNode::collectCorners(Quadtree *qtree)
{
	map<int, int>:: iterator it;

	auto getCorner = [&] (glm::vec2 pos) -> unsigned int
	{
		unsigned int cornerIndex = qtree->pointToInteger(pos);
		it = qtree->cornerToUnknown.find(cornerIndex);
		if(it == qtree->cornerToUnknown.end())
		{
			qtree->cornerToUnknown[cornerIndex] = qtree->nUnknowns;
			qtree->unknownsInfo.push_back(Unknown(pos));
			return qtree->nUnknowns++;
		}
		else
		{
			unsigned index = qtree->cornerToUnknown[cornerIndex];
			qtree->unknownsInfo[index].numVertices++;
			return index;
		}
	};
	
	if(isLeaf())
	{
		cornerUnknowns[0] = getCorner(minCoords);
		cornerUnknowns[1] = getCorner(glm::vec2(maxCoords.x, minCoords.y));
		cornerUnknowns[2] = getCorner(glm::vec2(minCoords.x, maxCoords.y));
		cornerUnknowns[3] = getCorner(maxCoords);
	}
	else
	{
		for(unsigned int i=0; i<4; i++)
			children[i]->collectCorners(qtree);
	}
}

void QuadtreeNode::draw(Image &image) const
{
	if(isLeaf())
	{
		glm::ivec2 P, Q;
		
		P = glm::ivec2(minCoords * glm::vec2(image.width(), image.height()));
		Q = glm::ivec2(maxCoords * glm::vec2(image.width(), image.height()));
		P = glm::max(P, glm::ivec2(0, 0));
		Q = glm::min(Q, glm::ivec2(image.width()-1, image.height()-1));
		image.drawRectangle(P.x, P.y, Q.x, Q.y, glm::vec3(0.0f, 0.0f, 0.0f));
		
		for(unsigned int i=0; i<points.size(); i++)
			image.drawFilledCircle((unsigned int)(image.width() * points[i].x), (unsigned int)(image.height() * points[i].y), 4, glm::vec3(0.35f, 0.35f, 0.35f));
	}
	else
	{
		for(unsigned int i=0; i<4; i++)
			children[i]->draw(image);
	}
}


Quadtree::Quadtree()
{
	root = NULL;
	nUnknowns = 0;
	pW = gW = sW = 1.0f;
}

Quadtree::~Quadtree()
{
	if(root != NULL)
		delete root;
}

void Quadtree::setWeights(double pointEqWeight, double gradientEqWeight, double smoothnessEqWeight)
{
	pW = pointEqWeight;
	gW = gradientEqWeight;
	sW = smoothnessEqWeight;
}

void Quadtree::compute(const PointCloud &cloud, unsigned int levels, ScalarField &field)
{
	nUnknowns = 0;
	nLevels = levels;
	field.init((1 << levels) + 1, (1 << levels) + 1);

	// Create quatree
	root = new QuadtreeNode();
	for(unsigned int i=0; i<4; i++)
		root->children[i] = NULL;
	root->points.resize(cloud.size());
	for(unsigned int i=0; i<cloud.size(); i++)
		root->points[i] = cloud.point(i);
	root->normals.resize(cloud.size());
	for(unsigned int i=0; i<cloud.size(); i++)
		root->normals[i] = cloud.normal(i);
	
	root->minCoords = glm::vec2(0.0f, 0.0f);
	root->maxCoords = glm::vec2(1.0f, 1.0f);
	
	root->subdivide(levels);
	
	// Collect corners
	root->collectCorners(this);

	constrainTCorners();
	nUnknowns = unknownsInfo.size() - unknownConstraints.size();

	std::vector<uint32_t> unknownToMatUnknown(unknownsInfo.size());
	uint32_t matUnknownId = 0;
	for(uint32_t i = 0; i < unknownsInfo.size(); i++)
	{
		const Unknown& uk = unknownsInfo[i];
		if(!uk.isConstrained) unknownToMatUnknown[i] = matUnknownId++;
		else unknownToMatUnknown[i] = std::numeric_limits<uint32_t>::max();
	}
	
	if(AppParams::instance()->bLogging)
		cout << "# Unknowns = " << nUnknowns << endl;
	
	// Prepare linear system
	unsigned int nEquations;
	
	if(AppParams::instance()->bLogging)
		cout << "Preparing the system" << endl;
	long lastTime = getTimeMilliseconds();
	
	nEquations = 0;

	vector<Eigen::Triplet<double>> triplets;
	vector<float> bCoeffs;

	std::function<void(uint32_t, float)> setUnkownValue;
	setUnkownValue = [&](uint32_t unknownId, float value)
	{
		if(unknownToMatUnknown[unknownId] == std::numeric_limits<uint32_t>::max())
		{
			const ConstrainedUnknown& constraint = unknownConstraints[unknownsInfo[unknownId].constrainId];
			for(uint32_t i=0; i < constraint.numOperators; i++)
			{
				setUnkownValue(constraint.unknownIds[i], value * constraint.weights[i]);
			}
		}
		else
		{
			triplets.push_back(Eigen::Triplet<double>(nEquations, unknownToMatUnknown[unknownId], value));
		}
	};

	auto setConstraint = [&](std::initializer_list<std::tuple<uint32_t, float>> list, float res)
	{
		for(auto t : list)
		{
			setUnkownValue(std::get<0>(t), std::get<1>(t));
		}

		bCoeffs.push_back(res);
		nEquations++;
	};

	unknownValues.resize(unknownsInfo.size(), 0.0f);

	// Point-cloud constraints
	for(unsigned int i=0; i<cloud.size(); i++)
	{
		// Position constraint
		glm::vec2 nPos = cloud.point(i);
		QuadtreeNode* node = pointToCell(nPos);
		std::array<float, 4> weights;
		node->eval(nPos, weights);
		setConstraint({{node->cornerUnknowns[0], pW * weights[0]}, 
					   {node->cornerUnknowns[1], pW * weights[1]}, 
					   {node->cornerUnknowns[2], pW * weights[2]}, 
					   {node->cornerUnknowns[3], pW * weights[3]}},
					   0.0f);
		
		// Normals constraint
		glm::vec2 nNorm = cloud.normal(i);
		std::array<std::array<float, 4>, 2> gradweights;
		node->evalGrad(nPos, gradweights);
		// X derivative
		setConstraint({{node->cornerUnknowns[0], gW * gradweights[0][0]}, 
					   {node->cornerUnknowns[1], gW * gradweights[0][1]}, 
					   {node->cornerUnknowns[2], gW * gradweights[0][2]}, 
					   {node->cornerUnknowns[3], gW * gradweights[0][3]}},
					   gW * nNorm.x);
		
		// Y derivative
		setConstraint({{node->cornerUnknowns[0], gW * gradweights[1][0]}, 
					   {node->cornerUnknowns[1], gW * gradweights[1][1]}, 
					   {node->cornerUnknowns[2], gW * gradweights[1][2]}, 
					   {node->cornerUnknowns[3], gW * gradweights[1][3]}},
					   gW * nNorm.y);

	}

	const unsigned int numNodesAtMaxDepth = 1 << nLevels;
	const glm::vec2 nodeSize = (root->maxCoords - root->minCoords) / static_cast<float>(numNodesAtMaxDepth);
	const std::array<glm::vec2, 2> unknownSides = {
		glm::vec2(1.0f, 0.0f),
		glm::vec2(0.0f, 1.0f)
	};

	// Smooth constraints
	for(uint32_t i = 0; i < unknownsInfo.size(); i++)
	{
		const Unknown& uk = unknownsInfo[i];
		if(!uk.isConstrained)
		{
			uint32_t numValidSides = 0;
			for(glm::vec2 side : unknownSides)
			{
				bool valid = true;
				for(float sign : {-1.0f, 1.0f})
				{
					QuadtreeNode* node = pointToCell(uk.coord + sign * nodeSize * side);
					if(node == nullptr) { valid = false; break; }
				}

				if(!valid) continue;
				numValidSides++;

				for(float sign : {-1.0f, 1.0f})
				{
					QuadtreeNode* node = pointToCell(uk.coord + sign * nodeSize * side);
					std::array<float, 4> weights;
					node->eval(uk.coord + sign * nodeSize * side, weights);
					for(uint32_t i=0; i < 4; i++)
					{
						if(glm::abs(weights[i]) > 1e-8)
						{
							setUnkownValue(node->cornerUnknowns[i], sW * weights[i]);
						}
					}
				}
			}

			if(numValidSides > 0) setConstraint({{i, -sW * 2.0f * static_cast<float>(numValidSides)}}, 0.0f);
		}
		else
		{
			cout << "no constraint" << std::endl;
		}
	}
	
	if(AppParams::instance()->bLogging)
		cout << nEquations << " equations and " << nUnknowns << " unknowns" << endl;
	
	Eigen::SparseMatrix<double> A(nEquations, nUnknowns), AtA;
	Eigen::VectorXd b(nEquations), Atb, x(nUnknowns);

	A.setFromTriplets(triplets.begin(), triplets.end());
	for(unsigned int i=0; i<bCoeffs.size(); i++)
		b(i) = bCoeffs[i];

	if(AppParams::instance()->bLogging)
		cout << "Building A & b in " << (getTimeMilliseconds() - lastTime) << " ms" << endl;

	if(AppParams::instance()->bLogging)
		cout << "Computing normal equation" << endl;
	lastTime = getTimeMilliseconds();
	
	AtA = A.transpose() * A;
	Atb = A.transpose() * b;
	
	if(AppParams::instance()->bLogging)
		cout << "Computed AtA & Atb in " << (getTimeMilliseconds() - lastTime) << " ms" << endl;

	if(AppParams::instance()->bLogging)
		cout << "Solving least squares" << endl;
	lastTime = getTimeMilliseconds();

	// Method: Gradient descent (Wikipedia)
	Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
	solver.compute(AtA);
	x = solver.solve(Atb);

	cout << "Least squares solved in " << (getTimeMilliseconds() - lastTime) << " ms" << endl;

	std::function<float(uint32_t)> getUnkownValue;
	getUnkownValue = [&](uint32_t unknownId) -> float
	{
		if(unknownToMatUnknown[unknownId] == std::numeric_limits<uint32_t>::max())
		{
			const ConstrainedUnknown& constraint = unknownConstraints[unknownsInfo[unknownId].constrainId];
			float res = 0.0f;
			for(uint32_t i=0; i < constraint.numOperators; i++)
			{
				res += constraint.weights[i] * getUnkownValue(constraint.unknownIds[i]);
			}
			return res;
		}
		else
		{
			return x(unknownToMatUnknown[unknownId]);
		}
	};
	
	double relative_error = (A*x - b).norm() / b.norm();
	cout << "The relative error is: " << relative_error << endl;

	unknownValues.resize(unknownsInfo.size());
	for(uint32_t i=0; i < unknownToMatUnknown.size(); i++)
	{
		unknownValues[i] = getUnkownValue(i);
	}

	cout << "The relative error is: " << relative_error << endl;

	// for(unsigned int j=0, pos=0; j<field.height(); j++)
	// 	for(unsigned int i=0; i<field.width(); i++, pos++)
	// 		field(i, j) = x(pos);
}

void Quadtree::constrainTCorners()
{
	auto countNumNulls = [](std::array<QuadtreeNode*, 4>& array)
	{
		uint32_t nulls = 0;
		for(auto ptr : array)
		{
			if(ptr == nullptr) nulls++;
		}
		return nulls;
	};

	unknownConstraints.clear();
	unsigned int numNodesAtMaxDepth = 1 << nLevels;
	glm::vec2 nodeSize = (root->maxCoords - root->minCoords) / static_cast<float>(numNodesAtMaxDepth);
	for(uint32_t i=0; i < unknownsInfo.size(); i++)
	{
		Unknown& uk = unknownsInfo[i];
		if(uk.numVertices < 4) // Its a T-corner
		{
			array<QuadtreeNode*, 4> nodes;
			array<uint32_t, 4> nodesDepth;
			nodes[0] = pointToCell(uk.coord - 0.5f * nodeSize, &nodesDepth[0]);
			nodes[1] = pointToCell(uk.coord + glm::vec2(0.5f, -0.5f) * nodeSize, &nodesDepth[1]);
			nodes[2] = pointToCell(uk.coord + glm::vec2(-0.5f, 0.5f) * nodeSize, &nodesDepth[2]);
			nodes[3] = pointToCell(uk.coord + 0.5f * nodeSize, &nodesDepth[3]);

			if(uk.numVertices >= 4 - countNumNulls(nodes)) continue; // Its just a octree boundary not a T-corner

			uint32_t minDepth = nLevels + 1;
			QuadtreeNode* minDepthNode;
			for(uint32_t n=0; n < nodes.size(); n++)
			{
				if(nodes[n] != nullptr && 
				   minDepth > nodesDepth[n])
				{
					minDepthNode = nodes[n];
					minDepth = nodesDepth[n];
				}
			}
			
			ConstrainedUnknown constrain;
			std::array<float, 4> nodeWeights;
			minDepthNode->eval(uk.coord, nodeWeights);

			constrain.numOperators = 0;
			for(uint32_t i=0; i < 4; i++)
			{
				if(glm::abs(nodeWeights[i]) > 1e-9)
				{
					const uint32_t oId = constrain.numOperators;
					constrain.weights[oId] = nodeWeights[i];
					constrain.unknownIds[oId] = minDepthNode->cornerUnknowns[i];
					constrain.numOperators++;
				}
			}

			unknownConstraints.push_back(constrain);
			uk.isConstrained = true;
			uk.constrainId = unknownConstraints.size() - 1;
		}
	}
}

float Quadtree::getMaxAbsValue()
{
	float res = 0.0f;
	for(float val : unknownValues)
	{
		res = glm::max(res, glm::abs(val));
	}
	return res;
}


void Quadtree::draw(Image &image)
{
	const std::array<glm::vec3, 7> palette = {
		glm::vec3(0.0f, 0.0f, 1.0f), 
		glm::vec3(0.0f, 0.5f, 1.0f), 
		glm::vec3(0.0f, 1.0f, 1.0f), 
		glm::vec3(1.0f, 1.0f, 1.0f), 
		glm::vec3(1.0f, 1.0f, 0.0f), 
		glm::vec3(1.0f, 0.5f, 0.0f), 
		glm::vec3(1.0f, 0.0f, 0.0f)
	};

	if(image.width() == 0)
		return;
	image.fill(glm::vec3(0.0f, 0.0f, 0.0f));
	const float maxValue = getMaxAbsValue();
	for(uint32_t i=0; i < image.width(); i++)
	{
		for(uint32_t j=0; j < image.height(); j++)
		{
			glm::vec2 pos = glm::vec2(static_cast<float>(i) / static_cast<float>(image.width()), 
									  static_cast<float>(j) / static_cast<float>(image.height()));
			pos = root->minCoords + pos * (root->maxCoords - root->minCoords);
			float val = glm::clamp(eval(pos) / maxValue, -1.0f, 0.999999f);
			val = static_cast<float>(palette.size() - 1) * 0.5f * (val + 1.0f);
			uint32_t cid = glm::floor(val);
			image(i, j) = glm::mix(palette[cid], palette[cid+1], glm::fract(val));
		}
	}

	root->draw(image);
	/*
	for(map<int, int>::iterator it=cornerToUnknown.begin(); it!=cornerToUnknown.end(); it++)
	{
		int cornerIndex = it->first;
		float y = float(cornerIndex / ((1 << nLevels) + 1)) / (1 << nLevels);
		float x = float(cornerIndex % ((1 << nLevels) + 1)) / (1 << nLevels);
		image.drawFilledCircle(int(image.width() * x), int(image.height() * y), 4, glm::vec3(0.75f, 0.75f, 0.75f));
	}
	*/
}

float Quadtree::eval(glm::vec2 point)
{
	QuadtreeNode* node = pointToCell(point);
	return (node == nullptr) ? NAN : node->eval(point, this);
}

// From: Point in sampling space
// To:   Quadtree node that contains it

QuadtreeNode *Quadtree::pointToCell(const glm::vec2 &P, uint32_t* outNodeDepth)
{
	// Check if its inside the octree
	if(glm::any(glm::lessThan(P, root->minCoords)) || 
	   glm::any(glm::greaterThan(P, root->maxCoords)))
	{
		return nullptr;
	}

	if(outNodeDepth != nullptr) *outNodeDepth = 0;
	return root->pointToCell(P, outNodeDepth);
}

// From: Point on a corner of the limit uniform grid
// To:   Unique integer id

unsigned int Quadtree::pointToInteger(const glm::vec2 &P) const
{
	return (unsigned int)round((1 << nLevels) * P.y) * ((1 << nLevels) + 1) + (unsigned int)round((1 << nLevels) * P.x);
}

void Quadtree::addPointEquation(unsigned int eqIndex, const glm::vec2 &P, vector<Eigen::Triplet<double>> &triplets, vector<float> &bCoeffs)
{
	QuadtreeNode *node;
	float x, y;
	
	node = pointToCell(P);
	x = (P.x - node->minCoords.x) / (node->maxCoords.x - node->minCoords.x);
	y = (P.y - node->minCoords.y) / (node->maxCoords.y - node->minCoords.y);
	triplets.push_back(Eigen::Triplet<double>(eqIndex, node->cornerUnknowns[3], pW*x*y));
	triplets.push_back(Eigen::Triplet<double>(eqIndex, node->cornerUnknowns[2], pW*(1.0f-x)*y));
	triplets.push_back(Eigen::Triplet<double>(eqIndex, node->cornerUnknowns[1], pW*x*(1.0f-y)));
	triplets.push_back(Eigen::Triplet<double>(eqIndex, node->cornerUnknowns[0], pW*(1.0f-x)*(1.0f-y)));
	bCoeffs.push_back(0.0f);
}

void Quadtree::addGradientEquations(unsigned int eqIndex, const glm::vec2 &P, const glm::vec2 &N, vector<Eigen::Triplet<double>> &triplets, vector<float> &bCoeffs)
{
	QuadtreeNode *node;
	float x, y;
	
	node = pointToCell(P);
	x = (P.x - node->minCoords.x) / (node->maxCoords.x - node->minCoords.x);
	y = (P.y - node->minCoords.y) / (node->maxCoords.y - node->minCoords.y);
	triplets.push_back(Eigen::Triplet<double>(eqIndex, node->cornerUnknowns[3], pW*x*y));
	triplets.push_back(Eigen::Triplet<double>(eqIndex, node->cornerUnknowns[2], pW*(1.0f-x)*y));
	triplets.push_back(Eigen::Triplet<double>(eqIndex, node->cornerUnknowns[1], pW*x*(1.0f-y)));
	triplets.push_back(Eigen::Triplet<double>(eqIndex, node->cornerUnknowns[0], pW*(1.0f-x)*(1.0f-y)));
	bCoeffs.push_back(0.0f);
	
	triplets.push_back(Eigen::Triplet<double>(eqIndex, node->cornerUnknowns[3], gW*y));
	triplets.push_back(Eigen::Triplet<double>(eqIndex, node->cornerUnknowns[2], -gW*y));
	triplets.push_back(Eigen::Triplet<double>(eqIndex, node->cornerUnknowns[1], gW*(1.0f-y)));
	triplets.push_back(Eigen::Triplet<double>(eqIndex, node->cornerUnknowns[0], -gW*(1.0f-y)));
	bCoeffs.push_back(gW*N.x);

	triplets.push_back(Eigen::Triplet<double>(eqIndex+1, node->cornerUnknowns[3], gW*x));
	triplets.push_back(Eigen::Triplet<double>(eqIndex+1, node->cornerUnknowns[2], gW*(1.0f-x)));
	triplets.push_back(Eigen::Triplet<double>(eqIndex+1, node->cornerUnknowns[1], -gW*x));
	triplets.push_back(Eigen::Triplet<double>(eqIndex+1, node->cornerUnknowns[0], -gW*(1.0f-x)));
	bCoeffs.push_back(gW*N.y);
}







