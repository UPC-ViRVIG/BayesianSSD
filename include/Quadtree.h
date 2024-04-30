#ifndef QUADTREE_H
#define QUADTREE_H

#include <map>
#include <vector>
#include <array>
#include <memory>
#include "PointCloud.h"
#include <optional>

/*
    Octree based on control points
*/
class Quadtree
{
public:
    static constexpr uint32_t Dim = 2;

    struct Node
    {
        glm::vec2 minCoord;
        glm::vec2 maxCoord;
        uint32_t depth;
        std::array<uint32_t, 4> controlPointsIdx;

        glm::vec2 transformToLocalCoord(glm::vec2 point)
        {
            return (point - minCoord) / (maxCoord - minCoord);
        }
    };
	
	void compute(const PointCloud &cloud, uint32_t maxDepth);

    uint32_t getNumVertices() const { return numVertices; }
    const std::vector<glm::vec2>& getVertices() const { return verticesPos; };
    const std::vector<uint32_t>& getTJointVerticesIndex() const { return tJointVertices; };
    void getNode(glm::vec2 point, std::optional<Node>& outNode) const;
    uint32_t getAdjacentNodes(uint32_t vertId, std::array<Quadtree::Node, 4>& outNodes) const;
    glm::vec2 getMinOctreeCoord() const { return minCoord; }
    glm::vec2 getMaxOctreeCoord() const { return maxCoord; }

private:
	struct InternalNode
    {
        InternalNode() {}
        InternalNode(bool isLeaf, uint32_t index = std::numeric_limits<uint32_t>::max())
        {
            childrenIndex = (index & CHILDREN_INDEX_MASK) | 
                            ((isLeaf) ? IS_LEAF_MASK : 0);
        }

        static constexpr uint32_t IS_LEAF_MASK = 1 << 31;
        static constexpr uint32_t CHILDREN_INDEX_MASK = ~(IS_LEAF_MASK);

        uint32_t childrenIndex;
        std::array<uint32_t, 4> controlPointsIdx;

        inline bool isLeaf() const
        {
            return childrenIndex & IS_LEAF_MASK;
        }

        inline uint32_t getChildrenIndex() const
        {
            return childrenIndex & CHILDREN_INDEX_MASK;
        }

        inline void setValues(bool isLeaf, uint32_t index = std::numeric_limits<uint32_t>::max())
        {
            childrenIndex = (index & CHILDREN_INDEX_MASK) | 
                            ((isLeaf) ? IS_LEAF_MASK : 0);
        }
    };

    // Octree dimensions
    glm::vec2 minCoord;
    glm::vec2 maxCoord;
    uint32_t maxDepth;

    // Octree data
    std::vector<InternalNode> octreeData;
    uint32_t numVertices;
    std::vector<glm::vec2> verticesPos;
    std::vector<uint32_t> tJointVertices;
};


#endif