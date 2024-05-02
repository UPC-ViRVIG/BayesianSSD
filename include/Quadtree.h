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
template<uint32_t Dim>
class Quadtree
{
public:
    static constexpr uint32_t Dim = Dim;
    using vec = VStruct<Dim>::type;
    static constexpr uint32_t NumNeighboursVert = 2 * Dim;

    struct Node
    {
        vec minCoord;
        vec maxCoord;
        uint32_t depth;
        std::array<uint32_t, NumNeighboursVert> controlPointsIdx;

        glm::vec2 transformToLocalCoord(vec point)
        {
            return (point - minCoord) / (maxCoord - minCoord);
        }
    };
	
	void compute(const PointCloud<Dim> &cloud, uint32_t maxDepth);

    uint32_t getNumVertices() const { return verticesPos.size(); }
    const std::vector<vec>& getVertices() const { return verticesPos; };
    const std::vector<uint32_t>& getTJointVerticesIndex() const { return tJointVertices; };
    void getNode(vec point, std::optional<Node>& outNode) const;
    uint32_t getAdjacentNodes(uint32_t vertId, std::array<Quadtree::Node, NumNeighboursVert>& outNodes) const;
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
        std::array<uint32_t, NumNeighboursVert> controlPointsIdx;

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
    vec minCoord;
    vec maxCoord;
    uint32_t maxDepth;

    // Octree data
    std::vector<InternalNode> octreeData;
    std::vector<vec> verticesPos;
    std::vector<uint32_t> tJointVertices;
};


template<uint32_t Dim>
void Quadtree::compute(const PointCloud<Dim> &cloud, uint32_t maxDepth)
{
    using Points = std::vector<vec>;
    struct NodeInfo
    {
        uint32_t nodeIndex;
        vec minCoord;
        vec maxCoord;
        uint32_t depth;
    };

    struct ControlPoint
    {
        vec coord;
        uint32_t numCorners;
    };
    std::map<uint32_t, uint32_t> idxToCPid;
    std::vector<ControlPoint> cpInfo;
    verticesPos.clear();
    tJointVertices.clear();
    numVertices = 0;

    octreeData.clear();
    octreeData.push_back(InternalNode()); // Create root node
    NodeInfo root {0, vec(0.0f), vec(1.0f), 0};
    minCoord = root.minCoord;
    maxCoord = root.maxCoord;
    this->maxDepth = maxDepth;

    const float octreeLength = static_cast<float>(1 << maxDepth);
    const vec nodeSize = (root.maxCoord - root.minCoord) / octreeLength;
    const vec octreeInvSize = 1.0f / (root.maxCoord - root.minCoord);
    auto getCPid = [&](vec point) -> uint32_t
    {
        vec norm = (point - root.minCoord) * octreeInvSize;
        uint32_t idx = 0;
        for(uint32_t i=Dim-1; i >= 0; i--)
        {
            idx = ((1 << maxDepth) + 1) * idx + glm::round(norm[i] * octreeLength);
        }
        
        auto it = idxToCPid.find(idx);
        if(it == idxToCPid.end())
        {
            idxToCPid[idx] = numVertices;
            cpInfo.push_back(ControlPoint {point, 1});
            verticesPos.push_back(point);
            return numVertices++;
        }
        else
        {
            cpInfo[it->second].numCorners++;
            return it->second;
        }
    };

    constexpr uint32_t numNodes = 1 << Dim;
    std::function<void(NodeInfo&, const Points&)> createNode;
    createNode = [&](NodeInfo& nodeInfo, const Points& nodePoints) -> void
    {
        if(nodeInfo.depth < maxDepth && nodePoints.size() > 0) // Create childrens
        {
            const uint32_t chIndex = octreeData.size();
            octreeData.resize(octreeData.size() + 4);
            octreeData[nodeInfo.nodeIndex].setValues(false, chIndex);
            std::array<Points, numNodes> points;
            points.fill(Points());
            vec center = 0.5f * (nodeInfo.maxCoord + nodeInfo.minCoord);
            for(vec p : nodePoints)
            {
                const uint32_t nodeIdx = ((p.x < center.x) ? 0 : 1) +
                                         ((p.y < center.y) ? 0 : 2);
                points[nodeIdx].push_back(p);
            }

            for(uint32_t i=0; i < 4; i++)
            {
                NodeInfo nInfo {chIndex + i,
                                glm::vec2(
                                    (i & 0b001) ? center.x : nodeInfo.minCoord.x,
                                    (i & 0b010) ? center.y : nodeInfo.minCoord.y
                                ), 
                                glm::vec2(
                                    (i & 0b001) ? nodeInfo.maxCoord.x : center.x,
                                    (i & 0b010) ? nodeInfo.maxCoord.y : center.y
                                ),
                                nodeInfo.depth + 1};
                createNode(nInfo, points[i]);
            }
        }
        else
        {
            InternalNode& node = octreeData[nodeInfo.nodeIndex];
            node.setValues(true);
            // Get control points
            for(uint32_t i=0; i < 4; i++)
            {
                glm::vec2 cp(
                    (i & 0b001) ? nodeInfo.maxCoord.x : nodeInfo.minCoord.x,
                    (i & 0b010) ? nodeInfo.maxCoord.y : nodeInfo.minCoord.y
                );
                node.controlPointsIdx[i] = getCPid(cp);
            }
        }
    };

    createNode(root, cloud.getPoints());

    // Recolect Tjoints
    tJointVertices.clear();
    for(uint32_t i=0; i < cpInfo.size(); i++)
    {
        const ControlPoint& cp = cpInfo[i];
        if(cp.numCorners < 4)
        {
            std::array<std::optional<Node>, 4> nodes;
            getNode(cp.coord - 0.5f * nodeSize, nodes[0]);
            getNode(cp.coord + glm::vec2(0.5f, -0.5f) * nodeSize, nodes[1]);
            getNode(cp.coord + glm::vec2(-0.5f, 0.5f) * nodeSize, nodes[2]);
            getNode(cp.coord + 0.5f * nodeSize, nodes[3]);

            uint32_t numNotFound = 0;
            for(auto& n : nodes) numNotFound += (n) ? 0 : 1;

            if(cp.numCorners >= 4 - numNotFound) continue; // Its just a octree boundary not a T-corner

            tJointVertices.push_back(i);
        }
    }
}



#endif