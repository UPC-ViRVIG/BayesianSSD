#ifndef NODE_TREE_H
#define NODE_TREE_H

#include <functional>
#include <map>
#include <vector>
#include <array>
#include <memory>
#include <optional>
#include "Vector.h"
#include "PointCloud.h"

/*
    Tree of nodes subdiving a part of the space
*/
template<uint32_t Dim>
class NodeTree
{
public:
    static constexpr uint32_t Dim = Dim;
    using vec = VStruct<Dim>::type;
    static constexpr uint32_t NumNeighboursVert = 2 * Dim;
    static constexpr uint32_t NumAdjacentNodes = 1 << Dim;

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

    struct Config
    {
        vec minCoord;
        vec maxCoord;
        float pointFilterMaxDistance;
        bool constraintNeighbourNodes;
        uint32_t maxDepth;
    };

	void compute(const PointCloud<Dim> &cloud, Config config);

    uint32_t getNumVertices() const { return verticesPos.size(); }
    const std::vector<vec>& getVertices() const { return verticesPos; };
    const std::vector<uint32_t>& getTJointVerticesIndex() const { return tJointVertices; };
    void getNode(vec point, std::optional<Node>& outNode) const;
    uint32_t getAdjacentNodes(uint32_t vertId, std::array<NodeTree::Node, NumNeighboursVert>& outNodes) const;
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
void NodeTree<Dim>::compute(const PointCloud<Dim> &cloud, Config config)
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

    const uint32_t maxDepth = config.maxDepth;
    std::map<uint32_t, uint32_t> idxToCPid;
    std::vector<ControlPoint> cpInfo;
    verticesPos.clear();
    tJointVertices.clear();
    uint32_t numVertices = 0;

    octreeData.clear();
    octreeData.push_back(InternalNode()); // Create root node
    minCoord = config.minCoord;
    maxCoord = config.maxCoord;
    NodeInfo root {0, minCoord, maxCoord, 0};    
    this->maxDepth = maxDepth;

    const float octreeLength = static_cast<float>(1 << maxDepth);
    const vec nodeSize = (root.maxCoord - root.minCoord) / octreeLength;
    const vec octreeInvSize = 1.0f / (root.maxCoord - root.minCoord);
    auto getCPid = [&](vec point) -> uint32_t
    {
        vec norm = (point - root.minCoord) * octreeInvSize;
        uint32_t idx = 0;
        for(uint32_t i=Dim-1; i < Dim; i--)
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
    const float filterMaxDistanceSq =  config.pointFilterMaxDistance *  config.pointFilterMaxDistance;
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
            float halfNodeSize = 0.5f * (nodeInfo.maxCoord - nodeInfo.minCoord).x;
            for(vec p : nodePoints)
            {
                for(uint32_t i=0; i < numNodes; i++)
                {
                    float sqDist = 0.0f;
                    for(uint32_t j=0; j < Dim; j++)
                    {
                        float off = (i & (1 << j)) ? halfNodeSize : -halfNodeSize;
                        float q = glm::abs(p[j] - center[j] - off) - halfNodeSize;
                        sqDist += glm::max(q, 0.0f) * glm::max(q, 0.0f);
                    }

                    if(sqDist < filterMaxDistanceSq + 1e-8)
                        points[i].push_back(p);   
                }

                // uint32_t nodeIdx = 0;
                // for(uint32_t i=0; i < Dim; i++)
                //     nodeIdx += ((p[i] < center[i]) ? 0 : (1 << i));
                // points[nodeIdx].push_back(p);                        
            }

            for(uint32_t i=0; i < 4; i++)
            {
                NodeInfo nInfo;
                nInfo.nodeIndex = chIndex + i;
                nInfo.depth = nodeInfo.depth + 1;
                for(uint32_t j=0; j < Dim; j++)
                {
                    nInfo.minCoord[j] = (i & (1 << j)) ? center[j] : nodeInfo.minCoord[j];
                    nInfo.maxCoord[j] = (i & (1 << j)) ? nodeInfo.maxCoord[j] : center[j];
                }

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
                glm::vec2 cp;
                for(uint32_t j=0; j < Dim; j++)
                    cp[j] = (i & (1 << j)) ? nodeInfo.maxCoord[j] : nodeInfo.minCoord[j];

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
        if(cp.numCorners < numNodes)
        {
            std::optional<Node> node;
            uint32_t numNotFound = 0;
            for(uint32_t i=0; i < numNodes; i++)
            {
                vec offset;
                for(uint32_t j=0; j < Dim; j++)
                    offset[j] = (i & (1 << j)) ? 0.5f : -0.5f;

                getNode(cp.coord + offset * nodeSize, node);

                numNotFound += node ? 0 : 1;
            }

            if(cp.numCorners >= numNodes - numNotFound) continue; // Its just a octree boundary not a T-corner

            tJointVertices.push_back(i);
        }
    }
}

template<uint32_t Dim>
void NodeTree<Dim>::getNode(vec point, std::optional<Node>& outNode) const
{
    if(glm::any(glm::lessThan(point, minCoord)) || 
	   glm::any(glm::greaterThan(point, maxCoord)))
	{
		outNode = std::optional<Node>();
        return;
	}

    outNode = std::optional<Node>(Node());
    outNode->minCoord = minCoord;
    outNode->maxCoord = maxCoord;
    outNode->depth = 0;

    uint32_t cIndex = 0;
    while(!octreeData[cIndex].isLeaf())
    {
        glm::vec2 center = 0.5f * (outNode->maxCoord + outNode->minCoord);
        uint32_t chIdx = 0;
        for(uint32_t i=0; i < Dim; i++)
        {
            chIdx += ((point[i] < center[i]) ? 0 : (1 << i));
            outNode->minCoord[i] = (point[i] < center[i]) ? outNode->minCoord[i] : center[i];
            outNode->maxCoord[i] = (point[i] < center[i]) ? center[i] : outNode->maxCoord[i];
        }

        cIndex = octreeData[cIndex].getChildrenIndex() + chIdx;

        outNode->depth++;
    }

    outNode->controlPointsIdx = octreeData[cIndex].controlPointsIdx;
}

template<uint32_t Dim>
uint32_t NodeTree<Dim>::getAdjacentNodes(uint32_t vertId, std::array<NodeTree::Node, NumNeighboursVert>& outNodes) const
{
    const float octreeLength = static_cast<float>(1 << maxDepth);
    const glm::vec2 nodeSize = (maxCoord - minCoord) / octreeLength;

    const glm::vec2 coord = getVertices()[vertId];
    constexpr uint32_t numNodes = 1 << Dim;
    std::optional<Node> node;
    uint32_t index = 0;
    for(uint32_t i=0; i < numNodes; i++)
    {
        vec offset;
        for(uint32_t j=0; j < Dim; j++)
            offset[j] = (i & (1 << j)) ? 0.5f : -0.5f;

        getNode(coord + offset * nodeSize, node);

        if(node)
        {
            outNodes[index++] = node.value();
        }
    }

    return index;
}

#endif