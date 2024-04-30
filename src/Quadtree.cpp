#include "Quadtree.h"

#include <functional>
#include <glm/glm.hpp>

void Quadtree::compute(const PointCloud &cloud, uint32_t maxDepth)
{
    using Points = std::vector<glm::vec2>;
    struct NodeInfo
    {
        uint32_t nodeIndex;
        glm::vec2 minCoord;
        glm::vec2 maxCoord;
        uint32_t depth;
    };

    struct ControlPoint
    {
        glm::vec2 coord;
        uint32_t numCorners;
    };
    std::map<uint32_t, uint32_t> idxToCPid;
    std::vector<ControlPoint> cpInfo;
    verticesPos.clear();
    tJointVertices.clear();
    numVertices = 0;

    octreeData.clear();
    octreeData.push_back(InternalNode()); // Create root node
    NodeInfo root {0, glm::vec2(0.0f), glm::vec2(1.0f), 0};
    minCoord = root.minCoord;
    maxCoord = root.maxCoord;
    this->maxDepth = maxDepth;

    const float octreeLength = static_cast<float>(1 << maxDepth);
    const glm::vec2 nodeSize = (root.maxCoord - root.minCoord) / octreeLength;
    const glm::vec2 octreeInvSize = 1.0f / (root.maxCoord - root.minCoord);
    auto getCPid = [&](glm::vec2 point) -> uint32_t
    {
        glm::vec2 norm = (point - root.minCoord) * octreeInvSize;
        uint32_t idx = ((1 << maxDepth) + 1) * glm::round(norm.y * octreeLength) + glm::round(norm.x * octreeLength);
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

    std::function<void(NodeInfo&, const Points&)> createNode;
    createNode = [&](NodeInfo& nodeInfo, const Points& nodePoints) -> void
    {
        if(nodeInfo.depth < maxDepth && nodePoints.size() > 0) // Create childrens
        {
            const uint32_t chIndex = octreeData.size();
            octreeData.resize(octreeData.size() + 4);
            octreeData[nodeInfo.nodeIndex].setValues(false, chIndex);
            std::array<Points, 4> points;
            points.fill(Points());
            glm::vec2 center = 0.5f * (nodeInfo.maxCoord + nodeInfo.minCoord);
            for(glm::vec2 p : nodePoints)
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

void Quadtree::getNode(glm::vec2 point, std::optional<Node>& outNode) const
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
        const uint32_t chIdx = ((point.x < center.x) ? 0 : 1) +
                               ((point.y < center.y) ? 0 : 2);
        
        outNode->minCoord = glm::vec2((point.x < center.x) ? outNode->minCoord.x : center.x,
                                      (point.y < center.y) ? outNode->minCoord.y : center.y);
        outNode->maxCoord = glm::vec2((point.x < center.x) ? center.x : outNode->maxCoord.x,
                                      (point.y < center.y) ? center.y : outNode->maxCoord.y);

        cIndex = octreeData[cIndex].getChildrenIndex() + chIdx;

        outNode->depth++;
    }

    outNode->controlPointsIdx = octreeData[cIndex].controlPointsIdx;
}

uint32_t Quadtree::getAdjacentNodes(uint32_t vertId, std::array<Quadtree::Node, 4>& outNodes) const
{
    const float octreeLength = static_cast<float>(1 << maxDepth);
    const glm::vec2 nodeSize = (maxCoord - minCoord) / octreeLength;

    const glm::vec2 coord = getVertices()[vertId];
    std::array<std::optional<Node>, 4> nodes;
    getNode(coord - 0.5f * nodeSize, nodes[0]);
    getNode(coord + glm::vec2(0.5f, -0.5f) * nodeSize, nodes[1]);
    getNode(coord + glm::vec2(-0.5f, 0.5f) * nodeSize, nodes[2]);
    getNode(coord + 0.5f * nodeSize, nodes[3]);
    
    uint32_t index = 0;
    for(std::optional<Node>& node : nodes)
    {
        if(node)
        {
            outNodes[index++] = node.value();
        }
    }

    return index;
}