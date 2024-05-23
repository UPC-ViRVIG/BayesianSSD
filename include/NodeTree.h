#ifndef NODE_TREE_H
#define NODE_TREE_H

#include <functional>
#include <map>
#include <vector>
#include <array>
#include <memory>
#include <stack>
#include <optional>
#include "Vector.h"
#include "PointCloud.h"

template<uint32_t Dim>
class LinearNodeTree;

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
    static constexpr uint32_t NumVerticesPerNode = 1 << Dim;

    struct Node
    {
        vec minCoord;
        vec maxCoord;
        uint32_t depth;
        std::array<uint32_t, NumVerticesPerNode> controlPointsIdx;

        vec transformToLocalCoord(vec point)
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
    uint32_t getAdjacentNodes(uint32_t vertId, std::array<NodeTree::Node, NumAdjacentNodes>& outNodes) const;
    vec getMinCoord() const { return minCoord; }
    vec getMaxCoord() const { return maxCoord; }
    uint32_t getMaxDepth() const { return maxDepth; }

    friend class LinearNodeTree<Dim>;

    // TODO: move this struct to private
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
        std::array<uint32_t, NumVerticesPerNode> controlPointsIdx;

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

    struct const_iterator
    {
    public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type   = std::ptrdiff_t;
        using value_type        = Node;
        using pointer           = Node*;  // or also value_type*
        using const_pointer     = const Node*;  // or also value_type*
        using reference         = Node&;  // or also value_type&
        using const_reference   = const Node&;  // or also value_type&
        static constexpr uint32_t NumNeigbourNodes = 1 << Dim;

        const_iterator(const NodeTree<Dim>& tree) : myTree(tree) {}
        void setToBegin() 
        {
            nodes = std::stack<InternalNode>(); // Clear stack
            Node node;
            node.minCoord = myTree.minCoord;
            node.maxCoord = myTree.maxCoord;
            node.depth = 0;
            nodes.push(InternalNode{0, node, 0}); // Insert root
            while(!myTree.octreeData[nodes.top().nodeIdx].isLeaf())
            {
                // Get first child
                const uint32_t startIdx = myTree.octreeData[nodes.top().nodeIdx].getChildrenIndex();
                nodes.push(InternalNode { startIdx, 
                                         getNodeChildren(nodes.top().node, 0),
                                         0 });
            }
            nodes.top().node.controlPointsIdx = myTree.octreeData[nodes.top().nodeIdx].controlPointsIdx;
        }

        // TODO: make a more efficient end iterator
        void setToEnd() 
        {
            nodes.push(InternalNode { std::numeric_limits<uint32_t>::max(), Node(), std::numeric_limits<uint32_t>::max() });
        }
        
        const_reference operator*() const { return nodes.top().node; }
        const_pointer operator->() const { return &nodes.top().node; }

        const_iterator& operator++() 
        {
            nodes.pop();
            while(!myTree.octreeData[nodes.top().nodeIdx].isLeaf())
            {
                while(!nodes.empty() && ++nodes.top().currentChildIdx >= NumNeigbourNodes) nodes.pop();
                if(nodes.empty())
                {
                    nodes.push(InternalNode { std::numeric_limits<uint32_t>::max(), Node(), std::numeric_limits<uint32_t>::max() });
                    return *this;
                }

                const uint32_t idx = myTree.octreeData[nodes.top().nodeIdx].getChildrenIndex() + nodes.top().currentChildIdx;
                nodes.push(InternalNode { idx, 
                                         getNodeChildren(nodes.top().node, nodes.top().currentChildIdx),
                                         std::numeric_limits<uint32_t>::max() });
            }
            nodes.top().node.controlPointsIdx = myTree.octreeData[nodes.top().nodeIdx].controlPointsIdx;
            return *this;
        }

        const_iterator operator++(int) { const_iterator tmp = *this; ++(*this); return tmp; }

        friend bool operator==(const const_iterator& a, const const_iterator& b) { return a.nodes.top().nodeIdx == b.nodes.top().nodeIdx; }
        friend bool operator!=(const const_iterator& a, const const_iterator& b) { return a.nodes.top().nodeIdx != b.nodes.top().nodeIdx; }
    private:
        const NodeTree<Dim>& myTree;
        struct InternalNode
        {
            uint32_t nodeIdx;
            Node node;
            uint32_t currentChildIdx;
        };
        std::stack<InternalNode> nodes;

        inline Node getNodeChildren(const Node& node, uint32_t chIdx)
        {
            Node res;
            res.depth = node.depth + 1;
            vec center = 0.5f * (node.maxCoord + node.minCoord);
            for(uint32_t i=0; i < Dim; i++)
            {
                res.minCoord[i] = (chIdx & (1 << i)) ? center[i] : node.minCoord[i];
                res.maxCoord[i] = (chIdx & (1 << i)) ? node.maxCoord[i] : center[i];
            }
            return res;
        }
    };

    const_iterator begin() const 
    { 
        const_iterator it(*this);
        it.setToBegin();
        return std::move(it);
    }

    const_iterator end() const 
    { 
        const_iterator it(*this);
        it.setToEnd();
        return std::move(it);
    }
private:

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
        // if(nodeInfo.depth < maxDepth) // Create childrens
        {
            const uint32_t chIndex = octreeData.size();
            octreeData.resize(octreeData.size() + numNodes);
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

            for(uint32_t i=0; i < numNodes; i++)
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
            for(uint32_t i=0; i < NumVerticesPerNode; i++)
            {
                vec cp;
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
        vec center = 0.5f * (outNode->maxCoord + outNode->minCoord);
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
uint32_t NodeTree<Dim>::getAdjacentNodes(uint32_t vertId, std::array<NodeTree::Node, NumAdjacentNodes>& outNodes) const
{
    const float octreeLength = static_cast<float>(1 << maxDepth);
    const vec nodeSize = (maxCoord - minCoord) / octreeLength;

    const vec coord = getVertices()[vertId];
    constexpr uint32_t numNodes = NumAdjacentNodes;
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