layout(std430, binding = 3) buffer sdfOctree
{
    uint sdfOctreeData[];
};

uniform vec3 planeNormal; // normalized
uniform vec3 startGridSize;

const uint isLeafMask = 1 << 31;
const uint childrenIndexMask = ~(1 << 31);

uint roundFloat(float a)
{
    return (a >= 0.5) ? 1 : 0;
}

// -- Function for linear interpolation --
float getDistance(vec3 point, out float distToGrid, out float nodeRelativeLength)
{
    vec3 fracPart = point * startGridSize;
    ivec3 arrayPos = ivec3(floor(fracPart));
    fracPart = fract(fracPart);

    int index = arrayPos.z * int(startGridSize.y * startGridSize.x) +
                arrayPos.y * int(startGridSize.x) +
                arrayPos.x;
    uint currentNode = sdfOctreeData[index];
    nodeRelativeLength = 1.0;

    while(!bool(currentNode & isLeafMask))
    {
        uint childIdx = (roundFloat(fracPart.z) << 2) + 
                        (roundFloat(fracPart.y) << 1) + 
                         roundFloat(fracPart.x);

        currentNode = sdfOctreeData[(currentNode & childrenIndexMask) + childIdx];
        fracPart = fract(2.0 * fracPart);
        nodeRelativeLength *= 0.5;
    }

    vec3 distToGridAxis = vec3(0.5) - abs(fracPart - vec3(0.5));
    distToGrid = min(min((abs(planeNormal.x) < 0.95) ? distToGridAxis.x : 1.0, 
                         (abs(planeNormal.y) < 0.95) ? distToGridAxis.y : 1.0),
                         (abs(planeNormal.z) < 0.95) ? distToGridAxis.z : 1.0);

    uint vIndex = currentNode & childrenIndexMask;

    float d00 = uintBitsToFloat(sdfOctreeData[vIndex]) * (1.0f - fracPart.x) +
                uintBitsToFloat(sdfOctreeData[vIndex + 1]) * fracPart.x;
    float d01 = uintBitsToFloat(sdfOctreeData[vIndex + 2]) * (1.0f - fracPart.x) +
                uintBitsToFloat(sdfOctreeData[vIndex + 3]) * fracPart.x;
    float d10 = uintBitsToFloat(sdfOctreeData[vIndex + 4]) * (1.0f - fracPart.x) +
                uintBitsToFloat(sdfOctreeData[vIndex + 5]) * fracPart.x;
    float d11 = uintBitsToFloat(sdfOctreeData[vIndex + 6]) * (1.0f - fracPart.x) +
                uintBitsToFloat(sdfOctreeData[vIndex + 7]) * fracPart.x;

    float d0 = d00 * (1.0f - fracPart.y) + d01 * fracPart.y;
    float d1 = d10 * (1.0f - fracPart.y) + d11 * fracPart.y;

    return d0 * (1.0f - fracPart.z) + d1 * fracPart.z;
}

#ifdef USE_SDF_GRADIENT
vec3 getGradient(vec3 point)
{
    vec3 fracPart = point * startGridSize;
    ivec3 arrayPos = ivec3(floor(fracPart));
    fracPart = fract(fracPart);

    int index = arrayPos.z * int(startGridSize.y * startGridSize.x) +
                arrayPos.y * int(startGridSize.x) +
                arrayPos.x;
    uint currentNode = sdfOctreeData[index];

    float nodeRelativeLength = 1.0;

    while(!bool(currentNode & isLeafMask))
    {
        uint childIdx = (roundFloat(fracPart.z) << 2) + 
                        (roundFloat(fracPart.y) << 1) + 
                         roundFloat(fracPart.x);

        currentNode = sdfOctreeData[(currentNode & childrenIndexMask) + childIdx];
        fracPart = fract(2.0 * fracPart);
        nodeRelativeLength *= 0.5;
    }

    uint vIndex = currentNode & childrenIndexMask;

    float gx = 0.0;
    {
        float d00 = uintBitsToFloat(sdfOctreeData[vIndex + 0]) * (1.0f - fracPart.y) +
                    uintBitsToFloat(sdfOctreeData[vIndex + 2]) * fracPart.y;
        float d01 = uintBitsToFloat(sdfOctreeData[vIndex + 1]) * (1.0f - fracPart.y) +
                    uintBitsToFloat(sdfOctreeData[vIndex + 3]) * fracPart.y;
        float d10 = uintBitsToFloat(sdfOctreeData[vIndex + 4]) * (1.0f - fracPart.y) +
                    uintBitsToFloat(sdfOctreeData[vIndex + 6]) * fracPart.y;
        float d11 = uintBitsToFloat(sdfOctreeData[vIndex + 5]) * (1.0f - fracPart.y) +
                    uintBitsToFloat(sdfOctreeData[vIndex + 7]) * fracPart.y;

        float d0 = d00 * (1.0f - fracPart.z) + d10 * fracPart.z;
        float d1 = d01 * (1.0f - fracPart.z) + d11 * fracPart.z;

        gx = d1 - d0;
    }

    float gy = 0.0;
    float gz = 0.0;
    {
        float d00 = uintBitsToFloat(sdfOctreeData[vIndex + 0]) * (1.0f - fracPart.x) +
                    uintBitsToFloat(sdfOctreeData[vIndex + 1]) * fracPart.x;
        float d01 = uintBitsToFloat(sdfOctreeData[vIndex + 2]) * (1.0f - fracPart.x) +
                    uintBitsToFloat(sdfOctreeData[vIndex + 3]) * fracPart.x;
        float d10 = uintBitsToFloat(sdfOctreeData[vIndex + 4]) * (1.0f - fracPart.x) +
                    uintBitsToFloat(sdfOctreeData[vIndex + 5]) * fracPart.x;
        float d11 = uintBitsToFloat(sdfOctreeData[vIndex + 6]) * (1.0f - fracPart.x) +
                    uintBitsToFloat(sdfOctreeData[vIndex + 7]) * fracPart.x;

        {
            float d0 = d00 * (1.0f - fracPart.z) + d10 * fracPart.z;
            float d1 = d01 * (1.0f - fracPart.z) + d11 * fracPart.z;

            gy = d1 - d0;
        }

        {
            float d0 = d00 * (1.0f - fracPart.y) + d01 * fracPart.y;
            float d1 = d10 * (1.0f - fracPart.y) + d11 * fracPart.y;

            gz = d1 - d0;
        }
    }

    // return normalize(vec3(gx, gy, gz));
    return vec3(gx/nodeRelativeLength, gy/nodeRelativeLength, gz/nodeRelativeLength);
}
#endif

#ifdef USE_VAR_OCTREE

layout(std430, binding = 4) buffer varOctree
{
    uint varOctreeData[];
};

float getVar(vec3 point, out float distToGrid, out float nodeRelativeLength)
{
    vec3 fracPart = point * startGridSize;
    ivec3 arrayPos = ivec3(floor(fracPart));
    fracPart = fract(fracPart);

    int index = arrayPos.z * int(startGridSize.y * startGridSize.x) +
                arrayPos.y * int(startGridSize.x) +
                arrayPos.x;
    uint currentNode = varOctreeData[index];
    nodeRelativeLength = 1.0;

    while(!bool(currentNode & isLeafMask))
    {
        uint childIdx = (roundFloat(fracPart.z) << 2) + 
                        (roundFloat(fracPart.y) << 1) + 
                         roundFloat(fracPart.x);

        currentNode = varOctreeData[(currentNode & childrenIndexMask) + childIdx];
        fracPart = fract(2.0 * fracPart);
        nodeRelativeLength *= 0.5;
    }

    vec3 distToGridAxis = vec3(0.5) - abs(fracPart - vec3(0.5));
    distToGrid = min(min((abs(planeNormal.x) < 0.95) ? distToGridAxis.x : 1.0, 
                         (abs(planeNormal.y) < 0.95) ? distToGridAxis.y : 1.0),
                         (abs(planeNormal.z) < 0.95) ? distToGridAxis.z : 1.0);

    uint vIndex = currentNode & childrenIndexMask;

    float d00 = uintBitsToFloat(varOctreeData[vIndex]) * (1.0f - fracPart.x) +
                uintBitsToFloat(varOctreeData[vIndex + 1]) * fracPart.x;
    float d01 = uintBitsToFloat(varOctreeData[vIndex + 2]) * (1.0f - fracPart.x) +
                uintBitsToFloat(varOctreeData[vIndex + 3]) * fracPart.x;
    float d10 = uintBitsToFloat(varOctreeData[vIndex + 4]) * (1.0f - fracPart.x) +
                uintBitsToFloat(varOctreeData[vIndex + 5]) * fracPart.x;
    float d11 = uintBitsToFloat(varOctreeData[vIndex + 6]) * (1.0f - fracPart.x) +
                uintBitsToFloat(varOctreeData[vIndex + 7]) * fracPart.x;

    float d0 = d00 * (1.0f - fracPart.y) + d01 * fracPart.y;
    float d1 = d10 * (1.0f - fracPart.y) + d11 * fracPart.y;

    return d0 * (1.0f - fracPart.z) + d1 * fracPart.z;
}

#endif