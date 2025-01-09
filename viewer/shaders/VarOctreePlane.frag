#version 430

#define USE_VAR_OCTREE 1

#include SdfFunction
#include ColorPalettes
#include Utils

in vec3 gridPosition;
in float distToCamera;

out vec4 fragColor;

uniform float varOctreeMaxValue;
uniform float varOctreeMinValue;

uniform int mode = 0;

#define M_PI 3.1415926535897932384626433832795

float cdf(float x, float mu, float std)
{
    return 0.5 * (1.0 + abs(erf((x - mu) / (sqrt(2.0) * std))));
};

void main()
{
    vec3 distToBox = abs(gridPosition - vec3(0.5));
    if(max(max(distToBox.x, distToBox.y), distToBox.z) > 0.5)
    {
        discard; return;
    }

    // Calculate needed derivatives
    float distToGrid = 0.0;
    float nodeRelativeLength;
    float dist = getDistance(gridPosition, distToGrid, nodeRelativeLength);
    float std = sqrt(abs(getVar(gridPosition, distToGrid, nodeRelativeLength)));

    float value;
    if(mode == 0)
    {
        float sqVarOctreeMaxValue = sqrt(abs(varOctreeMaxValue));
        float sqVarOctreeMinValue = sqrt(abs(varOctreeMinValue));
        value = (std - sqVarOctreeMinValue) / (sqVarOctreeMaxValue - sqVarOctreeMinValue);
    }
    else if(mode == 1)
    {
        value = 1.0 / sqrt(2.0 * M_PI * std) * exp(-0.5 * dist * dist / (std * std));
    }
    else
    {
        value = (dist > 0.0) ? 1.0 - cdf(0.0, dist, std) : cdf(0.0, dist, std);
    }

    vec3 finalColor = getViridisPaletteColor(value);
    fragColor = vec4(finalColor, 1.0);
}