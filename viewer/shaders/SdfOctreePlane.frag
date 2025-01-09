#version 430

#define USE_SDF_GRADIENT 1

#include SdfFunction
#include ColorPalettes

out vec4 fragColor;

in vec3 gridPosition;
in float distToCamera;

uniform float octreeValueRange = 1.0;

uniform float surfaceThickness = 3.5;
uniform float gridThickness = 0.01;
uniform float linesThickness = 2.5;

uniform float linesSpace = 0.03;

uniform bool printGrid = true;
uniform bool printIsolines = true;

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
    float dDist = max(length(vec2(dFdx(dist), dFdy(dist))), 0.0008);

    // Isosurface line
    float surfaceColorWeight = clamp(1.0 - pow(abs(dist) / (dDist * surfaceThickness), 8), 0.0, 1.0);
    
    // Grid lines
    float gridColorWeight = float(printGrid) * clamp(1.0 - pow(distToGrid * nodeRelativeLength / gridThickness, 8), 0.0, 1.0);

    // Isolevels lines
    float distToLevel = 0.5 - abs(fract(abs(dist) / linesSpace) - 0.5);
    float dDistToLevel = dDist / linesSpace;
    float linesColorWeight = float(printIsolines) * 0.5 * clamp(1.0 - pow(abs(distToLevel) / (dDistToLevel * linesThickness), 8), 0.0, 1.0);

    // Heat map color
    vec3 finalColor = getSdfPaletteColor(0.5 + 0.5 * dist / octreeValueRange);
    vec3 grad = 1.0 / 32.0 * getGradient(gridPosition);
    finalColor = (length(grad) > 1.0) ? vec3(1.0, 0.0, 0.0) : vec3(1-length(grad), 1.0, 1-length(grad));

    fragColor = vec4(mix(finalColor, vec3(0.0, 0.0, 0.0), max(max(surfaceColorWeight, gridColorWeight), linesColorWeight)), 1.0);
}