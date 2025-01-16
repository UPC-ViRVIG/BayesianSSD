#ifndef INTERPOLATION_METHOD_H
#define INTERPOLATION_METHOD_H

#include <glm/glm.hpp>
#include <array>
#include "Vector.h"

template<uint32_t Dim>
struct MultivariateLinearInterpolation
{
    static constexpr uint32_t NumControlPoints = 1 << Dim;
    using vec = VStruct<Dim>::type;

    static void eval(vec np, std::array<float, NumControlPoints>& outWeights)
    {
        for(uint32_t i=0; i < NumControlPoints; i++)
        {
            outWeights[i] = 1.0f;
            for(uint32_t j=0; j < Dim; j++)
            {
                outWeights[i] *= (i & (1 << j)) ? np[j] : (1.0f - np[j]);
            }
        }
    }

    static void evalGrad(vec np, vec nodeSize, std::array<std::array<float, NumControlPoints>, Dim>& outWeights)
    {
        for(uint32_t d=0; d < Dim; d++)
        {
            for(uint32_t i=0; i < NumControlPoints; i++)
            {
                outWeights[d][i] = 1.0f;
                for(uint32_t j=0; j < Dim; j++)
                {
                    if(d == j) outWeights[d][i] *= (i & (1 << j)) ? 1.0f : -1.0f;
                    else outWeights[d][i] *= (i & (1 << j)) ? np[j] : (1.0f - np[j]);
                }

                outWeights[d][i] *= 1.0f / nodeSize[d];
            }
        }
    }

    static void evalGrad(vec np, std::array<std::array<float, NumControlPoints>, Dim>& outWeights)
    {
        for(uint32_t d=0; d < Dim; d++)
        {
            for(uint32_t i=0; i < NumControlPoints; i++)
            {
                outWeights[d][i] = 1.0f;
                for(uint32_t j=0; j < Dim; j++)
                {
                    if(d == j) outWeights[d][i] *= (i & (1 << j)) ? 1.0f : -1.0f;
                    else outWeights[d][i] *= (i & (1 << j)) ? np[j] : (1.0f - np[j]);
                }
            }
        }
    }

    static constexpr uint32_t NumSecondGrad = (Dim == 2) ? 1 : 3;
    static void evalSecondGradInteg(vec nodeSize, std::array<std::array<float, NumControlPoints>, NumSecondGrad>& outWeights) {}
};

template<>
static inline void MultivariateLinearInterpolation<2>::evalSecondGradInteg(vec nodeSize, std::array<std::array<float, NumControlPoints>, NumSecondGrad>& outWeights) 
{
    const float invSizeSq = 1.0f / (nodeSize[0] * nodeSize[0]);
    outWeights[0][0] = invSizeSq;
    outWeights[0][1] = -invSizeSq;
    outWeights[0][2] = -invSizeSq;
    outWeights[0][3] = invSizeSq;
}

template<>
static inline void MultivariateLinearInterpolation<3>::evalSecondGradInteg(vec nodeSize, std::array<std::array<float, NumControlPoints>, NumSecondGrad>& outWeights) 
{
    const float invSizeSq = 1.0f / (nodeSize[0] * nodeSize[0]);
    //dx dy
    outWeights[0][0] = invSizeSq;
    outWeights[0][1] = -invSizeSq;
    outWeights[0][2] = -invSizeSq;
    outWeights[0][3] = invSizeSq;
    outWeights[0][4] = invSizeSq;
    outWeights[0][5] = -invSizeSq;
    outWeights[0][6] = -invSizeSq;
    outWeights[0][7] = invSizeSq;

    //dx dz
    outWeights[1][0] = invSizeSq;
    outWeights[1][1] = -invSizeSq;
    outWeights[1][2] = invSizeSq;
    outWeights[1][3] = -invSizeSq;
    outWeights[1][4] = -invSizeSq;
    outWeights[1][5] = invSizeSq;
    outWeights[1][6] = -invSizeSq;
    outWeights[1][7] = invSizeSq;

    //dy dz
    outWeights[2][0] = invSizeSq;
    outWeights[2][1] = invSizeSq;
    outWeights[2][2] = -invSizeSq;
    outWeights[2][3] = -invSizeSq;
    outWeights[2][4] = -invSizeSq;
    outWeights[2][5] = -invSizeSq;
    outWeights[2][6] = invSizeSq;
    outWeights[2][7] = invSizeSq;
    
}

struct BicubicInterpolation
{
    static constexpr uint32_t NumControlPoints = 4;
    static constexpr uint32_t NumBasis = 4;

    static void normalizeControlPointValues(glm::vec2 nodeSize, std::array<float, 4>& values, float factor=1.0f)
    {
        values[0] *= factor;
        values[1] *= nodeSize.x * factor;
        values[2] *= nodeSize.y * factor;
        values[3] *= nodeSize.x * nodeSize.y * factor;
    }

    static void eval(glm::vec2 np, glm::vec2 nodeSize, std::array<std::array<float, 4>, 4>& values)
    {
        auto pow = [](float val, uint32_t n)
        {
            float res = 1.0f;
            for(uint32_t i=0; i < n; i++) res *= val;
            return res;
        };

        float x = np.x; float y = np.y;
        values[0][0] = 4.0*pow(x, 3)*pow(y, 3) - 6.0*pow(x, 3)*pow(y, 2) + 2.0*pow(x, 3) - 6.0*pow(x, 2)*pow(y, 3) + 9.0*pow(x, 2)*pow(y, 2) - 3.0*pow(x, 2) + 2.0*pow(y, 3) - 3.0*pow(y, 2) + 1.0;
        values[0][1] = 2.0*pow(x, 3)*pow(y, 3) - 3.0*pow(x, 3)*pow(y, 2) + 1.0*pow(x, 3) - 4.0*pow(x, 2)*pow(y, 3) + 6.0*pow(x, 2)*pow(y, 2) - 2.0*pow(x, 2) + 2.0*x*pow(y, 3) - 3.0*x*pow(y, 2) + 1.0*x;
        values[0][2] = 2.0*pow(x, 3)*pow(y, 3) - 4.0*pow(x, 3)*pow(y, 2) + 2.0*pow(x, 3)*y - 3.0*pow(x, 2)*pow(y, 3) + 6.0*pow(x, 2)*pow(y, 2) - 3.0*pow(x, 2)*y + 1.0*pow(y, 3) - 2.0*pow(y, 2) + 1.0*y;
        values[0][3] = 1.0*pow(x, 3)*pow(y, 3) - 2.0*pow(x, 3)*pow(y, 2) + 1.0*pow(x, 3)*y - 2.0*pow(x, 2)*pow(y, 3) + 4.0*pow(x, 2)*pow(y, 2) - 2.0*pow(x, 2)*y + 1.0*x*pow(y, 3) - 2.0*x*pow(y, 2) + 1.0*x*y;
        values[1][0] = -4.0*pow(x, 3)*pow(y, 3) + 6.0*pow(x, 3)*pow(y, 2) - 2.0*pow(x, 3) + 6.0*pow(x, 2)*pow(y, 3) - 9.0*pow(x, 2)*pow(y, 2) + 3.0*pow(x, 2);
        values[1][1] = 2.0*pow(x, 3)*pow(y, 3) - 3.0*pow(x, 3)*pow(y, 2) + 1.0*pow(x, 3) - 2.0*pow(x, 2)*pow(y, 3) + 3.0*pow(x, 2)*pow(y, 2) - 1.0*pow(x, 2);
        values[1][2] = -2.0*pow(x, 3)*pow(y, 3) + 4.0*pow(x, 3)*pow(y, 2) - 2.0*pow(x, 3)*y + 3.0*pow(x, 2)*pow(y, 3) - 6.0*pow(x, 2)*pow(y, 2) + 3.0*pow(x, 2)*y;
        values[1][3] = 1.0*pow(x, 3)*pow(y, 3) - 2.0*pow(x, 3)*pow(y, 2) + 1.0*pow(x, 3)*y - 1.0*pow(x, 2)*pow(y, 3) + 2.0*pow(x, 2)*pow(y, 2) - 1.0*pow(x, 2)*y;
        values[2][0] = -4.0*pow(x, 3)*pow(y, 3) + 6.0*pow(x, 3)*pow(y, 2) + 6.0*pow(x, 2)*pow(y, 3) - 9.0*pow(x, 2)*pow(y, 2) - 2.0*pow(y, 3) + 3.0*pow(y, 2);
        values[2][1] = -2.0*pow(x, 3)*pow(y, 3) + 3.0*pow(x, 3)*pow(y, 2) + 4.0*pow(x, 2)*pow(y, 3) - 6.0*pow(x, 2)*pow(y, 2) - 2.0*x*pow(y, 3) + 3.0*x*pow(y, 2);
        values[2][2] = 2.0*pow(x, 3)*pow(y, 3) - 2.0*pow(x, 3)*pow(y, 2) - 3.0*pow(x, 2)*pow(y, 3) + 3.0*pow(x, 2)*pow(y, 2) + 1.0*pow(y, 3) - 1.0*pow(y, 2);
        values[2][3] = 1.0*pow(x, 3)*pow(y, 3) - 1.0*pow(x, 3)*pow(y, 2) - 2.0*pow(x, 2)*pow(y, 3) + 2.0*pow(x, 2)*pow(y, 2) + 1.0*x*pow(y, 3) - 1.0*x*pow(y, 2);
        values[3][0] = 4.0*pow(x, 3)*pow(y, 3) - 6.0*pow(x, 3)*pow(y, 2) - 6.0*pow(x, 2)*pow(y, 3) + 9.0*pow(x, 2)*pow(y, 2);
        values[3][1] = -2.0*pow(x, 3)*pow(y, 3) + 3.0*pow(x, 3)*pow(y, 2) + 2.0*pow(x, 2)*pow(y, 3) - 3.0*pow(x, 2)*pow(y, 2);
        values[3][2] = -2.0*pow(x, 3)*pow(y, 3) + 2.0*pow(x, 3)*pow(y, 2) + 3.0*pow(x, 2)*pow(y, 3) - 3.0*pow(x, 2)*pow(y, 2);
        values[3][3] = 1.0*pow(x, 3)*pow(y, 3) - 1.0*pow(x, 3)*pow(y, 2) - 1.0*pow(x, 2)*pow(y, 3) + 1.0*pow(x, 2)*pow(y, 2);

        for(auto& cpv : values) normalizeControlPointValues(nodeSize, cpv);
    }

    static void evalGrad(glm::vec2 np, glm::vec2 nodeSize, std::array<std::array<std::array<float, 4>, 4>, 2>& values)
    {
        auto pow = [](float val, uint32_t n)
        {
            float res = 1.0f;
            for(uint32_t i=0; i < n; i++) res *= val;
            return res;
        };

        float x = np.x; float y = np.y;

        // dX
        values[0][0][0] = 12.0*pow(x, 2)*pow(y, 3) - 18.0*pow(x, 2)*pow(y, 2) + 6.0*pow(x, 2) - 12.0*x*pow(y, 3) + 18.0*x*pow(y, 2) - 6.0*x;
        values[0][0][1] = 6.0*pow(x, 2)*pow(y, 3) - 9.0*pow(x, 2)*pow(y, 2) + 3.0*pow(x, 2) - 8.0*x*pow(y, 3) + 12.0*x*pow(y, 2) - 4.0*x + 2.0*pow(y, 3) - 3.0*pow(y, 2) + 1.0;
        values[0][0][2] = 6.0*pow(x, 2)*pow(y, 3) - 12.0*pow(x, 2)*pow(y, 2) + 6.0*pow(x, 2)*y - 6.0*x*pow(y, 3) + 12.0*x*pow(y, 2) - 6.0*x*y;
        values[0][0][3] = 3.0*pow(x, 2)*pow(y, 3) - 6.0*pow(x, 2)*pow(y, 2) + 3.0*pow(x, 2)*y - 4.0*x*pow(y, 3) + 8.0*x*pow(y, 2) - 4.0*x*y + 1.0*pow(y, 3) - 2.0*pow(y, 2) + 1.0*y;
        values[0][1][0] = -12.0*pow(x, 2)*pow(y, 3) + 18.0*pow(x, 2)*pow(y, 2) - 6.0*pow(x, 2) + 12.0*x*pow(y, 3) - 18.0*x*pow(y, 2) + 6.0*x;
        values[0][1][1] = 6.0*pow(x, 2)*pow(y, 3) - 9.0*pow(x, 2)*pow(y, 2) + 3.0*pow(x, 2) - 4.0*x*pow(y, 3) + 6.0*x*pow(y, 2) - 2.0*x;
        values[0][1][2] = -6.0*pow(x, 2)*pow(y, 3) + 12.0*pow(x, 2)*pow(y, 2) - 6.0*pow(x, 2)*y + 6.0*x*pow(y, 3) - 12.0*x*pow(y, 2) + 6.0*x*y;
        values[0][1][3] = 3.0*pow(x, 2)*pow(y, 3) - 6.0*pow(x, 2)*pow(y, 2) + 3.0*pow(x, 2)*y - 2.0*x*pow(y, 3) + 4.0*x*pow(y, 2) - 2.0*x*y;
        values[0][2][0] = -12.0*pow(x, 2)*pow(y, 3) + 18.0*pow(x, 2)*pow(y, 2) + 12.0*x*pow(y, 3) - 18.0*x*pow(y, 2);
        values[0][2][1] = -6.0*pow(x, 2)*pow(y, 3) + 9.0*pow(x, 2)*pow(y, 2) + 8.0*x*pow(y, 3) - 12.0*x*pow(y, 2) - 2.0*pow(y, 3) + 3.0*pow(y, 2);
        values[0][2][2] = 6.0*pow(x, 2)*pow(y, 3) - 6.0*pow(x, 2)*pow(y, 2) - 6.0*x*pow(y, 3) + 6.0*x*pow(y, 2);
        values[0][2][3] = 3.0*pow(x, 2)*pow(y, 3) - 3.0*pow(x, 2)*pow(y, 2) - 4.0*x*pow(y, 3) + 4.0*x*pow(y, 2) + 1.0*pow(y, 3) - 1.0*pow(y, 2);
        values[0][3][0] = 12.0*pow(x, 2)*pow(y, 3) - 18.0*pow(x, 2)*pow(y, 2) - 12.0*x*pow(y, 3) + 18.0*x*pow(y, 2);
        values[0][3][1] = -6.0*pow(x, 2)*pow(y, 3) + 9.0*pow(x, 2)*pow(y, 2) + 4.0*x*pow(y, 3) - 6.0*x*pow(y, 2);
        values[0][3][2] = -6.0*pow(x, 2)*pow(y, 3) + 6.0*pow(x, 2)*pow(y, 2) + 6.0*x*pow(y, 3) - 6.0*x*pow(y, 2);
        values[0][3][3] = 3.0*pow(x, 2)*pow(y, 3) - 3.0*pow(x, 2)*pow(y, 2) - 2.0*x*pow(y, 3) + 2.0*x*pow(y, 2);

        for(auto& cpv : values[0]) normalizeControlPointValues(nodeSize, cpv, 1.0f / nodeSize.x);

        // dY
        values[1][0][0] = 12.0*pow(x, 3)*pow(y, 2) - 12.0*pow(x, 3)*y - 18.0*pow(x, 2)*pow(y, 2) + 18.0*pow(x, 2)*y + 6.0*pow(y, 2) - 6.0*y;
        values[1][0][1] = 6.0*pow(x, 3)*pow(y, 2) - 6.0*pow(x, 3)*y - 12.0*pow(x, 2)*pow(y, 2) + 12.0*pow(x, 2)*y + 6.0*x*pow(y, 2) - 6.0*x*y;
        values[1][0][2] = 6.0*pow(x, 3)*pow(y, 2) - 8.0*pow(x, 3)*y + 2.0*pow(x, 3) - 9.0*pow(x, 2)*pow(y, 2) + 12.0*pow(x, 2)*y - 3.0*pow(x, 2) + 3.0*pow(y, 2) - 4.0*y + 1.0;
        values[1][0][3] = 3.0*pow(x, 3)*pow(y, 2) - 4.0*pow(x, 3)*y + 1.0*pow(x, 3) - 6.0*pow(x, 2)*pow(y, 2) + 8.0*pow(x, 2)*y - 2.0*pow(x, 2) + 3.0*x*pow(y, 2) - 4.0*x*y + 1.0*x;
        values[1][1][0] = -12.0*pow(x, 3)*pow(y, 2) + 12.0*pow(x, 3)*y + 18.0*pow(x, 2)*pow(y, 2) - 18.0*pow(x, 2)*y;
        values[1][1][1] = 6.0*pow(x, 3)*pow(y, 2) - 6.0*pow(x, 3)*y - 6.0*pow(x, 2)*pow(y, 2) + 6.0*pow(x, 2)*y;
        values[1][1][2] = -6.0*pow(x, 3)*pow(y, 2) + 8.0*pow(x, 3)*y - 2.0*pow(x, 3) + 9.0*pow(x, 2)*pow(y, 2) - 12.0*pow(x, 2)*y + 3.0*pow(x, 2);
        values[1][1][3] = 3.0*pow(x, 3)*pow(y, 2) - 4.0*pow(x, 3)*y + 1.0*pow(x, 3) - 3.0*pow(x, 2)*pow(y, 2) + 4.0*pow(x, 2)*y - 1.0*pow(x, 2);
        values[1][2][0] = -12.0*pow(x, 3)*pow(y, 2) + 12.0*pow(x, 3)*y + 18.0*pow(x, 2)*pow(y, 2) - 18.0*pow(x, 2)*y - 6.0*pow(y, 2) + 6.0*y;
        values[1][2][1] = -6.0*pow(x, 3)*pow(y, 2) + 6.0*pow(x, 3)*y + 12.0*pow(x, 2)*pow(y, 2) - 12.0*pow(x, 2)*y - 6.0*x*pow(y, 2) + 6.0*x*y;
        values[1][2][2] = 6.0*pow(x, 3)*pow(y, 2) - 4.0*pow(x, 3)*y - 9.0*pow(x, 2)*pow(y, 2) + 6.0*pow(x, 2)*y + 3.0*pow(y, 2) - 2.0*y;
        values[1][2][3] = 3.0*pow(x, 3)*pow(y, 2) - 2.0*pow(x, 3)*y - 6.0*pow(x, 2)*pow(y, 2) + 4.0*pow(x, 2)*y + 3.0*x*pow(y, 2) - 2.0*x*y;
        values[1][3][0] = 12.0*pow(x, 3)*pow(y, 2) - 12.0*pow(x, 3)*y - 18.0*pow(x, 2)*pow(y, 2) + 18.0*pow(x, 2)*y;
        values[1][3][1] = -6.0*pow(x, 3)*pow(y, 2) + 6.0*pow(x, 3)*y + 6.0*pow(x, 2)*pow(y, 2) - 6.0*pow(x, 2)*y;
        values[1][3][2] = -6.0*pow(x, 3)*pow(y, 2) + 4.0*pow(x, 3)*y + 9.0*pow(x, 2)*pow(y, 2) - 6.0*pow(x, 2)*y;
        values[1][3][3] = 3.0*pow(x, 3)*pow(y, 2) - 2.0*pow(x, 3)*y - 3.0*pow(x, 2)*pow(y, 2) + 2.0*pow(x, 2)*y;

        for(auto& cpv : values[1]) normalizeControlPointValues(nodeSize, cpv, 1.0f / nodeSize.x);
    }

    static void evalBasisValues(glm::vec2 np, glm::vec2 nodeSize, std::array<std::array<std::array<float, 4>, 4>, 4>& outValues)
    {
        eval(np, nodeSize, outValues[0]);
        std::array<std::array<std::array<float, 4>, 4>, 2>* gradients = 
            reinterpret_cast<std::array<std::array<std::array<float, 4>, 4>, 2>*>(&(outValues[1]));
        evalGrad(np, nodeSize, *gradients);

        auto pow = [](float val, uint32_t n)
        {
            float res = 1.0f;
            for(uint32_t i=0; i < n; i++) res *= val;
            return res;
        };

        float x = np.x; float y = np.y;

        // Compute dX dY
        auto& values = outValues[3];

        values[0][0] = 36.0*pow(x, 2)*pow(y, 2) - 36.0*pow(x, 2)*y - 36.0*x*pow(y, 2) + 36.0*x*y;
        values[0][1] = 18.0*pow(x, 2)*pow(y, 2) - 18.0*pow(x, 2)*y - 24.0*x*pow(y, 2) + 24.0*x*y + 6.0*pow(y, 2) - 6.0*y;
        values[0][2] = 18.0*pow(x, 2)*pow(y, 2) - 24.0*pow(x, 2)*y + 6.0*pow(x, 2) - 18.0*x*pow(y, 2) + 24.0*x*y - 6.0*x;
        values[0][3] = 9.0*pow(x, 2)*pow(y, 2) - 12.0*pow(x, 2)*y + 3.0*pow(x, 2) - 12.0*x*pow(y, 2) + 16.0*x*y - 4.0*x + 3.0*pow(y, 2) - 4.0*y + 1.0;
        values[1][0] = -36.0*pow(x, 2)*pow(y, 2) + 36.0*pow(x, 2)*y + 36.0*x*pow(y, 2) - 36.0*x*y;
        values[1][1] = 18.0*pow(x, 2)*pow(y, 2) - 18.0*pow(x, 2)*y - 12.0*x*pow(y, 2) + 12.0*x*y;
        values[1][2] = -18.0*pow(x, 2)*pow(y, 2) + 24.0*pow(x, 2)*y - 6.0*pow(x, 2) + 18.0*x*pow(y, 2) - 24.0*x*y + 6.0*x;
        values[1][3] = 9.0*pow(x, 2)*pow(y, 2) - 12.0*pow(x, 2)*y + 3.0*pow(x, 2) - 6.0*x*pow(y, 2) + 8.0*x*y - 2.0*x;
        values[2][0] = -36.0*pow(x, 2)*pow(y, 2) + 36.0*pow(x, 2)*y + 36.0*x*pow(y, 2) - 36.0*x*y;
        values[2][1] = -18.0*pow(x, 2)*pow(y, 2) + 18.0*pow(x, 2)*y + 24.0*x*pow(y, 2) - 24.0*x*y - 6.0*pow(y, 2) + 6.0*y;
        values[2][2] = 18.0*pow(x, 2)*pow(y, 2) - 12.0*pow(x, 2)*y - 18.0*x*pow(y, 2) + 12.0*x*y;
        values[2][3] = 9.0*pow(x, 2)*pow(y, 2) - 6.0*pow(x, 2)*y - 12.0*x*pow(y, 2) + 8.0*x*y + 3.0*pow(y, 2) - 2.0*y;
        values[3][0] = 36.0*pow(x, 2)*pow(y, 2) - 36.0*pow(x, 2)*y - 36.0*x*pow(y, 2) + 36.0*x*y;
        values[3][1] = -18.0*pow(x, 2)*pow(y, 2) + 18.0*pow(x, 2)*y + 12.0*x*pow(y, 2) - 12.0*x*y;
        values[3][2] = -18.0*pow(x, 2)*pow(y, 2) + 12.0*pow(x, 2)*y + 18.0*x*pow(y, 2) - 12.0*x*y;
        values[3][3] = 9.0*pow(x, 2)*pow(y, 2) - 6.0*pow(x, 2)*y - 6.0*x*pow(y, 2) + 4.0*x*y;

        for(auto& cpv : values) normalizeControlPointValues(nodeSize, cpv, 1.0f / (nodeSize.x * nodeSize.y));
    }

    // Node size must have the same size in all axis
    static void integrateLaplacian(glm::vec2 nodeSize, std::array<std::array<std::array<std::array<float, 4>, 4>, 4>, 4>& values)
    {
        auto pow = [](float val, uint32_t n)
        {
            float res = 1.0f;
            for(uint32_t i=0; i < n; i++) res *= val;
            return res;
        };

        const float ns = nodeSize.x;

        values[0][0][0][0] = 23.58857142857142/pow(ns, 2);
        values[0][0][0][1] = 6.1942857142857157/ns;
        values[0][0][0][2] = 6.1942857142857122/ns;
        values[0][0][0][3] = 1.2971428571428536;
        values[0][0][1][0] = -11.588571428571424/pow(ns, 2);
        values[0][0][1][1] = 4.1942857142857193/ns;
        values[0][0][1][2] = -0.19428571428571217/ns;
        values[0][0][1][3] = 0.2971428571428536;
        values[0][0][2][0] = -11.588571428571413/pow(ns, 2);
        values[0][0][2][1] = -0.19428571428571573/ns;
        values[0][0][2][2] = 4.1942857142857051/ns;
        values[0][0][2][3] = 0.29714285714285776;
        values[0][0][3][0] = -0.41142857142857636/pow(ns, 2);
        values[0][0][3][1] = 1.8057142857142843/ns;
        values[0][0][3][2] = 1.8057142857142878/ns;
        values[0][0][3][3] = -0.70285714285714196;
        values[0][1][0][0] = 6.1942857142857157/ns;
        values[0][1][0][1] = 3.8400000000000016;
        values[0][1][0][2] = 1.2971428571428536;
        values[0][1][0][3] = 0.58666666666666623*ns;
        values[0][1][1][0] = -4.1942857142857193/ns;
        values[0][1][1][1] = 1.1542857142857148;
        values[0][1][1][2] = -0.2971428571428536;
        values[0][1][1][3] = 0.1104761904761915*ns;
        values[0][1][2][0] = -0.19428571428571573/ns;
        values[0][1][2][1] = 0.16000000000000147;
        values[0][1][2][2] = 0.29714285714285776;
        values[0][1][2][3] = -0.079999999999999849*ns;
        values[0][1][3][0] = -1.8057142857142843/ns;
        values[0][1][3][1] = 0.8457142857142852;
        values[0][1][3][2] = 0.70285714285714196;
        values[0][1][3][3] = -0.22285714285714309*ns;
        values[0][2][0][0] = 6.1942857142857122/ns;
        values[0][2][0][1] = 1.2971428571428536;
        values[0][2][0][2] = 3.8399999999999892;
        values[0][2][0][3] = 0.58666666666666956*ns;
        values[0][2][1][0] = -0.19428571428571217/ns;
        values[0][2][1][1] = 0.2971428571428536;
        values[0][2][1][2] = 0.16000000000001791;
        values[0][2][1][3] = -0.07999999999999563*ns;
        values[0][2][2][0] = -4.1942857142857051/ns;
        values[0][2][2][1] = -0.29714285714285182;
        values[0][2][2][2] = 1.154285714285713;
        values[0][2][2][3] = 0.11047619047618884*ns;
        values[0][2][3][0] = -1.8057142857142878/ns;
        values[0][2][3][1] = 0.7028571428571464;
        values[0][2][3][2] = 0.8457142857142852;
        values[0][2][3][3] = -0.22285714285714442*ns;
        values[0][3][0][0] = 1.2971428571428536;
        values[0][3][0][1] = 0.58666666666666623*ns;
        values[0][3][0][2] = 0.58666666666666956*ns;
        values[0][3][0][3] = 0.22349206349206252*pow(ns, 2);
        values[0][3][1][0] = -0.2971428571428536;
        values[0][3][1][1] = 0.1104761904761915*ns;
        values[0][3][1][2] = 0.07999999999999563*ns;
        values[0][3][1][3] = -0.036825396825396983*pow(ns, 2);
        values[0][3][2][0] = -0.29714285714285182;
        values[0][3][2][1] = 0.080000000000000959*ns;
        values[0][3][2][2] = 0.11047619047618884*ns;
        values[0][3][2][3] = -0.036825396825398204*pow(ns, 2);
        values[0][3][3][0] = -0.7028571428571464;
        values[0][3][3][1] = 0.22285714285714264*ns;
        values[0][3][3][2] = 0.22285714285714442*ns;
        values[0][3][3][3] = -0.052698412698412467*pow(ns, 2);
        values[1][0][0][0] = -11.588571428571424/pow(ns, 2);
        values[1][0][0][1] = -4.1942857142857193/ns;
        values[1][0][0][2] = -0.19428571428571217/ns;
        values[1][0][0][3] = -0.2971428571428536;
        values[1][0][1][0] = 23.58857142857142/pow(ns, 2);
        values[1][0][1][1] = -6.1942857142857157/ns;
        values[1][0][1][2] = 6.1942857142857122/ns;
        values[1][0][1][3] = -1.2971428571428536;
        values[1][0][2][0] = -0.41142857142857636/pow(ns, 2);
        values[1][0][2][1] = -1.8057142857142843/ns;
        values[1][0][2][2] = 1.8057142857142878/ns;
        values[1][0][2][3] = 0.70285714285714196;
        values[1][0][3][0] = -11.588571428571413/pow(ns, 2);
        values[1][0][3][1] = 0.19428571428571573/ns;
        values[1][0][3][2] = 4.1942857142857051/ns;
        values[1][0][3][3] = -0.29714285714285776;
        values[1][1][0][0] = 4.1942857142857193/ns;
        values[1][1][0][1] = 1.1542857142857148;
        values[1][1][0][2] = 0.2971428571428536;
        values[1][1][0][3] = 0.1104761904761915*ns;
        values[1][1][1][0] = -6.1942857142857157/ns;
        values[1][1][1][1] = 3.8400000000000016;
        values[1][1][1][2] = -1.2971428571428536;
        values[1][1][1][3] = 0.58666666666666623*ns;
        values[1][1][2][0] = 1.8057142857142843/ns;
        values[1][1][2][1] = 0.8457142857142852;
        values[1][1][2][2] = -0.70285714285714196;
        values[1][1][2][3] = -0.22285714285714309*ns;
        values[1][1][3][0] = 0.19428571428571573/ns;
        values[1][1][3][1] = 0.16000000000000147;
        values[1][1][3][2] = -0.29714285714285776;
        values[1][1][3][3] = -0.079999999999999849*ns;
        values[1][2][0][0] = -0.19428571428571217/ns;
        values[1][2][0][1] = -0.2971428571428536;
        values[1][2][0][2] = 0.16000000000001791;
        values[1][2][0][3] = 0.07999999999999563*ns;
        values[1][2][1][0] = 6.1942857142857122/ns;
        values[1][2][1][1] = -1.2971428571428536;
        values[1][2][1][2] = 3.8399999999999892;
        values[1][2][1][3] = -0.58666666666666956*ns;
        values[1][2][2][0] = -1.8057142857142878/ns;
        values[1][2][2][1] = -0.7028571428571464;
        values[1][2][2][2] = 0.8457142857142852;
        values[1][2][2][3] = 0.22285714285714442*ns;
        values[1][2][3][0] = -4.1942857142857051/ns;
        values[1][2][3][1] = 0.29714285714285182;
        values[1][2][3][2] = 1.154285714285713;
        values[1][2][3][3] = -0.11047619047618884*ns;
        values[1][3][0][0] = 0.2971428571428536;
        values[1][3][0][1] = 0.1104761904761915*ns;
        values[1][3][0][2] = -0.07999999999999563*ns;
        values[1][3][0][3] = -0.036825396825396983*pow(ns, 2);
        values[1][3][1][0] = -1.2971428571428536;
        values[1][3][1][1] = 0.58666666666666623*ns;
        values[1][3][1][2] = -0.58666666666666956*ns;
        values[1][3][1][3] = 0.22349206349206252*pow(ns, 2);
        values[1][3][2][0] = 0.7028571428571464;
        values[1][3][2][1] = 0.22285714285714264*ns;
        values[1][3][2][2] = -0.22285714285714442*ns;
        values[1][3][2][3] = -0.052698412698412467*pow(ns, 2);
        values[1][3][3][0] = 0.29714285714285182;
        values[1][3][3][1] = 0.080000000000000959*ns;
        values[1][3][3][2] = -0.11047619047618884*ns;
        values[1][3][3][3] = -0.036825396825398204*pow(ns, 2);
        values[2][0][0][0] = -11.588571428571413/pow(ns, 2);
        values[2][0][0][1] = -0.19428571428571573/ns;
        values[2][0][0][2] = -4.1942857142857051/ns;
        values[2][0][0][3] = -0.29714285714285182;
        values[2][0][1][0] = -0.41142857142857636/pow(ns, 2);
        values[2][0][1][1] = 1.8057142857142843/ns;
        values[2][0][1][2] = -1.8057142857142878/ns;
        values[2][0][1][3] = 0.7028571428571464;
        values[2][0][2][0] = 23.588571428571413/pow(ns, 2);
        values[2][0][2][1] = 6.1942857142857157/ns;
        values[2][0][2][2] = -6.1942857142857086/ns;
        values[2][0][2][3] = -1.2971428571428572;
        values[2][0][3][0] = -11.588571428571427/pow(ns, 2);
        values[2][0][3][1] = 4.1942857142857157/ns;
        values[2][0][3][2] = 0.19428571428571217/ns;
        values[2][0][3][3] = -0.29714285714285799;
        values[2][1][0][0] = -0.19428571428571573/ns;
        values[2][1][0][1] = 0.16000000000000147;
        values[2][1][0][2] = -0.29714285714285182;
        values[2][1][0][3] = 0.080000000000000959*ns;
        values[2][1][1][0] = -1.8057142857142843/ns;
        values[2][1][1][1] = 0.8457142857142852;
        values[2][1][1][2] = -0.7028571428571464;
        values[2][1][1][3] = 0.22285714285714264*ns;
        values[2][1][2][0] = 6.1942857142857157/ns;
        values[2][1][2][1] = 3.8399999999999981;
        values[2][1][2][2] = -1.2971428571428572;
        values[2][1][2][3] = -0.58666666666666778*ns;
        values[2][1][3][0] = -4.1942857142857157/ns;
        values[2][1][3][1] = 1.1542857142857148;
        values[2][1][3][2] = 0.29714285714285799;
        values[2][1][3][3] = -0.11047619047619062*ns;
        values[2][2][0][0] = 4.1942857142857051/ns;
        values[2][2][0][1] = 0.29714285714285776;
        values[2][2][0][2] = 1.154285714285713;
        values[2][2][0][3] = 0.11047619047618884*ns;
        values[2][2][1][0] = 1.8057142857142878/ns;
        values[2][2][1][1] = -0.70285714285714196;
        values[2][2][1][2] = 0.8457142857142852;
        values[2][2][1][3] = -0.22285714285714442*ns;
        values[2][2][2][0] = -6.1942857142857086/ns;
        values[2][2][2][1] = -1.2971428571428572;
        values[2][2][2][2] = 3.8399999999999999;
        values[2][2][2][3] = 0.58666666666666667*ns;
        values[2][2][3][0] = 0.19428571428571217/ns;
        values[2][2][3][1] = -0.29714285714285799;
        values[2][2][3][2] = 0.16000000000000059;
        values[2][2][3][3] = -0.079999999999999932*ns;
        values[2][3][0][0] = 0.29714285714285776;
        values[2][3][0][1] = -0.079999999999999849*ns;
        values[2][3][0][2] = 0.11047619047618884*ns;
        values[2][3][0][3] = -0.036825396825398204*pow(ns, 2);
        values[2][3][1][0] = 0.70285714285714196;
        values[2][3][1][1] = -0.22285714285714309*ns;
        values[2][3][1][2] = 0.22285714285714442*ns;
        values[2][3][1][3] = -0.052698412698412467*pow(ns, 2);
        values[2][3][2][0] = -1.2971428571428572;
        values[2][3][2][1] = -0.58666666666666778*ns;
        values[2][3][2][2] = 0.58666666666666667*ns;
        values[2][3][2][3] = 0.22349206349206341*pow(ns, 2);
        values[2][3][3][0] = 0.29714285714285799;
        values[2][3][3][1] = -0.11047619047619062*ns;
        values[2][3][3][2] = 0.079999999999999932*ns;
        values[2][3][3][3] = -0.036825396825396706*pow(ns, 2);
        values[3][0][0][0] = -0.41142857142857636/pow(ns, 2);
        values[3][0][0][1] = -1.8057142857142843/ns;
        values[3][0][0][2] = -1.8057142857142878/ns;
        values[3][0][0][3] = -0.7028571428571464;
        values[3][0][1][0] = -11.588571428571413/pow(ns, 2);
        values[3][0][1][1] = 0.19428571428571573/ns;
        values[3][0][1][2] = -4.1942857142857051/ns;
        values[3][0][1][3] = 0.29714285714285182;
        values[3][0][2][0] = -11.588571428571427/pow(ns, 2);
        values[3][0][2][1] = -4.1942857142857157/ns;
        values[3][0][2][2] = 0.19428571428571217/ns;
        values[3][0][2][3] = 0.29714285714285799;
        values[3][0][3][0] = 23.588571428571413/pow(ns, 2);
        values[3][0][3][1] = -6.1942857142857157/ns;
        values[3][0][3][2] = -6.1942857142857086/ns;
        values[3][0][3][3] = 1.2971428571428572;
        values[3][1][0][0] = 1.8057142857142843/ns;
        values[3][1][0][1] = 0.8457142857142852;
        values[3][1][0][2] = 0.7028571428571464;
        values[3][1][0][3] = 0.22285714285714264*ns;
        values[3][1][1][0] = 0.19428571428571573/ns;
        values[3][1][1][1] = 0.16000000000000147;
        values[3][1][1][2] = 0.29714285714285182;
        values[3][1][1][3] = 0.080000000000000959*ns;
        values[3][1][2][0] = 4.1942857142857157/ns;
        values[3][1][2][1] = 1.1542857142857148;
        values[3][1][2][2] = -0.29714285714285799;
        values[3][1][2][3] = -0.11047619047619062*ns;
        values[3][1][3][0] = -6.1942857142857157/ns;
        values[3][1][3][1] = 3.8399999999999981;
        values[3][1][3][2] = 1.2971428571428572;
        values[3][1][3][3] = -0.58666666666666778*ns;
        values[3][2][0][0] = 1.8057142857142878/ns;
        values[3][2][0][1] = 0.70285714285714196;
        values[3][2][0][2] = 0.8457142857142852;
        values[3][2][0][3] = 0.22285714285714442*ns;
        values[3][2][1][0] = 4.1942857142857051/ns;
        values[3][2][1][1] = -0.29714285714285776;
        values[3][2][1][2] = 1.154285714285713;
        values[3][2][1][3] = -0.11047619047618884*ns;
        values[3][2][2][0] = 0.19428571428571217/ns;
        values[3][2][2][1] = 0.29714285714285799;
        values[3][2][2][2] = 0.16000000000000059;
        values[3][2][2][3] = 0.079999999999999932*ns;
        values[3][2][3][0] = -6.1942857142857086/ns;
        values[3][2][3][1] = 1.2971428571428572;
        values[3][2][3][2] = 3.8399999999999999;
        values[3][2][3][3] = -0.58666666666666667*ns;
        values[3][3][0][0] = -0.70285714285714196;
        values[3][3][0][1] = -0.22285714285714309*ns;
        values[3][3][0][2] = -0.22285714285714442*ns;
        values[3][3][0][3] = -0.052698412698412467*pow(ns, 2);
        values[3][3][1][0] = -0.29714285714285776;
        values[3][3][1][1] = -0.079999999999999849*ns;
        values[3][3][1][2] = -0.11047619047618884*ns;
        values[3][3][1][3] = -0.036825396825398204*pow(ns, 2);
        values[3][3][2][0] = -0.29714285714285799;
        values[3][3][2][1] = -0.11047619047619062*ns;
        values[3][3][2][2] = -0.079999999999999932*ns;
        values[3][3][2][3] = -0.036825396825396706*pow(ns, 2);
        values[3][3][3][0] = 1.2971428571428572;
        values[3][3][3][1] = -0.58666666666666778*ns;
        values[3][3][3][2] = -0.58666666666666667*ns;
        values[3][3][3][3] = 0.22349206349206341*pow(ns, 2);
    }

    static float integrateLaplacian(glm::vec2 nodeSize, std::array<std::array<float, 4>, 4>& values)
    {
        auto pow = [](double val, uint32_t n) -> double
        {
            double res = val;
            for(uint32_t i=1; i < n; i++) res *= val;
            return res;
        };

        const float ns = nodeSize.x;

        return 0.11174603174603126*pow(ns, 2)*pow(values[0][3], 2) - 0.036825396825396983*pow(ns, 2)*values[0][3]*values[1][3] - 0.036825396825398204*pow(ns, 2)*values[0][3]*values[2][3] - 0.052698412698412467*pow(ns, 2)*values[0][3]*values[3][3] + 0.11174603174603126*pow(ns, 2)*pow(values[1][3], 2) - 0.052698412698412467*pow(ns, 2)*values[1][3]*values[2][3] - 0.036825396825398204*pow(ns, 2)*values[1][3]*values[3][3] + 0.1117460317460317*pow(ns, 2)*pow(values[2][3], 2) - 0.036825396825396706*pow(ns, 2)*values[2][3]*values[3][3] + 0.1117460317460317*pow(ns, 2)*pow(values[3][3], 2) + 0.58666666666666623*ns*values[0][1]*values[0][3] + 0.1104761904761915*ns*values[0][1]*values[1][3] - 0.079999999999999849*ns*values[0][1]*values[2][3] - 0.22285714285714309*ns*values[0][1]*values[3][3] + 0.58666666666666956*ns*values[0][2]*values[0][3] - 0.07999999999999563*ns*values[0][2]*values[1][3] + 0.11047619047618884*ns*values[0][2]*values[2][3] - 0.22285714285714442*ns*values[0][2]*values[3][3] + 0.1104761904761915*ns*values[0][3]*values[1][1] + 0.07999999999999563*ns*values[0][3]*values[1][2] + 0.080000000000000959*ns*values[0][3]*values[2][1] + 0.11047619047618884*ns*values[0][3]*values[2][2] + 0.22285714285714264*ns*values[0][3]*values[3][1] + 0.22285714285714442*ns*values[0][3]*values[3][2] + 0.58666666666666623*ns*values[1][1]*values[1][3] - 0.22285714285714309*ns*values[1][1]*values[2][3] - 0.079999999999999849*ns*values[1][1]*values[3][3] - 0.58666666666666956*ns*values[1][2]*values[1][3] + 0.22285714285714442*ns*values[1][2]*values[2][3] - 0.11047619047618884*ns*values[1][2]*values[3][3] + 0.22285714285714264*ns*values[1][3]*values[2][1] - 0.22285714285714442*ns*values[1][3]*values[2][2] + 0.080000000000000959*ns*values[1][3]*values[3][1] - 0.11047619047618884*ns*values[1][3]*values[3][2] - 0.58666666666666778*ns*values[2][1]*values[2][3] - 0.11047619047619062*ns*values[2][1]*values[3][3] + 0.58666666666666667*ns*values[2][2]*values[2][3] - 0.079999999999999932*ns*values[2][2]*values[3][3] - 0.11047619047619062*ns*values[2][3]*values[3][1] + 0.079999999999999932*ns*values[2][3]*values[3][2] - 0.58666666666666778*ns*values[3][1]*values[3][3] - 0.58666666666666667*ns*values[3][2]*values[3][3] + 1.2971428571428536*values[0][0]*values[0][3] + 0.2971428571428536*values[0][0]*values[1][3] + 0.29714285714285776*values[0][0]*values[2][3] - 0.70285714285714196*values[0][0]*values[3][3] + 1.9200000000000008*pow(values[0][1], 2) + 1.2971428571428536*values[0][1]*values[0][2] + 1.1542857142857148*values[0][1]*values[1][1] - 0.2971428571428536*values[0][1]*values[1][2] + 0.16000000000000147*values[0][1]*values[2][1] + 0.29714285714285776*values[0][1]*values[2][2] + 0.8457142857142852*values[0][1]*values[3][1] + 0.70285714285714196*values[0][1]*values[3][2] + 1.9199999999999946*pow(values[0][2], 2) + 0.2971428571428536*values[0][2]*values[1][1] + 0.16000000000001791*values[0][2]*values[1][2] - 0.29714285714285182*values[0][2]*values[2][1] + 1.154285714285713*values[0][2]*values[2][2] + 0.7028571428571464*values[0][2]*values[3][1] + 0.8457142857142852*values[0][2]*values[3][2] - 0.2971428571428536*values[0][3]*values[1][0] - 0.29714285714285182*values[0][3]*values[2][0] - 0.7028571428571464*values[0][3]*values[3][0] - 1.2971428571428536*values[1][0]*values[1][3] + 0.70285714285714196*values[1][0]*values[2][3] - 0.29714285714285776*values[1][0]*values[3][3] + 1.9200000000000008*pow(values[1][1], 2) - 1.2971428571428536*values[1][1]*values[1][2] + 0.8457142857142852*values[1][1]*values[2][1] - 0.70285714285714196*values[1][1]*values[2][2] + 0.16000000000000147*values[1][1]*values[3][1] - 0.29714285714285776*values[1][1]*values[3][2] + 1.9199999999999946*pow(values[1][2], 2) - 0.7028571428571464*values[1][2]*values[2][1] + 0.8457142857142852*values[1][2]*values[2][2] + 0.29714285714285182*values[1][2]*values[3][1] + 1.154285714285713*values[1][2]*values[3][2] + 0.7028571428571464*values[1][3]*values[2][0] + 0.29714285714285182*values[1][3]*values[3][0] - 1.2971428571428572*values[2][0]*values[2][3] - 0.29714285714285799*values[2][0]*values[3][3] + 1.919999999999999*pow(values[2][1], 2) - 1.2971428571428572*values[2][1]*values[2][2] + 1.1542857142857148*values[2][1]*values[3][1] + 0.29714285714285799*values[2][1]*values[3][2] + 1.9199999999999999*pow(values[2][2], 2) - 0.29714285714285799*values[2][2]*values[3][1] + 0.16000000000000059*values[2][2]*values[3][2] + 0.29714285714285799*values[2][3]*values[3][0] + 1.2971428571428572*values[3][0]*values[3][3] + 1.919999999999999*pow(values[3][1], 2) + 1.2971428571428572*values[3][1]*values[3][2] + 1.9199999999999999*pow(values[3][2], 2) + 6.1942857142857157*values[0][0]*values[0][1]/ns + 6.1942857142857122*values[0][0]*values[0][2]/ns + 4.1942857142857193*values[0][0]*values[1][1]/ns - 0.19428571428571217*values[0][0]*values[1][2]/ns - 0.19428571428571573*values[0][0]*values[2][1]/ns + 4.1942857142857051*values[0][0]*values[2][2]/ns + 1.8057142857142843*values[0][0]*values[3][1]/ns + 1.8057142857142878*values[0][0]*values[3][2]/ns - 4.1942857142857193*values[0][1]*values[1][0]/ns - 0.19428571428571573*values[0][1]*values[2][0]/ns - 1.8057142857142843*values[0][1]*values[3][0]/ns - 0.19428571428571217*values[0][2]*values[1][0]/ns - 4.1942857142857051*values[0][2]*values[2][0]/ns - 1.8057142857142878*values[0][2]*values[3][0]/ns - 6.1942857142857157*values[1][0]*values[1][1]/ns + 6.1942857142857122*values[1][0]*values[1][2]/ns - 1.8057142857142843*values[1][0]*values[2][1]/ns + 1.8057142857142878*values[1][0]*values[2][2]/ns + 0.19428571428571573*values[1][0]*values[3][1]/ns + 4.1942857142857051*values[1][0]*values[3][2]/ns + 1.8057142857142843*values[1][1]*values[2][0]/ns + 0.19428571428571573*values[1][1]*values[3][0]/ns - 1.8057142857142878*values[1][2]*values[2][0]/ns - 4.1942857142857051*values[1][2]*values[3][0]/ns + 6.1942857142857157*values[2][0]*values[2][1]/ns - 6.1942857142857086*values[2][0]*values[2][2]/ns + 4.1942857142857157*values[2][0]*values[3][1]/ns + 0.19428571428571217*values[2][0]*values[3][2]/ns - 4.1942857142857157*values[2][1]*values[3][0]/ns + 0.19428571428571217*values[2][2]*values[3][0]/ns - 6.1942857142857157*values[3][0]*values[3][1]/ns - 6.1942857142857086*values[3][0]*values[3][2]/ns + 11.79428571428571*pow(values[0][0], 2)/pow(ns, 2) - 11.588571428571424*values[0][0]*values[1][0]/pow(ns, 2) - 11.588571428571413*values[0][0]*values[2][0]/pow(ns, 2) - 0.41142857142857636*values[0][0]*values[3][0]/pow(ns, 2) + 11.79428571428571*pow(values[1][0], 2)/pow(ns, 2) - 0.41142857142857636*values[1][0]*values[2][0]/pow(ns, 2) - 11.588571428571413*values[1][0]*values[3][0]/pow(ns, 2) + 11.794285714285706*pow(values[2][0], 2)/pow(ns, 2) - 11.588571428571427*values[2][0]*values[3][0]/pow(ns, 2) + 11.794285714285706*pow(values[3][0], 2)/pow(ns, 2);
    }
};

#endif