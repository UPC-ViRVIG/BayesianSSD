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
};

#endif