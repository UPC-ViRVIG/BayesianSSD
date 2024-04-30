#ifndef INTERPOLATION_METHOD_H
#define INTERPOLATION_METHOD_H

#include <glm/glm.hpp>
#include <array>

namespace BilinearInterpolation
{
    void eval(glm::vec2 np, std::array<float, 4>& outWeights)
    {
        outWeights[0] = (1.0f - np.x) * (1.0f - np.y);
        outWeights[1] = np.x * (1.0f - np.y);
        outWeights[2] = (1.0f - np.x) * np.y;
        outWeights[3] = np.x * np.y;
    }

    void evalGrad(glm::vec2 np, std::array<std::array<float, 4>, 2>& outWeights)
    {
        // In X
        outWeights[0][0] = -(1.0f - np.y);
        outWeights[0][1] = (1.0f - np.y);
        outWeights[0][2] = -np.y;
        outWeights[0][3] = np.y;

        // In Y
        outWeights[1][0] = -(1.0f - np.x);
        outWeights[1][1] = -np.x;
        outWeights[1][2] = (1.0f - np.x);
        outWeights[1][3] = np.x;
    }
}

namespace TrilinarInterpolation
{
    void eval(glm::vec2 np, std::array<float, 4>& outWeights)
    {
        // TODO
    }

    void evalGrad(glm::vec2 np, std::array<std::array<float, 4>, 2>& outWeights)
    {
        // TODO
    }
}

#endif