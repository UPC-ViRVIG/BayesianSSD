#ifndef EIGEN_DECOMPOSITION_LAPLACIAN_H
#define EIGEN_DECOMPOSITION_LAPLACIAN_H

#include <numbers>
#include <Eigen/Sparse>
#include <glm/glm.hpp>

namespace EigenDecompositionLaplacian
{
    Eigen::MatrixXd getMatrix(glm::ivec3 gridSize, std::vector<glm::ivec3> gridPoints, uint32_t numVals)
    {
        std::vector<std::pair<float, glm::ivec3>> eigenValues(gridSize.x * gridSize.y * gridSize.z);
        glm::vec3 gridSizeSq = gridSize * gridSize;
        for(int i=0; i < eigenValues.size(); i++)
        {
            glm::ivec3 coords(i % gridSize.x, (i / gridSize.x) % gridSize.y, i / (gridSize.x * gridSize.y));
            const float value = static_cast<float>(coords.x * coords.x) / gridSizeSq.x + 
                                static_cast<float>(coords.y * coords.y) / gridSizeSq.y + 
                                static_cast<float>(coords.z * coords.z) / gridSizeSq.z;
            eigenValues[i] = std::make_pair(value, coords);
        }

        std::sort(eigenValues.begin(), eigenValues.end(), 
                    [](const std::pair<float, glm::ivec3>& a, 
                    const std::pair<float, glm::ivec3>& b){
                        return a.first < b.first;
                    });

        Eigen::MatrixXd eigenVectors = Eigen::MatrixXd::Zero(gridPoints.size(), numVals);
        for(uint32_t v=0; v < numVals; v++)
        {
            glm::ivec3 coords = eigenValues[v].second;
            for(uint32_t p=0; p < gridPoints.size(); p++)
            {
                double value = 1.0;
                for(uint32_t d=0; d < 3; d++)
                {
                    value = value * glm::cos(static_cast<double>(coords[d]) * std::numbers::pi * 
                                             static_cast<double>(gridPoints[p][d]) / static_cast<double>(gridSize[d]-1));
                }

                eigenVectors(p, v) = value;
            }
        }

        return std::move(eigenVectors);
    }
}

#endif