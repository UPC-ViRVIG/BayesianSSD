#ifndef BOUNDING_BOX_H
#define BOUNDING_BOX_H

#include <glm/glm.hpp>

struct BoundingBox
{
    BoundingBox() : min(INFINITY), max(-INFINITY) {}
    BoundingBox(glm::vec3 min, glm::vec3 max)
        : min(min),
          max(max)
    {}
    glm::vec3 min;
    glm::vec3 max;

    glm::vec3 getSize() const
    {
        return max - min;
    }

    glm::vec3 getCenter() const
    {
        return min + 0.5f * getSize();
    }

    void addMargin(float margin)
    {
        min -= glm::vec3(margin);
        max += glm::vec3(margin);
    }
};

#endif