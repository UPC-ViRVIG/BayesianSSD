#ifndef VECTOR_H
#define VECTOR_H

#include <array>
#include <glm/glm.hpp>

template<uint32_t Dim>
struct VStruct { typedef std::array<float, Dim> type; };

template<>
struct VStruct<2> { typedef glm::vec2 type; };

template<>
struct VStruct<3> { typedef glm::vec3 type; };

#endif