cmake_minimum_required(VERSION 3.16)

project(recon LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FetchContent)
set(CMAKE_POLICY_DEFAULT_CMP0148 NEW)

# Eigen
FetchContent_Declare(eigen_lib
GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
GIT_TAG 46126273552afe13692929523d34006f54c19719 # 3.4
)

FetchContent_GetProperties(eigen_lib)
if(NOT eigen_lib_POPULATED)
    FetchContent_Populate(eigen_lib)
    add_library(eigen INTERFACE)
    target_include_directories(eigen INTERFACE ${eigen_lib_SOURCE_DIR})
endif()

# glm
FetchContent_Declare(glm_lib
	  GIT_REPOSITORY https://github.com/g-truc/glm.git
	  GIT_TAG 89e52e327d7a3ae61eb402850ba36ac4dd111987 # 0.9.8
	)

FetchContent_GetProperties(glm_lib)
if(NOT glm_lib_POPULATED)
    FetchContent_Populate(glm_lib)
    add_subdirectory(${glm_lib_SOURCE_DIR} ${glm_lib_BINARY_DIR})
endif()
set(SDFLIB_GLM_TARGET glm PARENT_SCOPE)


file(GLOB SOURCE_FILES ./src/*.cpp)
add_executable(recon src/tools/main.cpp ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PRIVATE src)
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_link_libraries(${PROJECT_NAME} PUBLIC eigen glm)