cmake_minimum_required(VERSION 3.16)

project(recon LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FetchContent)
set(CMAKE_POLICY_DEFAULT_CMP0148 NEW)

option(USE_OPENMP "Use openMP" OFF)

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

# happly
FetchContent_Declare(happly_lib
	  GIT_REPOSITORY https://github.com/nmwsharp/happly.git
	  GIT_TAG 8a606309daaa680eee495c8279feb0b704148f4a # 0.9.8
	)

FetchContent_GetProperties(happly_lib)
if(NOT happly_lib_POPULATED)
    FetchContent_Populate(happly_lib)
    add_library(happly INTERFACE)
    target_include_directories(happly INTERFACE ${happly_lib_SOURCE_DIR})
endif()

# cereal
FetchContent_Declare(cereal_lib
    GIT_REPOSITORY https://github.com/USCiLab/cereal.git
    GIT_TAG ebef1e929807629befafbb2918ea1a08c7194554 # 1.3.2
)

FetchContent_GetProperties(cereal_lib)
if(NOT cereal_lib_POPULATED)
    FetchContent_Populate(cereal_lib)
    set(BUILD_DOC OFF)
    set(BUILD_SANDBOX  OFF)
    set(BUILD_TESTS OFF)
    set(CEREAL_INSTALL OFF)
    set(SKIP_PERFORMANCE_COMPARISON ON)
    add_subdirectory(${cereal_lib_SOURCE_DIR} ${cereal_lib_BINARY_DIR})
endif()

# Nanoflann
FetchContent_Declare(nanoflann_lib
    GIT_REPOSITORY https://github.com/jlblancoc/nanoflann.git
    GIT_TAG f1e47f0dbae72c11b008fd0d5b1191d9ba1f29bb # 1.5.0
)

FetchContent_GetProperties(nanoflann_lib)
if(NOT nanoflann_lib_POPULATED)
    FetchContent_Populate(nanoflann_lib)
    add_library(nanoflann INTERFACE)
    target_include_directories(nanoflann INTERFACE ${nanoflann_lib_SOURCE_DIR}/include)
endif()

# MarchingCubes
FetchContent_Declare(meshreconstruction_lib
    GIT_REPOSITORY https://github.com/Magnus2/MeshReconstruction.git
    GIT_TAG 423213b1222cbe97ad45434ab112500111a966ff
)

FetchContent_GetProperties(meshreconstruction_lib)
if(NOT meshreconstruction_lib_POPULATED)
    FetchContent_Populate(meshreconstruction_lib)
    add_subdirectory(${meshreconstruction_lib_SOURCE_DIR} ${meshreconstruction_lib_BINARY_DIR})
endif()

#Json struct
FetchContent_Declare(json_struct_lib
    GIT_REPOSITORY https://github.com/jorgen/json_struct.git
    GIT_TAG 19a71df0df00da616865eb0a885627ccbe0ec5de
)

FetchContent_GetProperties(json_struct_lib)
if(NOT json_struct_lib_POPULATED)
    FetchContent_Populate(json_struct_lib)
    add_subdirectory(${json_struct_lib_SOURCE_DIR} ${json_struct_lib_BINARY_DIR})
endif()


#OpenMP
add_library(openmp INTERFACE)
if(USE_OPENMP)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        message("Enabling openmp llvm extension")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp:llvm")
        target_compile_definitions(openmp INTERFACE -DOPENMP_AVAILABLE)
    else()
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
        	message("OpenMP version ${OpenMP_CXX_VERSION}")
            target_link_libraries(openmp INTERFACE OpenMP::OpenMP_CXX)
            target_compile_definitions(openmp INTERFACE -DOPENMP_AVAILABLE)
        else()
            message("Disabling openmp")
        endif()
    endif()
endif()

add_subdirectory(libs)

file(GLOB SOURCE_FILES ./src/*.cpp)
add_executable(recon_3d src/tools/3D/main.cpp ${SOURCE_FILES})
target_include_directories(recon_3d PRIVATE src)
target_include_directories(recon_3d PUBLIC include)
target_link_libraries(recon_3d PUBLIC eigen glm happly cereal::cereal nanoflann LowRankSVD openmp MeshReconstruction json_struct petsc)
if(MSVC)
    target_compile_options(recon_3d PRIVATE "/bigobj")
endif()

add_executable(recon_2d src/tools/2D/main.cpp ${SOURCE_FILES})
target_include_directories(recon_2d PRIVATE src)
target_include_directories(recon_2d PUBLIC include)
target_link_libraries(recon_2d PUBLIC eigen glm happly cereal::cereal nanoflann LowRankSVD openmp MeshReconstruction json_struct petsc)
if(MSVC)
    target_compile_options(recon_2d PRIVATE "/bigobj")
endif()

add_library(recon_lib STATIC ${SOURCE_FILES})
target_include_directories(recon_lib PRIVATE src)
target_include_directories(recon_lib PUBLIC include)
target_link_libraries(recon_lib PUBLIC eigen glm happly cereal::cereal nanoflann LowRankSVD openmp MeshReconstruction json_struct petsc)
if(MSVC)
    target_compile_options(recon_lib PRIVATE "/bigobj")
endif()

add_subdirectory(viewer)