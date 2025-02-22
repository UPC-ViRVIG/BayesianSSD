cmake_minimum_required(VERSION 3.16)

# Use if PkgConfig is avaliable
# set(ENV{PKG_CONFIG_PATH} "$ENV{CONDA_PREFIX}/lib/pkgconfig")
# find_package(PkgConfig REQUIRED)
# pkg_check_modules(GSL REQUIRED IMPORTED_TARGET gsl)

# Add gsl

find_library(gsl_lib NAMES gsl.lib PATHS "$ENV{CONDA_PREFIX}/Library/lib/")

if(gsl_lib)
    add_library(gsl INTERFACE)
    target_link_libraries(gsl INTERFACE ${gsl_lib})
    target_include_directories(gsl INTERFACE "$ENV{CONDA_PREFIX}/Library/include")
    message(${gsl_lib})
endif()

set(LowRankSVD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/LowRankSVDCodes/single_core_gsl_code")
if((EXISTS "${LowRankSVD_DIR}") AND (TARGET gsl))
    file(GLOB SOURCE_FILES "${LowRankSVD_DIR}/low_rank_svd_algorithms_gsl.c"
                           "${LowRankSVD_DIR}/matrix_vector_functions_gsl.c")
    add_library(LowRankSVD ${SOURCE_FILES})
    target_include_directories(LowRankSVD PUBLIC "${LowRankSVD_DIR}")
    target_link_libraries(LowRankSVD gsl)
    target_compile_definitions(LowRankSVD PUBLIC LOW_RANK_SVD_AVAILABLE)
    message("Using LowRankSVD library")
else()
    add_library(LowRankSVD INTERFACE)
endif()


add_library(petsc INTERFACE)
find_package(PkgConfig)
if(PkgConfig_FOUND)
    set(ENV{PKG_CONFIG_PATH} "$ENV{CONDA_PREFIX}/lib/pkgconfig")
    pkg_check_modules(PETSC IMPORTED_TARGET GLOBAL PETSc)
    if(${PETSC_FOUND})
        target_link_libraries(petsc INTERFACE PkgConfig::PETSC)
        target_compile_definitions(petsc INTERFACE PETSC_AVAILABLE)
    endif()
endif()