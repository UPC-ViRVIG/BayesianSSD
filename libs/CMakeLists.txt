cmake_minimum_required(VERSION 3.16)

# Use if PkgConfig is avaliable
# set(ENV{PKG_CONFIG_PATH} "$ENV{CONDA_PREFIX}/lib/pkgconfig")
# find_package(PkgConfig REQUIRED)
# pkg_check_modules(GSL REQUIRED IMPORTED_TARGET gsl)

# Add gsl
add_library(gsl INTERFACE)
find_library(gsl_lib NAMES gsl.lib PATHS "$ENV{CONDA_PREFIX}/Library/lib/" REQUIRED)
target_link_libraries(gsl INTERFACE ${gsl_lib})
target_include_directories(gsl INTERFACE "$ENV{CONDA_PREFIX}/Library/include")

file(GLOB SOURCE_FILES ./LowRankSVDCodes/single_core_gsl_code/low_rank_svd_algorithms_gsl.c
                       ./LowRankSVDCodes/single_core_gsl_code/matrix_vector_functions_gsl.c)
add_library(LowRandSVD ${SOURCE_FILES})
target_include_directories(LowRandSVD PUBLIC ./LowRankSVDCodes/single_core_gsl_code/)
target_link_libraries(LowRandSVD gsl)

