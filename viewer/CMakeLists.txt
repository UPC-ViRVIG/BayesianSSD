

# Sdflib
FetchContent_Declare(sdflib_lib
GIT_REPOSITORY https://github.com/UPC-ViRVIG/SdfLib.git
GIT_TAG 2b60726409e7a1fd59e97a6013219b010027313c # development branch
)

FetchContent_GetProperties(sdflib_lib)
if(NOT sdflib_lib_POPULATED)
    FetchContent_Populate(sdflib_lib)
    set(SDFLIB_USE_ASSIMP OFF)
    set(SDFLIB_USE_ENOKI OFF)
    set(SDFLIB_USE_OPENMP OFF)
    add_subdirectory(${sdflib_lib_SOURCE_DIR} ${sdflib_lib_BINARY_DIR})
endif()

# MyRender
FetchContent_Declare(myrender_lib
GIT_REPOSITORY https://github.com/triter3/MyRender.git
GIT_TAG d6eb948f2e91d919110d7a8ad095a75d94fae6a2
)

FetchContent_GetProperties(myrender_lib)
if(NOT myrender_lib_POPULATED)
    FetchContent_Populate(myrender_lib)
    set(MYRENDER_BUILD_EXAMPLES  OFF)
    add_subdirectory(${myrender_lib_SOURCE_DIR} ${myrender_lib_BINARY_DIR})
endif()

# args
FetchContent_Declare(args_lib
    GIT_REPOSITORY https://github.com/Taywee/args.git
    GIT_TAG a48e1f880813b367d2354963a58dedbf2b708584 # 6.3.0
)

FetchContent_GetProperties(args_lib)
if(NOT args_lib_POPULATED)
    FetchContent_Populate(args_lib)
    add_library(args INTERFACE)
    target_include_directories(args INTERFACE ${args_lib_SOURCE_DIR})
endif()

add_executable(Viewer ./main.cpp)
target_link_libraries(Viewer PUBLIC SdfLib MyRender args)