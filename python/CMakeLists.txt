cmake_minimum_required(VERSION 3.18)
project(pytree LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Armadillo REQUIRED)
find_package(OpenMP)

set(PTREE_CORE_SOURCES
    ../src/APTree.cpp
    ../src/PTreeModel.cpp
    ../src/common.cpp
    ../src/json_io.cpp
    ../src/ptree_core.cpp
)

add_library(ptree_core STATIC ${PTREE_CORE_SOURCES})
target_include_directories(ptree_core PRIVATE ../src)
if(TARGET Armadillo::Armadillo)
    target_link_libraries(ptree_core PRIVATE Armadillo::Armadillo)
else()
    target_include_directories(ptree_core PRIVATE ${ARMADILLO_INCLUDE_DIRS})
    target_link_libraries(ptree_core PRIVATE ${ARMADILLO_LIBRARIES})
endif()
target_compile_definitions(ptree_core PRIVATE PTREE_STANDALONE)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ptree_core PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(ptree_core PRIVATE ARMA_USE_OPENMP)
endif()

pybind11_add_module(_core src/pytree/_core.cpp)
target_link_libraries(_core PRIVATE ptree_core)
if(TARGET Armadillo::Armadillo)
    target_link_libraries(_core PRIVATE Armadillo::Armadillo)
else()
    target_include_directories(_core PRIVATE ${ARMADILLO_INCLUDE_DIRS})
    target_link_libraries(_core PRIVATE ${ARMADILLO_LIBRARIES})
endif()
target_include_directories(_core PRIVATE ../src)
target_compile_definitions(_core PRIVATE PTREE_STANDALONE)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_core PRIVATE ARMA_USE_OPENMP)
endif()

install(TARGETS _core DESTINATION pytree)
