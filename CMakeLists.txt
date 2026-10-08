cmake_minimum_required(VERSION 3.18)
project(mlp_accelerator_sim LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE pybind11_result
)
if(NOT pybind11_result EQUAL 0)
    message(FATAL_ERROR "Install pybind11 for ${Python_EXECUTABLE}: python -m pip install pybind11")
endif()
find_package(pybind11 CONFIG REQUIRED HINTS "${pybind11_cmake_dir}")

find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    find_path(EIGEN3_INCLUDE_DIR Eigen/Dense PATH_SUFFIXES eigen3)
    if(NOT EIGEN3_INCLUDE_DIR)
        message(FATAL_ERROR "Install Eigen3 or set -DEIGEN3_INCLUDE_DIR=<directory containing Eigen/>")
    endif()
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
endif()

add_executable(mlp_accelerator_sim src/main.cpp src/accelerator/ram.cpp)
target_include_directories(mlp_accelerator_sim PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(mlp_accelerator_sim PRIVATE cxx_std_17)
if(MINGW)
    target_link_options(mlp_accelerator_sim PRIVATE -static)
endif()

pybind11_add_module(nn NO_EXTRAS src/mlp/nn.cpp)
target_compile_features(nn PRIVATE cxx_std_17)
target_link_libraries(nn PRIVATE Eigen3::Eigen)
if(MINGW)
    target_link_options(nn PRIVATE -static)
endif()
set_target_properties(nn PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python")
foreach(config IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(nn PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${config} "${CMAKE_BINARY_DIR}/python"
        RUNTIME_OUTPUT_DIRECTORY_${config} "${CMAKE_BINARY_DIR}/python")
endforeach()
