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

add_library(mlp_reference STATIC src/mlp/reference.cpp)
target_include_directories(mlp_reference PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(mlp_reference PUBLIC cxx_std_17)
target_link_libraries(mlp_reference PUBLIC Eigen3::Eigen)

# Keep the original implementation and Python API intact during the migration.
pybind11_add_module(nn NO_EXTRAS src/mlp/nn.cpp)
target_compile_features(nn PRIVATE cxx_std_17)
target_link_libraries(nn PRIVATE Eigen3::Eigen)
if(MINGW)
    # Bundle the MinGW runtime so importing nn does not require compiler DLLs.
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

include(CTest)
if(BUILD_TESTING)
    add_executable(reference_tests tests/reference_tests.cpp)
    target_link_libraries(reference_tests PRIVATE mlp_reference)
    if(MINGW)
        target_link_options(reference_tests PRIVATE -static)
    endif()
    add_test(NAME reference_numerics COMMAND reference_tests)
    add_test(NAME reference_legacy_parity
        COMMAND "${Python_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_legacy_parity.py"
            "$<TARGET_FILE:reference_tests>" "$<TARGET_FILE_DIR:nn>")
endif()
