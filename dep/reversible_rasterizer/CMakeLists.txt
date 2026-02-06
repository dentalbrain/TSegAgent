cmake_minimum_required(VERSION 3.27)
project(reversible_rasterizer)

include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-fpermissive -fPIC -fopenmp -w")
SET(CMAKE_C_FLAGS "-fPIC ${CMAKE_C_FLAGS}")

set(BUILD_EXECUTABLES OFF)
set(BUILD_PYTHON_LIB ON)
set(BUILD_OFFSCREEN_VERSION ON)
add_compile_definitions(NOMINMAX)

# Eigen3
FetchContent_Declare(Eigen3 URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
        CMAKE_ARGS -DBUILD_TESTING=OFF -DEIGEN_BUILD_DOC=OFF -DEIGEN_TEST_NOQT=ON
)
FetchContent_MakeAvailable(Eigen3)
include_directories(${EIGEN_INCLUDE_DIRECTORIES})

# OpenGL
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS})

if (NOT BUILD_OFFSCREEN_VERSION)
    # GLFW
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    add_subdirectory(dep/glfw)
    include_directories(dep/glfw/include)
    include_directories(dep/glfw/deps)
endif ()

# GLAD 2
add_library(glad dep/glad/src/gl.c dep/glad/include/glad/glad.h)
include_directories(dep/glad/include)
target_include_directories(glad PUBLIC dep/glad/include)
target_compile_options(glad PUBLIC "-fPIC")
set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl")

# GLM
add_subdirectory(dep/glm)
include_directories(dep/glm)

# Assimp
set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE)
set(ASSIMP_BUILD_ZLIB ON CACHE BOOL "" FORCE)
add_subdirectory(dep/assimp)
include_directories(dep/assimp/include)

include_directories(./src)

if (BUILD_OFFSCREEN_VERSION)
    find_package(OpenGL COMPONENTS EGL REQUIRED)
    include_directories(${OPENGL_EGL_INCLUDE_DIR})

    add_compile_definitions(USE_OFFSCREEN_RENDERER)
    set(RENDERER_SOURCE src/offscreen_renderer.cpp)
    set(CUSTOM_GLFW_TARGET_LINK ${OPENGL_egl_LIBRARY})
else ()
    set(RENDERER_SOURCE src/renderer.cpp)
    set(CUSTOM_GLFW_TARGET_LINK "glfw")
endif ()

set(GLSL_DIR ${CMAKE_SOURCE_DIR}/src/shaders)
set(EMBED_OUTPUT_DIR ${CMAKE_BINARY_DIR}/generated_shaders)
set(EMBED_STAMP_FILE ${EMBED_OUTPUT_DIR}/.shaders_embedded.stamp)
add_custom_command(
        OUTPUT ${EMBED_STAMP_FILE}
        COMMAND ${CMAKE_COMMAND}
        -DGLSL_DIR=${GLSL_DIR}
        -DOUTPUT_DIR=${EMBED_OUTPUT_DIR}
        -DSTAMP_FILE=${EMBED_STAMP_FILE}
        -P ${CMAKE_SOURCE_DIR}/cmake/embed_glsl.cmake
        DEPENDS
        ${CMAKE_SOURCE_DIR}/cmake/embed_glsl.cmake
        COMMENT "Embedding all GLSL shaders under ${GLSL_DIR}"
        VERBATIM
)
add_custom_target(EmbedShaders ALL DEPENDS ${EMBED_STAMP_FILE})

if (BUILD_PYTHON_LIB)
    find_package(Python)
    FetchContent_Declare(pybind11 URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz)
    FetchContent_MakeAvailable(pybind11)
    add_compile_definitions(FOUND_PYBIND11)
    pybind11_add_module(librevras SHARED
            src/api.h
            src/api.cpp
            src/glspec.h
            src/PyBind.cpp
            src/model_loader.cpp
            src/model_loader.h
            ${RENDERER_SOURCE}
            src/renderer.h
            src/config.h
            src/mesh.hpp
            src/shader.h
            src/shader.cpp
    )
    add_dependencies(librevras EmbedShaders)
    target_include_directories(librevras PRIVATE ${EMBED_OUTPUT_DIR})
    target_link_libraries(librevras PUBLIC
            ${OPENGL_LIBRARIES}
            ${CUSTOM_GLFW_TARGET_LINK}
            glad
            Eigen3::Eigen
            assimp
            pybind11::module)
    target_compile_definitions(librevras PUBLIC FOUND_PYBIND11)
    set_target_properties(librevras PROPERTIES
            BUILD_WITH_INSTALL_RPATH ON
            INSTALL_RPATH "\$ORIGIN"
            SKIP_BUILD_RPATH OFF
            )
endif ()

if (BUILD_EXECUTABLES)
    add_executable(reversible_rasterizer_exe src/main.cpp
            src/api.h
            src/api.cpp
            src/glspec.h
            src/model_loader.cpp
            src/model_loader.h
            ${RENDERER_SOURCE}
            src/renderer.h
            src/config.h
            src/mesh.hpp
            src/shader.h
            src/shader.cpp
    )
    add_dependencies(reversible_rasterizer_exe EmbedShaders)
    target_include_directories(reversible_rasterizer_exe PRIVATE ${EMBED_OUTPUT_DIR})
    target_link_libraries(reversible_rasterizer_exe
            ${OPENGL_LIBRARIES}
            ${CUSTOM_GLFW_TARGET_LINK}
            glad
            Eigen3::Eigen
            assimp
    )
endif ()
