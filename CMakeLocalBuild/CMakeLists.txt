cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(fkBaseModel LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 50)
# set(CMAKE_GENERATOR "Visual Studio 17 2022" CACHE INTERNAL "Name of generator.")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY run)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY run)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY run)

# Ensure cross-compiler toolchain is set if needed
# set(CMAKE_C_COMPILER arm-none-eabi-gcc)


add_library(fkModel_lib STATIC
  ../PreCompiledHeader/FKpch.cpp
  ../PreCompiledHeader/FKpch.h
  ../CUDAfiles/ComputeFK.cuh
  ../Structures/StructureManagment.h
  )


set_target_properties( fkModel_lib
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )

add_executable(fkModel main.cpp CUDAfiles/ComputeFK.cu)

target_link_libraries(fkModel PRIVATE fkModel_lib)

# install(TARGETS fkModel DESTINATION run)

target_include_directories(fhnModel PRIVATE /usr/AppData/local/NVIDIA)

SET (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_50 -std=c++17")
