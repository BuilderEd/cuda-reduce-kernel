cmake_minimum_required(VERSION 3.10)
project(REDUCE CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories(include)

find_package(OpenMP REQUIRED)

find_package(CUDA REQUIRED)


include_directories(include)


file(GLOB SOURCES "src/*.cpp" "src/*.cu" "*.cpp")


add_executable(reduce ${SOURCES})


#-O3 -ffast-math -march=native -mtune=native
target_compile_options(reduce PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>: -O3 -ffast-math -march=native -mtune=native -Wall -g -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>: 
    -Xcompiler -fopenmp 
    -Xcompiler -O3
    -Xcompiler -ffast-math
    -Xcompiler -march=native
    -Xcompiler -Wall
    -O3 --use_fast_math>
)
target_link_libraries(reduce PRIVATE OpenMP::OpenMP_CXX 
     ${CUDA_LIBRARIES}
    )

set_property(TARGET reduce PROPERTY CUDA_ARCHITECTURES 80) 
