cmake_minimum_required(VERSION 3.20)
project(dr_moagi_billion3_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(DM3D_ENABLE_NATIVE_OPT "Enable host-specific native optimization" OFF)

find_package(Threads REQUIRED)

add_executable(dm3d_mmc
    cpp/dm3d_multimodal_coding_environment.cpp
)

add_executable(dm3d_verify
    cpp/dm3d_inward_e2e_verifier.cpp
)

foreach(target dm3d_mmc dm3d_verify)
    target_link_libraries(${target} PRIVATE Threads::Threads)

    if(MSVC)
        target_compile_options(${target} PRIVATE /W4)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
        if(DM3D_ENABLE_NATIVE_OPT)
            target_compile_options(${target} PRIVATE -O3 -march=native)
        endif()
    endif()
endforeach()

enable_testing()
add_test(NAME dm3d_end_to_end COMMAND dm3d_verify)
