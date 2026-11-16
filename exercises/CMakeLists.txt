cmake_minimum_required(VERSION 3.18)

# Gadi's gpuvolta nodes use V100 GPUs. Override for other GPUs at configure time.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA architectures to build for")
endif()

project(cuda_exercises LANGUAGES C CXX CUDA)
find_package(CUDAToolkit REQUIRED)

option(EXERCISES_BUILD_MKL "Build the CPU Intel MKL GEMM example" ON)

if(EXERCISES_BUILD_MKL)
    find_path(MKL_INCLUDE_DIR mkl.h HINTS ENV MKLROOT PATH_SUFFIXES include)
    foreach(component IN ITEMS intel_lp64 sequential core)
        find_library(MKL_${component}_LIBRARY NAMES mkl_${component}
            HINTS ENV MKLROOT PATH_SUFFIXES lib/intel64 lib)
        if(NOT MKL_${component}_LIBRARY)
            message(FATAL_ERROR "MKL library mkl_${component} not found. Load intel-mkl or set EXERCISES_BUILD_MKL=OFF.")
        endif()
    endforeach()
    if(NOT MKL_INCLUDE_DIR)
        message(FATAL_ERROR "mkl.h not found. Load intel-mkl or set EXERCISES_BUILD_MKL=OFF.")
    endif()
    find_package(Threads REQUIRED)
endif()

# Restrict discovery to exercise folders, so in-tree build/compiler-probe files
# cannot accidentally become exercise targets. Include nested legacy solutions.
file(GLOB exercise_dirs CONFIGURE_DEPENDS LIST_DIRECTORIES TRUE
    "${CMAKE_CURRENT_SOURCE_DIR}/[0-9][0-9]-*")
list(APPEND exercise_dirs "${CMAKE_CURRENT_SOURCE_DIR}/legacy")

foreach(exercise_dir IN LISTS exercise_dirs)
    file(GLOB_RECURSE sources CONFIGURE_DEPENDS
        "${exercise_dir}/*.cu" "${exercise_dir}/*.c" "${exercise_dir}/*.cpp")
    foreach(source IN LISTS sources)
        get_filename_component(stem "${source}" NAME_WE)
        get_filename_component(extension "${source}" EXT)
        get_filename_component(source_dir "${source}" DIRECTORY)
        file(RELATIVE_PATH folder "${CMAKE_CURRENT_SOURCE_DIR}" "${source_dir}")
        if(stem STREQUAL "perf_gemm_intelmkl" AND NOT EXERCISES_BUILD_MKL)
            continue()
        endif()

        # Prefix with the folder to distinguish repeated vector_add/axpy names.
        string(MAKE_C_IDENTIFIER "ex_${folder}_${stem}" target)
        add_executable(${target} "${source}")
        set_target_properties(${target} PROPERTIES OUTPUT_NAME "${target}"
            C_STANDARD 11 C_STANDARD_REQUIRED YES
            CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES
            CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES)
        set(output_dir "${CMAKE_BINARY_DIR}")
        set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${output_dir}")
        foreach(config IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
            set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${config} "${output_dir}")
        endforeach()

        if(extension STREQUAL ".cu")
            target_link_libraries(${target} PRIVATE CUDA::cudart)
            if(stem STREQUAL "perf_gemm_cuda" OR folder STREQUAL "13-cublas")
                target_link_libraries(${target} PRIVATE CUDA::cublas)
            endif()
            if(folder STREQUAL "09-dynamic-parallelism")
                # CMake/nvcc handle device-runtime linking for separable CUDA
                # compilation; FindCUDAToolkit does not define CUDA::cudadevrt.
                set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
            elseif(folder STREQUAL "15-profiling")
                target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
            elseif(folder STREQUAL "16-debugging")
                target_compile_options(${target} PRIVATE
                    $<$<COMPILE_LANGUAGE:CUDA>:-G> $<$<COMPILE_LANGUAGE:CUDA>:-g>)
            endif()
        elseif(stem STREQUAL "perf_gemm_intelmkl")
            target_include_directories(${target} PRIVATE "${MKL_INCLUDE_DIR}")
            target_link_libraries(${target} PRIVATE
                "${MKL_intel_lp64_LIBRARY}" "${MKL_sequential_LIBRARY}" "${MKL_core_LIBRARY}"
                Threads::Threads ${CMAKE_DL_LIBS} m)
        endif()
    endforeach()
endforeach()
