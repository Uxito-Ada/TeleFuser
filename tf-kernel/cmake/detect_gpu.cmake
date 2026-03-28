# GPU Architecture Detection Module
# Detects the SM architecture of the local NVIDIA GPU
#
# Usage:
#   include(${CMAKE_CURRENT_LIST_DIR}/cmake/detect_gpu.cmake)
#   detect_gpu_arch(OUTPUT_VAR)
#   if(OUTPUT_VAR)
#       message(STATUS "Detected SM: ${OUTPUT_VAR}")
#   endif()

# Try to detect GPU architecture using various methods
function(detect_gpu_arch OUTPUT_VARIABLE)
    set(DETECTED_ARCH "")

    # Method 1: Use nvidia-smi to get compute capability
    find_program(NVIDIA_SMI_EXECUTABLE nvidia-smi)
    if(NVIDIA_SMI_EXECUTABLE)
        # Try to get compute capability from nvidia-smi
        execute_process(
            COMMAND ${NVIDIA_SMI_EXECUTABLE} --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE NVIDIA_SMI_OUTPUT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE NVIDIA_SMI_RESULT
        )
        if(NVIDIA_SMI_RESULT EQUAL 0 AND NVIDIA_SMI_OUTPUT)
            # Parse compute capability (e.g., "8.0" -> "SM80")
            string(REGEX MATCH "^([0-9]+)\\.([0-9]+)" COMPUTE_CAP "${NVIDIA_SMI_OUTPUT}")
            if(COMPUTE_CAP)
                string(REGEX REPLACE "^([0-9]+)\\.([0-9]+).*" "\\1" CAP_MAJOR "${COMPUTE_CAP}")
                string(REGEX REPLACE "^([0-9]+)\\.([0-9]+).*" "\\2" CAP_MINOR "${COMPUTE_CAP}")

                # Map to our supported architectures
                if(CAP_MAJOR EQUAL 8)
                    set(DETECTED_ARCH "SM80")
                elseif(CAP_MAJOR EQUAL 9)
                    set(DETECTED_ARCH "SM90")
                elseif(CAP_MAJOR EQUAL 10)
                    set(DETECTED_ARCH "SM100")
                elseif(CAP_MAJOR GREATER 10)
                    set(DETECTED_ARCH "SM100")
                endif()

                if(DETECTED_ARCH)
                    message(STATUS "Detected GPU compute capability: ${CAP_MAJOR}.${CAP_MINOR} -> ${DETECTED_ARCH}")
                endif()
            endif()
        endif()
    endif()

    # Method 2: Use nvcc to detect architecture
    if(NOT DETECTED_ARCH AND CUDAToolkit_FOUND)
        execute_process(
            COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} --list-gpu-code
            OUTPUT_VARIABLE NVCC_GPU_CODES
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE NVCC_RESULT
        )
        if(NVCC_RESULT EQUAL 0 AND NVCC_GPU_CODES)
            # Find the highest supported architecture
            set(HIGHEST_ARCH 0)
            foreach(GPU_CODE ${NVCC_GPU_CODES})
                # Extract sm_XX from strings like "sm_80"
                string(REGEX MATCH "sm_([0-9]+)" GPU_MATCH "${GPU_CODE}")
                if(GPU_MATCH)
                    string(REGEX REPLACE "sm_([0-9]+)" "\\1" GPU_NUM "${GPU_CODE}")
                    if(GPU_NUM GREATER HIGHEST_ARCH)
                        set(HIGHEST_ARCH ${GPU_NUM})
                    endif()
                endif()
            endforeach()

            # Map highest detected to our supported categories
            if(HIGHEST_ARCH GREATER_EQUAL 100)
                set(DETECTED_ARCH "SM100")
            elseif(HIGHEST_ARCH GREATER_EQUAL 90)
                set(DETECTED_ARCH "SM90")
            elseif(HIGHEST_ARCH GREATER_EQUAL 80)
                set(DETECTED_ARCH "SM80")
            endif()

            if(DETECTED_ARCH)
                message(STATUS "Detected GPU via nvcc: sm_${HIGHEST_ARCH} -> ${DETECTED_ARCH}")
            endif()
        endif()
    endif()

    # Method 3: Check CUDA_ARCH environment variable
    if(NOT DETECTED_ARCH)
        set(CUDA_ARCH_FROM_ENV "$ENV{CUDA_ARCH}")
        if(CUDA_ARCH_FROM_ENV)
            if(CUDA_ARCH_FROM_ENV GREATER_EQUAL 100)
                set(DETECTED_ARCH "SM100")
            elseif(CUDA_ARCH_FROM_ENV GREATER_EQUAL 90)
                set(DETECTED_ARCH "SM90")
            elseif(CUDA_ARCH_FROM_ENV GREATER_EQUAL 80)
                set(DETECTED_ARCH "SM80")
            endif()
            if(DETECTED_ARCH)
                message(STATUS "Detected GPU from CUDA_ARCH environment: ${DETECTED_ARCH}")
            endif()
        endif()
    endif()

    set(${OUTPUT_VARIABLE} ${DETECTED_ARCH} PARENT_SCOPE)
endfunction()
