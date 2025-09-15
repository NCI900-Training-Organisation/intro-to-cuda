GPU Workflow
========================================

.. admonition:: Overview
   :class: Overview

    * **Time:** 60 min

    #. Learn the workflow for programming GPUs using CUDA.
    #. Understand the key steps involved in GPU programming.


The workflow for programming GPUs using CUDA involves several key steps that are essential for efficient execution of parallel tasks. Below is a high-level overview of the GPU workflow:

1. **Kernel Definition**: 

   - Define a CUDA kernel using the `__global__` keyword. This kernel will be executed on the GPU.

   - The kernel contains the code that will run on the GPU, typically involving parallel computations.

2. **Memory Management**:

   - Allocate memory on the GPU using functions like `cudaMalloc()`.

   - Copy data from the host (CPU) to the device (GPU) using `cudaMemcpy()`.

   - Ensure that memory is properly managed to avoid leaks and ensure efficient access.

3. **Kernel Launch**:

   - Launch the kernel using the `<<<grid, block>>>` syntax, where `grid` specifies the number of blocks and `block` specifies the number of threads per block.

   - Each thread executes the kernel code independently, allowing for parallel execution.

4. **Thread Indexing**:

   - Use built-in variables like `threadIdx`, `blockIdx`, and `blockDim` to determine the unique index of each thread.

   - This indexing allows each thread to operate on different data elements, enabling parallel processing.

5. **Synchronization**:

   - Use synchronization functions like `__syncthreads()` to ensure that all threads in a block have completed their tasks before proceeding.

   - This is important for operations that require data consistency among threads.

6. **Memory Cleanup**:

   - Free the allocated memory on the GPU using `cudaFree()`.

   - Ensure that all resources are properly released to avoid memory leaks.

7. **Error Handling**:

   - Implement error handling to check for issues during memory allocation, kernel execution, and data transfer.

   - Use functions like `cudaGetLastError()` to retrieve error codes and handle exceptions appropriately.
   

Kernel Definition
-----------------

.. code-block:: cpp

   __global__ void add_vectors(float *a, float *b, float *c, int n) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;

       if (idx < n)
           c[idx] = a[idx] + b[idx];
   }


#. ``__global__`` : Indicates that this function is a kernel that runs on the GPU.
#. ``void`` : CUDA kernels **must return void**
#. Parameters: Pointers, that are passed to a kernel must reference **device memory**.
#. Thread Indexing:  Each thread should have a unique index and can access different elements of the data (input arrays).


Memory Management
-----------------

Memory management in CUDA mainly involves four key operations: 

#. Device memory allocaation.
#. Host to Device (H2D) memory copy.
#. Device to Host (D2H) memory copy.
#. Device memory deallocation.

Proper memory management is crucial for performance and avoiding memory leaks.



``cudaMalloc`` is used to allocate memory on the GPU.

.. code-block:: cpp

   float *d_a, *d_b, *d_c;
   int n = 1024;

   // Allocate memory on the GPU
   cudaMalloc((void**)&d_a, n * sizeof(float));
   cudaMalloc((void**)&d_b, n * sizeof(float));
   cudaMalloc((void**)&d_c, n * sizeof(float));

The above code allocates memory on the GPU for three float arrays, each of size `n`. Data transfer between the host (CPU) and device (GPU) is done using `cudaMemcpy`.


.. code-block:: cpp

   // Copy data from host to device
   cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(d_c, h_c, n * sizeof(float), cudaMemcpyHostToDevice);


The above code copies data from host arrays `h_a`, `h_b`, and `h_c` to device arrays `d_a`, `d_b`, and `d_c`. ``cudaMemcpyHostToDevice`` specifies the direction of the copy operation, indicating that data is being transferred from host memory to device memory.

.. code-block:: cpp

   // Copy data from device to host
   cudaMemcpy(h_a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost);
   cudaMemcpy(h_b, d_b, n * sizeof(float), cudaMemcpyDeviceToHost);
   cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

The above code copies data back from device arrays `d_a`, `d_b`, and `d_c` to host arrays `h_a`, `h_b`, and `h_c`. ``cudaMemcpyDeviceToHost`` specifies that data is being transferred from device memory back to host memory.

Finally, it is important to free the allocated memory on the GPU, after kernel execution, to avoid memory leaks:


.. code-block:: cpp

   // Free device memory
   cudaFree(d_a);
   cudaFree(d_b);
   cudaFree(d_c);

This code releases the memory allocated on the GPU for the arrays `d_a`, `d_b`, and `d_c`.

The complete code will look like this:  

.. code-block:: cpp

   #include <stdio.h>
   #include <cuda_runtime.h> // Provides access to CUDA runtime API functions

 
   __global__ void add_vectors(float *a, float *b, float *c, int n) 
   {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;

       if (idx < n)
           c[idx] = a[idx] + b[idx];
   }

   int main() 
   {
       int n = 1024;
       float *h_a, *h_b, *h_c;
       float *d_a, *d_b, *d_c;

       // Allocate host memory
       h_a = (float*)malloc(n * sizeof(float));
       h_b = (float*)malloc(n * sizeof(float));
       h_c = (float*)malloc(n * sizeof(float));

       // Initialize host arrays
       for (int i = 0; i < n; i++) {
           h_a[i] = i;
           h_b[i] = i;
       }

       // Allocate device memory
       cudaMalloc((void**)&d_a, n * sizeof(float));
       cudaMalloc((void**)&d_b, n * sizeof(float));
       cudaMalloc((void**)&d_c, n * sizeof(float));

       // Copy data from host to device
       cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
       cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

       // Launch kernel with 256 threads per block and enough blocks to cover all elements
       add_vectors<<<(n + 255) / 256, 256>>>(d_a, d_b, d_c, n);

       // Copy result back to host
       cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

       // Free device memory
       cudaFree(d_a);
       cudaFree(d_b);
       cudaFree(d_c);

       // Free host memory
       free(h_a);
       free(h_b);
       free(h_c);

       return 0;
   }


In the above code ``cudaMalloc`` and ``cudaMemcpy`` are both is a synchronous call — it blocks until the copy is finished and all prior device work is complete.
Kernel launches are asynchronous, meaning they return immediately and the CPU can continue executing code while the GPU processes the kernel. But in this case, there is an implicit synchronozation beacuse we are using a default steam (will be discussed later).

A better code will look like this:

.. code-block:: cpp

    #include <stdio.h>
    #include <cuda_runtime.h>

    __global__ void add_vectors(float *a, float *b, float *c, int n) 
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
            c[idx] = a[idx] + b[idx];
    }

    int main() 
    {
        int n = 1024;
        float *h_a, *h_b, *h_c;
        float *d_a, *d_b, *d_c;

        // Allocate host memory
        h_a = (float*)malloc(n * sizeof(float));
        h_b = (float*)malloc(n * sizeof(float));
        h_c = (float*)malloc(n * sizeof(float));

        // Initialize host arrays
        for (int i = 0; i < n; i++) {
            h_a[i] = i;
            h_b[i] = i;
        }

        // Allocate device memory with error checks
        if (cudaMalloc((void**)&d_a, n * sizeof(float)) != cudaSuccess) {
            fprintf(stderr, "Error allocating device memory for d_a\n");
            return -1;
        }

        if (cudaMalloc((void**)&d_b, n * sizeof(float)) != cudaSuccess) {
            fprintf(stderr, "Error allocating device memory for d_b\n");
            cudaFree(d_a);
            return -1;
        }

        if (cudaMalloc((void**)&d_c, n * sizeof(float)) != cudaSuccess) {
            fprintf(stderr, "Error allocating device memory for d_c\n");
            cudaFree(d_a);
            cudaFree(d_b);
            return -1;
        }

        // Copy data from host to device
        if (cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            fprintf(stderr, "Error copying h_a to d_a\n");
            return -1;
        }

        if (cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            fprintf(stderr, "Error copying h_b to d_b\n");
            return -1;
        }

        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
        add_vectors<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
            return -1;
        }

        // Ensure kernel has completed
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
            return -1;
        }

        // Copy result back to host
        if (cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            fprintf(stderr, "Error copying d_c to h_c\n");
            return -1;
        }

     
        // Free device memory
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        // Free host memory
        free(h_a);
        free(h_b);
        free(h_c);

        return 0;
    }

.. admonition:: Explanation
    :class: attention

    ``cudaDeviceSynchronize()`` function blocks the host (CPU) until all previously issued commands on the device (GPU) are complete.


Common CUDA Error Codes
-------------------------
CUDA provides a set of error codes to help developers identify issues during GPU programming. These error codes are returned by CUDA API functions and can be checked to ensure that operations are successful. Below is a list of common CUDA error codes along with their meanings:

.. list-table:: Common CUDA Error Codes
   :header-rows: 1
   :widths: 25 10 65

   * - Constant
     - Value
     - Meaning
   * - ``cudaSuccess``
     - 0
     - Operation completed successfully.
   * - ``cudaErrorMemoryAllocation``
     - 2
     - Memory allocation failed (e.g., in ``cudaMalloc``).
   * - ``cudaErrorInvalidValue``
     - 11
     - Invalid parameter passed to a CUDA function.
   * - ``cudaErrorInvalidDevicePointer``
     - 17
     - Device pointer is invalid.
   * - ``cudaErrorInvalidMemcpyDirection``
     - 21
     - Direction passed to ``cudaMemcpy`` is not valid.
   * - ``cudaErrorLaunchFailure``
     - 4
     - Kernel launch failed for an unspecified reason.
   * - ``cudaErrorInvalidConfiguration``
     - 9
     - Invalid block size or grid size in kernel launch.
   * - ``cudaErrorLaunchTimeout``
     - 6
     - Kernel execution took too long (often on Windows with WDDM).
   * - ``cudaErrorUnknown``
     - 30
     - Unknown error occurred.



.. admonition:: Key Points
   :class: hint

    #. The GPU workflow involves defining kernels, managing memory, launching kernels, and synchronizing threads.
    #. Proper memory management is crucial for performance and avoiding leaks.
    #. Thread indexing is essential for parallel execution, allowing each thread to work on different data elements.
    #. Synchronization ensures that threads complete their tasks before proceeding, maintaining data consistency.
    #. Error handling is important to catch issues during execution and ensure robustness of the code.

