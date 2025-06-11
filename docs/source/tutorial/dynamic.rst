
Dynamic Parallelism in CUDA
===============================

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 20 min

    
    #. Understand the concept of dynamic parallelism in CUDA.
    #. Learn how to implement nested kernel launches on the GPU.
    #. Explore the advantages and use cases of dynamic parallelism.




Dynamic parallelism allows a CUDA kernel running on the GPU to launch additional kernels **from the device itself**, without needing to return control to the 
CPU. This enables **nested parallelism** directly on the GPU, allowing for more flexible and adaptive computation models.

Advantages of Dynamic Parallelism:

1. GPU Can Launch Work Without CPU Involvement

    * Traditional CUDA: All kernel launches must come from the host (CPU).
    * Dynamic Parallelism: GPU can launch additional kernels internally.
    * **Result**: Eliminates the need to return to the CPU to schedule new GPU work, saving time and reducing latency.

2. Better for Irregular or Recursive Workloads

Dynamic parallelism is especially useful when:

    * The **amount of work is not known in advance**.
    * Computation patterns depend on **data-dependent branching**.
    * Examples include:
        - Graph traversal
        - Tree-based algorithms
        - Adaptive mesh refinement
        - Sparse linear algebra
        - N-body simulations

3. More Natural Expression of Recursive or Hierarchical Algorithms

Many algorithms are naturally recursive or hierarchical in nature:

* Quicksort
* Depth-first search
* Octree or quadtree traversal

Dynamic parallelism allows you to implement these algorithms **cleanly**, without flattening them into an iterative model.

4. Reduced CPU-GPU Synchronization Overhead

- Avoids unnecessary memory transfers and sync points.
- Allows decision logic to remain on the GPU, improving efficiency.



.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Feature
     - Benefit
   * - Nested kernel launches
     - Enables on-demand dynamic parallelism
   * - No CPU sync required
     - Lowers latency and avoids host-device transfer overhead
   * - Supports irregular workloads
     - Great for graphs, trees, adaptive data structures
   * - Recursive-friendly
     - Natural expression of hierarchical logic
   * - More GPU autonomy
     - Allows more decisions to be made directly on the device


Compilation Requirements
-----------------------------

To use dynamic parallelism:

* GPU must support compute capability **≥ 3.5**.
* Use the following NVCC flags:

.. code-block:: bash

    nvcc -arch=sm_35 -rdc=true -o program program.cu

* ``-rdc=true`` enables Relocatable Device Code, required for device-side kernel launches.

Code Example
-----------------

.. code-block:: cpp

    #include <stdio.h>
    #include <cuda_runtime.h>

    __global__ void childKernel(int *data, int n) 
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] += 1;
        }
    }

    __global__ void parentKernel(int *data, int n) 
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        childKernel<<<blocks, threads>>>(data, n);
        cudaDeviceSynchronize();
    }

    int main() 
    {
        const int N = 1024;
        size_t size = N * sizeof(int);

        int *h_data = (int *)malloc(size);
        for (int i = 0; i < N; ++i) {
            h_data[i] = i;
        }

        int *d_data;
        cudaMalloc(&d_data, size);
        cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

        parentKernel<<<1, 1>>>(d_data, N);
        cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

        for (int i = 0; i < 10; ++i) {
            printf("h_data[%d] = %d\n", i, h_data[i]);
        }

        cudaFree(d_data);
        free(h_data);
        return 0;
    }


.. admonition:: Key Points
   :class: hint
   
    #. Dynamic parallelism allows kernels to launch other kernels from the device.
    #. It is useful for irregular workloads and recursive algorithms.
    #. Requires compute capability ≥ 3.5 and specific NVCC flags.
    #. Reduces CPU-GPU synchronization overhead, improving performance.

