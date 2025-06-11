CUDA Graphs
====================

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 20 min

    #. Understand the concept of CUDA graphs.
    #. Learn how to create and launch CUDA graphs.
    #. Explore the advantages and use cases of CUDA graphs.

CUDA graphs are a powerful feature in CUDA that allow you to capture a sequence of operations (kernels, memory
copies, etc.) into a graph structure. This enables more efficient execution by reducing the overhead of 
launching individual kernels and memory operations. CUDA graphs can significantly improve performance for 
applications with complex workflows or repetitive patterns.

CUDA Graph Basics
----------------------------

Usuall, every kernel launch or memory copy call from CPU to GPU incurs overhead. Repeated sequences 
(e.g., training steps in deep learning) can be inefficient due to this overhead. With CUDA Graphs
you record the entire sequence once, and replay it as many times as needed with minimal CPU interaction.

Components of CUDA Graphs
----------------------------

CUDA graphs consist of several components:


1. Nodes: Each node represents an operation:
    * Kernel launch
    * Memory copy
    * Memory set
    * Host function call

2. Edges: Dependencies between nodes (execution order)
3. Graph (``cudaGraph_t``): The entire captured DAG (Directed Acyclic Graph)
4. GraphExec (``cudaGraphExec_t``): An executable version of the graph

There are some key limitations to keep in mind:

* Graphs are static: You cannot change their structure after instantiation (though CUDA 12 introduced some dynamic features).
* Capturing must happen in a single stream.
* Proper synchronization and stream management are crucial.
* Debugging can be harder than traditional launches.


.. code-block:: c
    :linenos:

    #include <stdio.h>
    #include <cuda_runtime.h>

    // CUDA kernel: simple vector addition
    __global__ void vectorAdd(const float *A, const float *B, float *C, int N) 
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
            C[i] = A[i] + B[i];
    }

    int main() 
    {
        const int N = 1 << 16;
        const int size = N * sizeof(float);
    
        float *h_A = (float*)malloc(size);
        float *h_B = (float*)malloc(size);
        float *h_C = (float*)malloc(size);
    
        // Initialize host input arrays
        for (int i = 0; i < N; ++i) {
            h_A[i] = i;
            h_B[i] = 2 * i;
        }
    
        float *d_A, *d_B, *d_C;
        cudaMalloc((void**)&d_A, size);
        cudaMalloc((void**)&d_B, size);
        cudaMalloc((void**)&d_C, size);
    
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
        cudaStream_t stream;
        cudaStreamCreate(&stream);
    
        // Begin CUDA Graph capture
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    
        // Kernel launch (recorded)
        vectorAdd<<<(N + 255)/256, 256, 0, stream>>>(d_A, d_B, d_C, N);
    
        // Copy result to host (recorded)
        cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream);
    
        // End capture
        cudaStreamEndCapture(stream, &graph);
    
        // Instantiate executable graph
        cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    
        // Execute the graph multiple times
        for (int i = 0; i < 5; ++i) {
            cudaGraphLaunch(graphExec, stream);
            cudaStreamSynchronize(stream);
            printf("Run %d: C[100] = %f\n", i, h_C[100]);  // Should be 300 (100 + 200)
        }
    
        // Clean up
        cudaGraphExecDestroy(graphExec);
        cudaGraphDestroy(graph);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C);
        cudaStreamDestroy(stream);
    
        return 0;
    }

