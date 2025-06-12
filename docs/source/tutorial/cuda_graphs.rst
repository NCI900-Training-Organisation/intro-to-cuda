CUDA Graphs
====================

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 60 min

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
can be inefficient due to this overhead. With CUDA Graphs you record the entire sequence once, and 
replay it as many times as needed with minimal CPU interaction.

Components of CUDA Graphs
----------------------------

CUDA graphs consist of several components:


1. Nodes: Each node represents an operation:
    * Kernel launch
    * Memory copy
    * Memory set
    * Host function call etc.

2. Edges: Dependencies between nodes (execution order)
3. Graph (``cudaGraph_t``): The entire captured DAG (Directed Acyclic Graph)
4. GraphExec (``cudaGraphExec_t``): An executable version of the graph

There are some key limitations to keep in mind:

* Graphs are static: You cannot change their structure after instantiation (though CUDA 12 introduced some dynamic features).
* Capturing must happen in a single stream.
* Proper synchronization and stream management are crucial.
* Debugging can be harder than traditional launches.



The steps to create and launch a CUDA graph are as follows:

Create CUDA Stream:

.. code-block:: c
    :linenos:

    cudaStream_t stream;
    cudaStreamCreate(&stream);

* Creates a CUDA stream — an independent queue of GPU operations allowing asynchronous and ordered execution.

 Begin CUDA Graph Capture:

.. code-block:: c
    :linenos:

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

* Starts capturing all GPU operations submitted to stream.
* cudaStreamCaptureModeGlobal means all GPU work in the stream is recorded as graph nodes.

Record each operation as a node in the graph:

.. code-block:: c
    :linenos:

    cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream); 
    cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream);

    vectorAdd<<<(N + 255)/256, 256, 0, stream>>>(d_A, d_B, d_C, N);

    cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream);


This creates a foure nodes in the graph:
* Two H2D memory  copy nodes 
* One kernel launch node (vectorAdd)
* One D2H memory copy node 

End CUDA Graph Capture: 

.. code-block:: c
    :linenos:

    cudaStreamEndCapture(stream, &graph);

Instantiate the CUDA graph for execution

.. code-block:: c
    :linenos:

    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

* Creates an executable instance of the CUDA graph (graphExec) from the captured graph.

Launch the CUDA graph multiple times

.. code-block:: c
    :linenos:

    for (int i = 0; i < 5; ++i) {
        cudaGraphLaunch(graphExec, stream);
        cudaStreamSynchronize(stream);
        printf("Run %d: C[100] = %f\n", i, h_C[100]);  // Should be 300 (100 + 200)
    }

* Launches the entire graph (memcpy H2D → kernel → memcpy D2H) five times in a row.
* Synchronizes on the stream to ensure GPU finishes before reading the result.
* Prints element 100 of the output vector to verify correctness.

Finall destroy the graph and stream:

.. code-block:: c
    :linenos:

    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(graphExec);
    cudaStreamDestroy(stream);




Custom CUDA Graphs
----------------------------

Custom CUDA graphs allow you to define your own graph structures and operations, providing flexibility 
for advanced use cases. You can create custom nodes, edges, and even define your own execution logic.

We have the followiing program that demonstrates how to create a custom CUDA graph. We have a simple
kernel that:

* Squares each element of the device array d_data.
* Each thread computes its unique idx.
* Bounds check to avoid out-of-range access.

.. code-block:: c
    :linenos:

    __global__ void square(float *d_data, int N) 
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < N)
            d_data[idx] = d_data[idx] * d_data[idx];
        }
    }

We define the initial set of instruction that:

* Defines problem size (65536 floats).
* Allocates host memory for input and output.
* Initializes input array with values [0, 1, 2, ..., N-1].
* Allocates device memory for input/output array.
* Creates a CUDA stream for asynchronous execution.

.. code-block:: c
    :linenos:

    const int N = 1 << 16;  // 65536 elements
    const int size = N * sizeof(float);

    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);

    for (int i = 0; i < N; ++i) h_input[i] = (float)i;


    float *d_data;
    check(cudaMalloc(&d_data, size), "Alloc d_data");

    cudaStream_t stream;
    check(cudaStreamCreate(&stream), "Create stream");



Then we create an empty CUDA graph:

.. code-block:: c
    :linenos:

    cudaGraph_t graph;
    check(cudaGraphCreate(&graph, 0), "Create graph");

Now create the first node in the graph which is a memset operation:

.. code-block:: c
    :linenos:

    cudaMemsetParams memsetParams = {};
    memsetParams.dst = d_data;
    memsetParams.value = 0;
    memsetParams.pitch = 0;
    memsetParams.elementSize = sizeof(float);
    memsetParams.width = N;
    memsetParams.height = 1;

    cudaGraphNode_t memsetNode;
    check(cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams), "Add memset");


The we create a H2D copy node to copy the input data from host to device:

.. code-block:: c
    :linenos:

    cudaMemcpy3DParms copyH2D = {};
    copyH2D.srcPtr = make_cudaPitchedPtr(h_input, size, N, 1);
    copyH2D.dstPtr = make_cudaPitchedPtr(d_data, size, N, 1);
    copyH2D.extent = make_cudaExtent(size, 1, 1);
    copyH2D.kind = cudaMemcpyHostToDevice;

    cudaGraphNode_t memcpyH2DNode;
    check(cudaGraphAddMemcpyNode(&memcpyH2DNode, graph, NULL, 0, &copyH2D), "Add memcpy H2D");


Next we add a kernel node to launch the square kernel:

.. code-block:: c
    :linenos:

    int n = N;
    void* kernelArgs[] = { &d_data, &n };
    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)square;
    kernelParams.gridDim = dim3((N + 255) / 256);
    kernelParams.blockDim = dim3(256);
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = NULL;

    cudaGraphNode_t kernelNode;
    check(cudaGraphAddKernelNode(&kernelNode, graph, NULL, 0, &kernelParams), "Add kernel");


Finally, we add a D2H copy node to copy the output data from device to host:

.. code-block:: c
    :linenos:

    cudaMemcpy3DParms copyD2H = {};
    copyD2H.srcPtr = make_cudaPitchedPtr(d_data, size, N, 1);
    copyD2H.dstPtr = make_cudaPitchedPtr(h_output, size, N, 1);
    copyD2H.extent = make_cudaExtent(size, 1, 1);
    copyD2H.kind = cudaMemcpyDeviceToHost;

    cudaGraphNode_t memcpyD2HNode;
    check(cudaGraphAddMemcpyNode(&memcpyD2HNode, graph, NULL, 0, &copyD2H), "Add memcpy D2H");

So we have created for nodes in the graph. Now we need to specify the dependencies between them:

.. code-block:: c
    :linenos:

    check(cudaGraphAddDependencies(graph, &memsetNode, &memcpyH2DNode, 1), "memset → H2D");
    check(cudaGraphAddDependencies(graph, &memcpyH2DNode, &kernelNode, 1), "H2D → kernel");
    check(cudaGraphAddDependencies(graph, &kernelNode, &memcpyD2HNode, 1), "kernel → D2H");

The above code specifed the execution order (DAG) of the nodes. 
* Memset must finish before Host→Device memcpy.
* Host→Device memcpy must finish before kernel launch.
* Kernel must finish before Device→Host memcpy.
Edges the nodes will be automatically created by the CUDA graph API.


Finally instantiate the graph and launch it:
.. code-block:: c
    :linenos:

    cudaGraphExec_t graphExec;
    check(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0), "Instantiate graph");
    check(cudaGraphLaunch(graphExec, stream), "Launch graph");
    check(cudaStreamSynchronize(stream), "Synchronize stream");




Updating node parameters
----------------------------

You can update node parameters in a CUDA graph after instantiation, allowing for dynamic behavior.
This mean you can run the same graph with different parameters without needing to recreate it.

To do this we can update memcpy H2D node parameters:

.. code-block:: c
    :linenos:

    copyH2D.srcPtr = make_cudaPitchedPtr(h_input[i], size, N, 1);
    copyH2D.dstPtr = make_cudaPitchedPtr(d_data[i], size, N, 1);
    check(cudaGraphExecMemcpyNodeSetParams(graphExec, memcpyH2DNode, &copyH2D), "Update memcpy H2D params");


Then we can launch the graph again with the updated parameters:

.. code-block:: c
    :linenos:

    int n = N;
    void* kernelArgsNew[] = { &d_data[i], &n };
    cudaKernelNodeParams kernelParamsNew = {};
    kernelParamsNew.func = (void*)square;
    kernelParamsNew.gridDim = dim3((N + 255) / 256);
    kernelParamsNew.blockDim = dim3(256);
    kernelParamsNew.kernelParams = kernelArgsNew;
    kernelParamsNew.extra = NULL;
    check(cudaGraphExecKernelNodeSetParams(graphExec, kernelNode, &kernelParamsNew), "Update kernel params");


Update memcpy D2H node parameters:

.. code-block:: c
    :linenos:

    copyD2H.srcPtr = make_cudaPitchedPtr(d_data[i], size, N, 1);
    copyD2H.dstPtr = make_cudaPitchedPtr(h_output[i], size, N, 1);
    check(cudaGraphExecMemcpyNodeSetParams(graphExec, memcpyD2HNode, &copyD2H), "Update memcpy D2H params");


Launch graph and synchronize:

 .. code-block:: c
    :linenos:

    check(cudaGraphLaunch(graphExec, stream), "Launch graph");
    check(cudaStreamSynchronize(stream), "Sync stream");


In In each iteration:

* The host-to-device memcpy parameters are updated to copy the ith input buffer to the ith device buffer.
* The kernel arguments are updated to point to the current device buffer.
* The device-to-host memcpy parameters are updated to copy results back to the ith output buffer.
* The graph is launched, performing the memset, memcpy, kernel, and memcpy operations with the updated parameters.

.. admonition:: Key Points
   :class: hint

    #. CUDA graphs capture a sequence of operations into a graph structure.
    #. They reduce overhead by allowing multiple operations to be launched as a single entity.
    #. Custom CUDA graphs allow for dynamic behavior and parameter updates.
    #. Proper synchronization and stream management are crucial for correct execution.
    #. CUDA graphs can significantly improve performance for complex workflows or repetitive patterns.