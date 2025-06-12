#include <stdio.h>
#include <cuda_runtime.h>

__global__ void square(float *d_data, int N) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
        d_data[idx] = d_data[idx] * d_data[idx];
}

void check(cudaError_t err, const char* msg) 
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() 
{
    const int N = 1 << 16;
    const int size = N * sizeof(float);

    // Host buffers
    float *h_input[3], *h_output[3];
    for (int i = 0; i < 3; ++i) {
        h_input[i] = (float*)malloc(size);
        h_output[i] = (float*)malloc(size);
        for (int j = 0; j < N; ++j)
            h_input[i][j] = (float)(j + i * 1000);  // Different input data
    }

    // Device buffers
    float *d_data[3];
    for (int i = 0; i < 3; ++i) {
        check(cudaMalloc(&d_data[i], size), "Alloc d_data");
    }

    cudaStream_t stream;
    check(cudaStreamCreate(&stream), "Create stream");

    // Create graph
    cudaGraph_t graph;
    check(cudaGraphCreate(&graph, 0), "Create graph");

    // Memset node for d_data[0] initially (just to fill with zero)
    cudaMemsetParams memsetParams = {};
    memsetParams.dst = d_data[0];
    memsetParams.value = 0;
    memsetParams.pitch = 0;
    memsetParams.elementSize = sizeof(float);
    memsetParams.width = N;
    memsetParams.height = 1;
    cudaGraphNode_t memsetNode;
    check(cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams), "Add memset");

    // Memcpy H2D node for d_data[0]
    cudaMemcpy3DParms copyH2D = {};
    copyH2D.srcPtr = make_cudaPitchedPtr(h_input[0], size, N, 1);
    copyH2D.dstPtr = make_cudaPitchedPtr(d_data[0], size, N, 1);
    copyH2D.extent = make_cudaExtent(size, 1, 1);
    copyH2D.kind = cudaMemcpyHostToDevice;
    cudaGraphNode_t memcpyH2DNode;
    check(cudaGraphAddMemcpyNode(&memcpyH2DNode, graph, NULL, 0, &copyH2D), "Add memcpy H2D");

    // Kernel node — dummy args for now, will update dynamically
    int n = N;
    void* kernelArgs[] = { &d_data[0], &n };
    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)square;
    kernelParams.gridDim = dim3((N + 255) / 256);
    kernelParams.blockDim = dim3(256);
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = NULL;
    cudaGraphNode_t kernelNode;
    check(cudaGraphAddKernelNode(&kernelNode, graph, NULL, 0, &kernelParams), "Add kernel");

    // Memcpy D2H node for d_data[0]
    cudaMemcpy3DParms copyD2H = {};
    copyD2H.srcPtr = make_cudaPitchedPtr(d_data[0], size, N, 1);
    copyD2H.dstPtr = make_cudaPitchedPtr(h_output[0], size, N, 1);
    copyD2H.extent = make_cudaExtent(size, 1, 1);
    copyD2H.kind = cudaMemcpyDeviceToHost;
    cudaGraphNode_t memcpyD2HNode;
    check(cudaGraphAddMemcpyNode(&memcpyD2HNode, graph, NULL, 0, &copyD2H), "Add memcpy D2H");

    // Dependencies: memset -> memcpyH2D -> kernel -> memcpyD2H
    check(cudaGraphAddDependencies(graph, &memsetNode, &memcpyH2DNode, 1), "memset -> memcpyH2D");
    check(cudaGraphAddDependencies(graph, &memcpyH2DNode, &kernelNode, 1), "memcpyH2D -> kernel");
    check(cudaGraphAddDependencies(graph, &kernelNode, &memcpyD2HNode, 1), "kernel -> memcpyD2H");

    // Instantiate graph exec
    cudaGraphExec_t graphExec;
    check(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0), "Instantiate graph");

    // Now loop for all 3 data sets, update kernel and memcpy nodes dynamically and launch
    for (int i = 0; i < 3; ++i) {
        // Update memcpy H2D node params
        copyH2D.srcPtr = make_cudaPitchedPtr(h_input[i], size, N, 1);
        copyH2D.dstPtr = make_cudaPitchedPtr(d_data[i], size, N, 1);
        check(cudaGraphExecMemcpyNodeSetParams(graphExec, memcpyH2DNode, &copyH2D), "Update memcpy H2D params");

        // Update kernel node params
        int n = N;
        void* kernelArgsNew[] = { &d_data[i], &n };
        cudaKernelNodeParams kernelParamsNew = {};
        kernelParamsNew.func = (void*)square;
        kernelParamsNew.gridDim = dim3((N + 255) / 256);
        kernelParamsNew.blockDim = dim3(256);
        kernelParamsNew.kernelParams = kernelArgsNew;
        kernelParamsNew.extra = NULL;
        check(cudaGraphExecKernelNodeSetParams(graphExec, kernelNode, &kernelParamsNew), "Update kernel params");

        // Update memcpy D2H node params
        copyD2H.srcPtr = make_cudaPitchedPtr(d_data[i], size, N, 1);
        copyD2H.dstPtr = make_cudaPitchedPtr(h_output[i], size, N, 1);
        check(cudaGraphExecMemcpyNodeSetParams(graphExec, memcpyD2HNode, &copyD2H), "Update memcpy D2H params");

        // Launch graph
        check(cudaGraphLaunch(graphExec, stream), "Launch graph");
        check(cudaStreamSynchronize(stream), "Sync stream");

        // Check results
        printf("Result sample for buffer %d: h_output[%d] = %f\n", i, 10, h_output[i][10]);
    }

    // Cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    for (int i = 0; i < 3; ++i) {
        cudaFree(d_data[i]);
        free(h_input[i]);
        free(h_output[i]);
    }

    return 0;
}
