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

    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);
    for (int i = 0; i < N; ++i) h_input[i] = (float)i;

    float *d_data;
    check(cudaMalloc(&d_data, size), "Alloc d_data");

    cudaStream_t stream;
    check(cudaStreamCreate(&stream), "Create stream");

    // Create CUDA graph
    cudaGraph_t graph;
    check(cudaGraphCreate(&graph, 0), "Create graph");

    // Prepare memset node params
    cudaMemsetParams memsetParams = {};
    memsetParams.dst = d_data;
    memsetParams.value = 0;
    memsetParams.pitch = 0;
    memsetParams.elementSize = sizeof(float);
    memsetParams.width = N;
    memsetParams.height = 1;

    cudaGraphNode_t memsetNode, memcpyH2DNode, kernelNode, memcpyD2HNode;

    // Add memset node (no dependencies)
    check(cudaGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &memsetParams), "Add memset");

    // Prepare memcpy H2D params
    cudaMemcpy3DParms copyH2D = {};
    copyH2D.srcPtr = make_cudaPitchedPtr(h_input, size, N, 1);
    copyH2D.dstPtr = make_cudaPitchedPtr(d_data, size, N, 1);
    copyH2D.extent = make_cudaExtent(size, 1, 1);
    copyH2D.kind = cudaMemcpyHostToDevice;

    check(cudaGraphAddMemcpyNode(&memcpyH2DNode, graph, NULL, 0, &copyH2D), "Add memcpy H2D");

    // Prepare kernel node params
    int n = N;  // Non-const copy for kernel argument
    void* kernelArgs[] = { &d_data, &n };
    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)square;
    kernelParams.gridDim = dim3((N + 255) / 256);
    kernelParams.blockDim = dim3(256);
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = NULL;

    check(cudaGraphAddKernelNode(&kernelNode, graph, NULL, 0, &kernelParams), "Add kernel");

    // Prepare memcpy D2H params
    cudaMemcpy3DParms copyD2H = {};
    copyD2H.srcPtr = make_cudaPitchedPtr(d_data, size, N, 1);
    copyD2H.dstPtr = make_cudaPitchedPtr(h_output, size, N, 1);
    copyD2H.extent = make_cudaExtent(size, 1, 1);
    copyD2H.kind = cudaMemcpyDeviceToHost;

    check(cudaGraphAddMemcpyNode(&memcpyD2HNode, graph, NULL, 0, &copyD2H), "Add memcpy D2H");

    // Add dependencies: memset → memcpyH2D → kernel → memcpyD2H
    check(cudaGraphAddDependencies(graph, &memsetNode, &memcpyH2DNode, 1), "memset → H2D");
    check(cudaGraphAddDependencies(graph, &memcpyH2DNode, &kernelNode, 1), "H2D → kernel");
    check(cudaGraphAddDependencies(graph, &kernelNode, &memcpyD2HNode, 1), "kernel → D2H");

    // Instantiate and launch
    cudaGraphExec_t graphExec;
    check(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0), "Instantiate graph");
    check(cudaGraphLaunch(graphExec, stream), "Launch graph");
    check(cudaStreamSynchronize(stream), "Synchronize stream");

    printf("Result sample: h_output[10] = %f\n", h_output[10]);  // Should print 100 (10*10)

    // Cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_data);
    free(h_input);
    free(h_output);

    return 0;
}
