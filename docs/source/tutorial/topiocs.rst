- **Streams and Concurrency**  
  Using multiple CUDA streams for overlapping kernel execution and memory transfers.  
  Understanding concurrency and synchronization between kernels.  
  Techniques like CUDA Graphs to optimize task dependencies.

- **Unified Memory (Managed Memory)**  
  Using `cudaMallocManaged` for CPU/GPU accessible memory.  
  Page migration, prefetching (`cudaMemPrefetchAsync`), and memory advice (`cudaMemAdvise`).

- **Dynamic Parallelism**  
  Launching kernels from kernels (nested parallelism).  
  Useful for adaptive algorithms and irregular workloads.

- **Cooperative Groups**  
  Synchronization and data sharing across thread blocks beyond `__syncthreads()`.

- **Warp-Level Primitives**  
  Intrinsics like `__shfl_sync()`, `__ballot_sync()`, `__any_sync()` for warp-wide communication without shared memory.

- **Asynchronous Memory Copy**  
  Using pinned host memory and overlapping DMA transfers with kernel execution.

- **CUDA Graphs**  
  Defining and launching graphs of kernels and memory operations to reduce overhead.

- **Texture and Surface Memory**  
  Specialized memory for 2D/3D spatial locality; useful in image processing and rendering.

- **Streams and Event Profiling**  
  Profiling GPU execution with CUDA events for timing and debugging.

- **Occupancy and Performance Tuning**  
  Tuning thread/block/grid sizes for optimal occupancy; using occupancy calculators.

- **CUDA Cooperative Kernel Launch**  
  Kernels that synchronize across thread blocks.

- **Multi-GPU Programming**  
  Peer-to-peer GPU memory access; using CUDA-aware MPI or NCCL for multi-GPU communication.

- **CUDA Libraries and Frameworks**  
  Using high-performance libraries like cuBLAS, cuFFT, cuDNN and integrating them.

- **Low-Level PTX and Inline Assembly**  
  Writing or inspecting PTX assembly; inline PTX usage for fine control and optimization.

- **CUDA Profiler and Nsight Tools**  
  Tools like Nsight Compute and Nsight Systems for profiling and bottleneck analysis.
