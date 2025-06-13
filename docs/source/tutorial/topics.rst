

Topics to Cover
=====================

- **Cooperative Groups**  
  Synchronization and data sharing across thread blocks beyond `__syncthreads()`.

- **CUDA Cooperative Kernel Launch**  
  Kernels that synchronize across thread blocks.

- **Warp-Level Primitives**  
  Intrinsics like `__shfl_sync()`, `__ballot_sync()`, `__any_sync()` for warp-wide communication without shared memory.

- **Texture and Surface Memory**  
  Specialized memory for 2D/3D spatial locality; useful in image processing and rendering.

- **Multi-GPU Programming**  
  Peer-to-peer GPU memory access; using CUDA-aware MPI or NCCL for multi-GPU communication.

- **CUDA Libraries and Frameworks**  
  Using high-performance libraries like cuBLAS, cuFFT, cuDNN and integrating them.

- **Low-Level PTX and Inline Assembly**  
  Writing or inspecting PTX assembly; inline PTX usage for fine control and optimization.

- **CUDA Profiler and Nsight Tools**  
  Tools like Nsight Compute and Nsight Systems for profiling and bottleneck analysis.
