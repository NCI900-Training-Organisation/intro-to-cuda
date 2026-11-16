# CUDA exercises

Build all C/C++ and CUDA programs, including nested legacy solutions, with CMake
3.18 or newer. On Gadi, load the compiler, CUDA, MKL and an available CMake module,
then run from the repository root:

```bash
module load gcc/14.2.0 cuda/12.9.0 intel-mkl/2024.2.1
cmake -S exercises -B exercises/build -DCMAKE_BUILD_TYPE=Release
cmake --build exercises/build --parallel 4
```

Executables appear directly in `exercises/build/` with unique names matching
their CMake targets, such as
`ex_02_gpu_exec_model_double_global_index`, so you can build one program:

```bash
cmake --build exercises/build --target ex_02_gpu_exec_model_double_global_index
```

The PBS scripts use these uniquely named binaries directly in `exercises/build/`.

The default CUDA architecture is `70` for Gadi's V100 GPUs; override with
`-DCMAKE_CUDA_ARCHITECTURES=80`, for example, for an A100. Without MKL, add
`-DEXERCISES_BUILD_MKL=OFF` to skip only the CPU MKL benchmark. Python helpers
run directly and are not compiled. Existing per-folder Makefiles still work.

Folders 01–08 match the numbered notebooks. Advanced topics continue the sequence in documentation order. Source filenames describe their purpose; each existing Makefile builds its local sources.

To run any numbered exercise on Gadi without Jupyter, use the
[PBS job scripts](../job-script/README.md). For example, from the repository root:

```bash
cmake -S exercises -B exercises/build -DCMAKE_BUILD_TYPE=Release
cmake --build exercises/build --parallel 4
cd job-script
qsub 02-gpu-exec-model.pbs
```

Compile separately before submitting; the job scripts only run the prebuilt executables.

| Folder | Notebook | Documentation |
| --- | --- | --- |
| [01-gpu](01-gpu/) | [01-gpu.ipynb](../notebooks/01-gpu.ipynb) | [GPU Architecture](../docs/source/tutorial/exercises/01-gpu.rst) |
| [02-gpu-exec-model](02-gpu-exec-model/) | [02-gpu-exec-model.ipynb](../notebooks/02-gpu-exec-model.ipynb) | [GPU Execution Model](../docs/source/tutorial/exercises/02-gpu-exec-model.rst) |
| [03-gpu-workflow](03-gpu-workflow/) | [03-gpu-workflow.ipynb](../notebooks/03-gpu-workflow.ipynb) | [GPU Workflow](../docs/source/tutorial/exercises/03-gpu-workflow.rst) |
| [04-warps](04-warps/) | [04-warps.ipynb](../notebooks/04-warps.ipynb) | [Warps](../docs/source/tutorial/exercises/04-warps.rst) |
| [05-asynchronous](05-asynchronous/) | [05-asynchronous.ipynb](../notebooks/05-asynchronous.ipynb) | [Asynchronous CUDA Calls](../docs/source/tutorial/exercises/05-asynchronous.rst) |
| [06-shared-memory](06-shared-memory/) | [06-shared-memory.ipynb](../notebooks/06-shared-memory.ipynb) | [Shared Memory](../docs/source/tutorial/exercises/06-shared-memory.rst) |
| [07-events](07-events/) | [07-events.ipynb](../notebooks/07-events.ipynb) | [CUDA Events](../docs/source/tutorial/exercises/07-events.rst) |
| [08-unified-memory](08-unified-memory/) | [08-unified-memory.ipynb](../notebooks/08-unified-memory.ipynb) | [Unified Memory](../docs/source/tutorial/exercises/08-unified-memory.rst) |
| [09-dynamic-parallelism](09-dynamic-parallelism/) | — | [Dynamic Parallelism](../docs/source/tutorial/exercises/09-dynamic-parallelism.rst) |
| [10-occupancy](10-occupancy/) | — | [Occupancy](../docs/source/tutorial/exercises/10-occupancy.rst) |
| [11-cuda-graphs](11-cuda-graphs/) | — | [CUDA Graphs](../docs/source/tutorial/exercises/11-cuda-graphs.rst) |
| [12-memory-pools](12-memory-pools/) | — | [Memory Pools](../docs/source/tutorial/exercises/12-memory-pools.rst) |
| [13-cublas](13-cublas/) | — | [cuBLAS](../docs/source/tutorial/exercises/13-cublas.rst) |
| [14-multi-gpu](14-multi-gpu/) | — | [Multi-GPU Workflow](../docs/source/tutorial/exercises/14-multi-gpu.rst) |
| [15-profiling](15-profiling/) | — | [Profiling with Nsight Systems](../docs/source/tutorial/exercises/15-profiling.rst) |
| [16-debugging](16-debugging/) | — | [Debugging with cuda-gdb](../docs/source/tutorial/exercises/16-debugging.rst) |

`00-performance-basics/` contains the introductory GEMM benchmarks; `legacy/` contains the older examples used by `notebooks/cuda.ipynb`.

Notebook output folders (for example, `thread_indexing_output/`) contain generated files and are separate from exercise source folders.
