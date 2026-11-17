# CUDA exercises

Build all C/C++ and CUDA programs, including nested legacy solutions, with CMake
3.18 or newer and Ninja. On Gadi, load the compiler, CUDA, MKL, CMake and Ninja,
then run from the repository root:

```bash
module load gcc/14.2.0 cuda/12.9.0 intel-mkl/2024.2.1
cmake -G Ninja -S exercises -B exercises/build -DCMAKE_BUILD_TYPE=Release
cmake --build exercises/build --parallel 4
```

Executables appear directly in `exercises/build/` with unique names matching
their CMake targets, such as
`ex-03-gpu-exec-model-04-double-global-index`, so you can build one program:

```bash
cmake --build exercises/build --target ex-03-gpu-exec-model-04-double-global-index
```

The PBS scripts use these uniquely named binaries directly in `exercises/build/`.

The default CUDA architecture is `70` for Gadi's V100 GPUs; override with
`-DCMAKE_CUDA_ARCHITECTURES=80`, for example, for an A100. Without MKL, add
`-DEXERCISES_BUILD_MKL=OFF` to skip only the CPU MKL benchmark. Python helpers
run directly and are not compiled. CMake is the build system for every exercise.

Folders 01–09 match the numbered notebooks. Advanced topics continue the sequence in documentation order. Source filenames describe their purpose; the shared CMake configuration builds all sources.

To run any numbered exercise on Gadi without Jupyter, use the
[PBS job scripts](../job-script/readme.md). For example, from the repository root:

```bash
cmake -G Ninja -S exercises -B exercises/build -DCMAKE_BUILD_TYPE=Release
cmake --build exercises/build --parallel 4
cd job-script
qsub 03-gpu-exec-model.pbs
```

Compile separately before submitting; the job scripts only run the prebuilt executables.

| Folder | Notebook | Documentation |
| --- | --- | --- |
| [01-performance-basics](01-performance-basics/) | [01-performance-basics.ipynb](../notebooks/01-performance-basics.ipynb) | [Performance Basics](../docs/source/tutorial/exercises/01-performance-basics.rst) |
| [02-gpu](02-gpu/) | [02-gpu.ipynb](../notebooks/02-gpu.ipynb) | [GPU Architecture](../docs/source/tutorial/exercises/02-gpu.rst) |
| [03-gpu-exec-model](03-gpu-exec-model/) | [03-gpu-exec-model.ipynb](../notebooks/03-gpu-exec-model.ipynb) | [GPU Execution Model](../docs/source/tutorial/exercises/03-gpu-exec-model.rst) |
| [04-gpu-workflow](04-gpu-workflow/) | [04-gpu-workflow.ipynb](../notebooks/04-gpu-workflow.ipynb) | [GPU Workflow](../docs/source/tutorial/exercises/04-gpu-workflow.rst) |
| [05-warps](05-warps/) | [05-warps.ipynb](../notebooks/05-warps.ipynb) | [Warps](../docs/source/tutorial/exercises/05-warps.rst) |
| [06-asynchronous](06-asynchronous/) | [06-asynchronous.ipynb](../notebooks/06-asynchronous.ipynb) | [Asynchronous CUDA Calls](../docs/source/tutorial/exercises/06-asynchronous.rst) |
| [07-shared-memory](07-shared-memory/) | [07-shared-memory.ipynb](../notebooks/07-shared-memory.ipynb) | [Shared Memory](../docs/source/tutorial/exercises/07-shared-memory.rst) |
| [08-events](08-events/) | [08-events.ipynb](../notebooks/08-events.ipynb) | [CUDA Events](../docs/source/tutorial/exercises/08-events.rst) |
| [09-unified-memory](09-unified-memory/) | [09-unified-memory.ipynb](../notebooks/09-unified-memory.ipynb) | [Unified Memory](../docs/source/tutorial/exercises/09-unified-memory.rst) |
| [10-dynamic-parallelism](10-dynamic-parallelism/) | — | [Dynamic Parallelism](../docs/source/tutorial/exercises/10-dynamic-parallelism.rst) |
| [11-occupancy](11-occupancy/) | — | [Occupancy](../docs/source/tutorial/exercises/11-occupancy.rst) |
| [12-cuda-graphs](12-cuda-graphs/) | — | [CUDA Graphs](../docs/source/tutorial/exercises/12-cuda-graphs.rst) |
| [13-memory-pools](13-memory-pools/) | — | [Memory Pools](../docs/source/tutorial/exercises/13-memory-pools.rst) |
| [14-cublas](14-cublas/) | — | [cuBLAS](../docs/source/tutorial/exercises/14-cublas.rst) |
| [15-multi-gpu](15-multi-gpu/) | — | [Multi-GPU Workflow](../docs/source/tutorial/exercises/15-multi-gpu.rst) |
| [16-profiling](16-profiling/) | — | [Profiling with Nsight Systems](../docs/source/tutorial/exercises/16-profiling.rst) |
| [17-debugging](17-debugging/) | — | [Debugging with cuda-gdb](../docs/source/tutorial/exercises/17-debugging.rst) |

`01-performance-basics/` contains the introductory GEMM benchmarks; `18-legacy/` contains the older examples, with reference solutions in `01-solutions/`.

Notebook output folders (for example, `thread-indexing-output/`) contain generated files and are separate from exercise source folders.

The execution-model exercise has numbered steps in this order:

1. `03-gpu-exec-model/01-print-thread-index/`
2. `03-gpu-exec-model/02-double-thread-index/`
3. `03-gpu-exec-model/03-print-global-index/`
4. `03-gpu-exec-model/04-double-global-index/`
5. `03-gpu-exec-model/05-thread-indexing/`

Asset and solution subfolders are numbered too. Filenames and executable names
use kebab-case. `CMakeLists.txt` keeps the filename required by CMake. Notebook
generated-source examples also use CMake via `EXERCISES_EXTRA_SOURCES`.

If `exercises/build` was configured with another generator, move it aside and
start with a fresh `exercises/build` directory before selecting Ninja.
