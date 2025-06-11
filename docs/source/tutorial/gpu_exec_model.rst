GPU Execution Model
========================================

.. admonition:: Overview
   :class: Overview

    * **Time:** 60 min

    #. Learn the CUDA execution model.
    #. Understand the difference between CUDA kernels and functions.
    #. Learn how to launch CUDA kernels and manage thread indexing.




CUDA (Compute Unified Device Architecture) is a parallel computing platform and programming model developed by NVIDIA. It allows developers to use a 
CUDA-enabled GPU for general-purpose processing, leveraging its massive parallelism capabilities.



CUDA Kernels vs Functions
-----------------------------

In CUDA, both kernels and functions are used to define code behavior, but they have distinct purposes, execution models, and qualifiers. 
CUDA supports three types of functions based on qualifiers:

.. list-table:: CUDA Function Types
   :header-rows: 1
   :widths: 30 70

   * - Qualifier
     - Meaning

   * - ``__global__``
     - Defines a *kernel* function that runs on the device (GPU) and is called from the host (CPU).

   * - ``__device__``
     - Defines a function that runs on the device and can only be called from other device or global functions.

   * - ``__host__``
     - Defines a function that runs on the host (CPU). This is the default for C/C++ functions.

   * - ``__host__ __device__``
     - The function can be compiled for and called from both host and device. Often used for small inline functions or utilities.

CUDA Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A kernel is a special type of function that is executed in parallel by many GPU threads. Kernels are defined using the ``__global__`` keyword and are launched 
using triple angle brackets syntax.

.. code-block:: cpp

   __global__ void add(int *a, int *b, int *c) {
       int i = threadIdx.x;
       c[i] = a[i] + b[i];
   }

   // Launch kernel with 1 block of 256 threads
   add<<<1, 256>>>(d_a, d_b, d_c);

CUDA Device Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Device functions run on the GPU but are called only from other device or global functions. They are useful for modularizing GPU code.

.. code-block:: cpp

   __device__ int add(int x, int y) {
       return x + y;
   }

Host Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Host functions are standard C/C++ functions that execute on the CPU. They typically manage memory and kernel launches.

.. code-block:: cpp

   void initArrays(int *a, int *b) {
       for (int i = 0; i < N; ++i) {
           a[i] = i;
           b[i] = i * 2;
       }
   }



.. list-table:: Difference Between CUDA Kernel and Function
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - CUDA Kernel (``__global__``)
     - CUDA Function (``__device__``, ``__host__``)

   * - Purpose
     - Entry point for GPU parallel execution
     - General subroutine for computation

   * - Runs On
     - GPU
     - CPU or GPU

   * - Called From
     - Host (CPU)
     - Device or Host

   * - Invocation
     - ``<<<grid, block>>>``
     - Standard function call

   * - Return Type
     - Must be ``void``
     - Can return values

   * - Parallelism
     - Runs across many GPU threads
     - Runs per thread (device) or serially (host)

   * - Use Case
     - Massively parallel tasks
     - Helper or utility routines


Execution Model
-----------------------------

In CUDA, parallelism is achieved through a hierarchy of **grids**, **blocks**, and **threads**. When launching a kernel, you specify the grid and block 
configuration using the syntax:

.. code-block:: cpp

    kernel_name<<<numBlocks, threadsPerBlock>>>(args);


The execution model of CUDA is based on a hierarchy of threads organized into blocks and grids. This model allows for massive parallelism by executing many 
threads concurrently on the GPU. The basic structure is as follows:

.. list-table:: CUDA Execution Model
   :header-rows: 1
   :widths: 30 35 35

   * - Level
     - Description
     - Example

   * - Grid
     - A grid is a collection of blocks that execute a kernel.
     - A grid can be 1D, 2D, or 3D.

   * - Block
     - A block is a group of threads that execute together and can share memory.
     - Blocks can also be 1D, 2D, or 3D.

   * - Thread
     - The smallest unit of execution. Each thread executes the same kernel code but operates on different data.
     - Threads are identified by their unique thread index within a block.



Example (1D Launch):

.. code-block:: cpp

   add<<<4, 256>>>(a, b, c);

In the example above, the kernel ``add()`` is launched with 1-D grid which has 4 blocks, each block containing 256 threads. This launches a kernel with:

* 4 blocks in the grid
* 256 threads in each block
* Total threads = 4 × 256 = **1024 threads**

Each performing the ``add()`` operation.

The same kernel can be launched with different configurations, such as 2D or 3D grids and blocks, to suit the problem's dimensionality. For example, 
a 2D grid with 2D blocks might look like this:

.. code-block:: cpp

   kernel_name<<<dim3(2, 2), dim3(16, 16)>>>(args);

This launches a kernel with:
* 2 blocks in the grid (2D)
* Each block has 16 × 16 = 256 threads
* Total threads = 4 × 256 = **1024 threads**

Similarly we can launch a 3D kernel:

.. code-block:: cpp

   kernel_name<<<dim3(2, 2, 2), dim3(4, 4, 4)>>>(args);


This launches a kernel with:

* 2 × 2 × 2 = 8 blocks in the grid (3D)
* Each block has 4 × 4 × 4 = 64 threads
* Total threads = 8 × 64 = **512 threads**


.. admonition:: Explanation
   :class: attention

   ``dim3`` is a built-in C++ struct used to define the dimensions of a grid or a block when launching a kernel.
   It's essentially a convenience structure that stores 3D dimensions (x, y, z), where:

    * x is required
    * y and z default to 1 if not specified


Thread and Block Indexing
-----------------------------

Each thread is aware of its position in the block and the grid using built-in variables:

.. list-table:: Built-in CUDA Thread Identifiers
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Meaning

   * - ``threadIdx.x``, ``threadIdx.y``, ``threadIdx.z``
     - Thread’s index within the block

   * - ``blockIdx.x``, ``blockIdx.y``, ``blockIdx.z``
     - Block’s index within the grid

   * - ``blockDim.x``, ``blockDim.y``, ``blockDim.z``
     - Number of threads per block in each dimension

   * - ``gridDim.x``, ``gridDim.y``, ``gridDim.z``
     - Number of blocks in the grid in each dimension


Computing Global Thread Index
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In most cases when you are working with CUDA, you will need to compute a unique global thread index that identifies each thread across the entire grid.
This is essential for accessing global memory or performing operations that require a unique identifier for each thread.

To compute the global thread index, you can use the following formula:

.. code-block:: cpp

   int idx = blockIdx.x * blockDim.x + threadIdx.x;

This formula combines the block index (`blockIdx.x`) and the thread index within the block (`threadIdx.x`) to give a unique identifier for each thread across 
the entire grid. This global index allows each thread to operate on a unique portion of data, enabling parallel processing of large datasets. While the 
example above is for a 1D grid and block, the same concept applies to 2D and 3D configurations. 

To compute the global thread index in 2D you can extend the formula as follows:

.. code-block:: cpp

   int idx = (blockIdx.x * blockDim.x + threadIdx.x) +
             (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x;  


This formula accounts for both dimensions, allowing each thread to have a unique index in a 2D grid.

.. image:: ../figs/thread_index.drawio.png
    :width: 600px
    :align: center
    :alt: CUDA Thread Indexing
    :caption: CUDA Thread Indexing



To compute the global thread index in 3D, you can further extend it:


.. code-block:: cpp

   int idx = (blockIdx.x * blockDim.x + threadIdx.x) +
             (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x +
             (blockIdx.z * blockDim.z + threadIdx.z) * gridDim.x * gridDim.y * blockDim.x * blockDim.y;

This formula accounts for all three dimensions, ensuring that each thread has a unique index in a 3D grid.


.. list-table:: CUDA Thread Indexing Example
   :header-rows: 1
   :widths: 30 70

   * - Dimension
     - Example Code

   * - 1D
     - ``int idx = blockIdx.x * blockDim.x + threadIdx.x;``

   * - 2D
     - ``int idx = (blockIdx.x * blockDim.x + threadIdx.x) + (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x;``

   * - 3D
     - ``int idx = (blockIdx.x * blockDim.x + threadIdx.x) + (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x + (blockIdx.z * blockDim.z + threadIdx.z) * gridDim.x * gridDim.y * blockDim.x * blockDim.y;``

This indexing allows each thread to access its unique portion of data in global memory, enabling efficient parallel processing.

Why Global Thread Index Matters
---------------------------------

In CUDA, threads run the same code (SIMT model) - which stands for Single Instruction, Multiple Threads. This means that all threads in a block execute the same 
instruction at the same time, but they can operate on different data.  

So without a way to uniquely identify itself, every thread would operate on the same memory location. The  **global thread index** helps distribute work across 
threads so they operate on **independent data**.

Suppose we want to add two arrays ``a`` and ``b`` of size ``n``, and store the result in ``c``. The CUDA kernel would look like this:

.. code-block:: cpp

   __global__ void add(int *a, int *b, int *c, int n) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x; // Compute global thread index
       if (idx < n) { // Ensure we don't go out of bounds
           c[idx] = a[idx] + b[idx]; // Perform addition
       }
   }


Kernel Launch for the Example would look like this:

.. code-block:: cpp

   int n = 1000;
   int threadsPerBlock = 256;
   int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

   add_arrays<<<numBlocks, threadsPerBlock>>>(a, b, c, n);


.. important::

    When the number of CUDA threads exceeds the number of array elements, extra threads are launched, but only those whose global index is within bounds 
    should perform useful work. The others must be guarded with a boundary check to avoid out-of-bounds memory access.

* We launch enough threads to cover all elements in the array.
* Each thread calculates a unique ``idx``.
* Thread 0 processes index 0, thread 1 processes index 1, ..., thread 999 processes index 999.
* Threads with index >= ``n`` are skipped via the boundary check.

.. admonition:: Key Points
   :class: hint

    #. CUDA kernels are launched with a grid of blocks, each containing threads.
    #. Each thread has a unique global index computed from its block and thread indices.
    #. The execution model allows for massive parallelism by executing many threads concurrently.
    #. Understanding the execution model is crucial for writing efficient CUDA code.
