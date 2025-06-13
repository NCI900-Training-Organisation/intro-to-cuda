cuBLAS
=============

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 30 min

    #. Understand the cuBLAS library and its role in CUDA programming.
    #. Learn how to perform basic matrix operations using cuBLAS.
    #. Explore advanced features of cuBLAS for performance optimization.



The cuBLAS library is NVIDIA's implementation of the **Basic Linear Algebra Subprograms (BLAS)** on NVIDIA 
GPUs. It provides highly optimized routines for performing basic linear algebra operations such as matrix
multiplication, vector addition, and more. cuBLAS is designed to take advantage of the parallel processing
capabilities of NVIDIA GPUs, making it a powerful tool for high-performance computing applications.
cuBLAS is particularly useful for applications that require efficient matrix operations like scientific 
computing.

cuBLAS provides highly optimized routines for common linear algebra operations, such as:

* Vector operations (Level 1 BLAS)
* Matrix-vector operations (Level 2 BLAS)
* Matrix-matrix operations (Level 3 BLAS)

Matrix Multiplication using cuBLAS
-----------------------------

The operation is: `C = alpha * A * B + beta * C`, where `A`, `B`, and `C` are matrices.

.. important::

    The header file for cuBLAS is ``cublas_v2.h``, and the library is linked with ``-lcublas``.


Column and Row Major Order
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Matrix A (Row-major):

.. list-table::
   :header-rows: 0
   :widths: auto

   * - 1
     - 2
   * - 3
     - 4

Matrix A (Column-major):

.. list-table::
   :header-rows: 0
   :widths: auto

   * - 1
     - 3
   * - 2
     - 4



cuBLAS uses column-major ordering (like Fortran), while C/C++ uses row-major. Use IDX2C or manually 
transpose if necessary.

.. code-block:: c
    :linenos:

    #define IDX2C(i,j,ld) (((j)*(ld))+(i))  // Macro to index column-major order


cuBLAS Handle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

cuBLAS is a stateful library. Instead of using global state, it uses a handle object (cublasHandle_t) 
to track:

* Which CUDA stream to use
* Allocated workspace
* Algorithm preferences
* Error state


.. admonition:: Explanation
   :class: attention

    A stateful library is a library that maintains internal state across function calls using a context or 
    handle. This state influences how the functions behave and allows the library to manage things like 
    configuration, resources, or execution context over time.

    You can have multiple cuBLAS kernels running in the same program — and even concurrently.



This design allows:

* Multiple independent cuBLAS contexts
* Thread safety (you can use different handles in different threads)
* Better control over performance tuning


The handle is created using ``cublasCreate``, which initializes the cuBLAS library and prepares it for use.

.. code-block:: c
    :linenos:

    cublasHandle_t handle;
    cublasCreate(&handle);


The handle must be destroyed when no longer needed to free resources:

.. code-block:: c
    :linenos:

    cublasDestroy(handle);


cuBLAS Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``cublasSgemm`` is the function for single-precision matrix multiplication. It performs the operation:
`C = alpha * A * B + beta * C`, where:

* `A` is an m x n matrix
* `B` is an n x k matrix
* `C` is an m x k matrix
* `alpha` is a scalar multiplier for the product A * B
* `beta` is a scalar multiplier for the existing matrix C   


.. admonition:: Explanation
   :class: attention
   
    The leading dimension (ld) is the distance in memory between the start of one column and the start 
    of the next column. For column-major storage (used by cuBLAS), it refers to the number of rows in 
    the matrix. For row-major storage (used by C/C++), it refers to the number of columns, but 
    cuBLAS doesn't use this directly unless you transpose manually.

.. code-block:: c
    :linenos:

    cublasSgemm(    // Single-precision matrix multiplication
        handle,     // cuBLAS handle
        CUBLAS_OP_N, // Operation on A (CUBLAS_OP_N for no transpose)
        CUBLAS_OP_N, // Operation on B (CUBLAS_OP_N for no transpose)
        N,           // Number of rows in A and C
        N,           // Number of columns in B and C
        N,           // Number of columns in A and rows in B
        &alpha,      // Scalar multiplier for A*B
        d_A,         // Pointer to matrix A in device memory
        N,           // Leading dimension of A
        d_B,         // Pointer to matrix B in device memory
        N,           // Leading dimension of B
        &beta,       // Scalar multiplier for C  
        d_C,         // Pointer to matrix C in device memory
        N);          // Leading dimension of C



.. admonition:: Explanation
   :class: attention

    In cuBLAS don't have to manually configure thread blocks and grids like you do in raw CUDA kernel 
    launches. cuBLAS internally

    * Inspects the matrix sizes and layout
    * Picks the best kernel and block/thread/grid configuration
    * Launches the kernel using its own internal logic




The final result `C` is in column-major order, which is the default for cuBLAS. So tp print the result,
we can use the following code:



.. code-block:: c
    :linenos:

    
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << h_C[IDX2C(i, j, N)] << " ";
        }
        std::cout << "\n";
    }

.. admonition:: Key Points
   :class: hint

    * cuBLAS is a stateful library that uses a handle to manage its state.
    * The handle is created with `cublasCreate` and destroyed with `cublasDestroy`.
    * Matrix multiplication is performed using `cublasSgemm`, which requires specifying the operation type, dimensions, and pointers to the matrices.
    * cuBLAS uses column-major order for matrices, which is different from the row-major order used in C/C++.
    * The leading dimension is important for correctly accessing matrix elements in memory.

