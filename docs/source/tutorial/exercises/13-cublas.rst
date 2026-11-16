Exercise 13: cuBLAS
===================

.. admonition:: Exercise
   :class: todo

   **Time: 30 min**

      1. Inspect and then run the code ``exercises/13-cublas/sgemm_cublas.cu`` to see you how cuBLAS works.
      2. What happens when when ``CUBLAS_OP_T`` is changed to ``CUBLAS_OP_N``?

Build with CMake
----------------

From the repository root on Gadi, configure CMake and build this exercise's
executables manually. CMake 3.18 or newer must be available.

.. code-block:: bash

   module load gcc/14.2.0 cuda/12.9.0 intel-mkl/2024.2.1
   cmake -S exercises -B exercises/build -DCMAKE_BUILD_TYPE=Release
   cmake --build exercises/build --target ex_13_cublas_sgemm_cublas --parallel 4

The executable is written directly into ``exercises/build``:

.. code-block:: text

   exercises/build/ex_13_cublas_sgemm_cublas

Run the executable inside a GPU allocation, or submit the batch job below.
Rebuild after changing the source; the job script does not compile it.

Submit a batch job
------------------

Build the executables in ``exercises/build`` first (see :doc:`../../job_submission`).
From the repository root,
change to ``job-script`` and submit the prebuilt program on Gadi:

.. code-block:: bash

   cd job-script
   qsub 13-cublas.pbs

If you are already in ``job-script``, run only the ``qsub`` command.
See :doc:`../../job_submission` for setup, monitoring and output locations.
