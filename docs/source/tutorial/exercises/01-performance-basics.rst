Exercise 01: Performance Basics
===============================

**Time:** 20 min

Compare matrix multiplication on the CPU using Intel MKL and on the GPU using
cuBLAS. Inspect ``exercises/01-performance-basics/perf-gemm-intelmkl.c`` and
``exercises/01-performance-basics/perf-gemm-cuda.cu``. The matching notebook is
``notebooks/01-performance-basics.ipynb``; you can run both programs without it.

#. Inspect the matrix size, data type and number of repetitions in each program.
#. Run both executables and record their reported times.
#. Explain which setup, allocation and transfer costs each measurement includes
   before comparing the CPU and GPU results.

Submit a batch job
------------------

.. note::

   Load GCC, CUDA and Intel MKL, then run this single command from the repository
   root. All executables are written directly into ``exercises/build``.

   .. code-block:: bash

      cmake -S exercises -B exercises/build -DCMAKE_BUILD_TYPE=Release && cmake --build exercises/build --parallel 4

From the repository root, change to ``job-script`` and submit:

.. code-block:: bash

   cd job-script
   qsub 01-performance-basics.pbs

See :doc:`../../job-submission` for setup, monitoring and output locations.
