Exercise 14: Multi-GPU Workflow
===============================

.. admonition:: Exercise
   :class: todo

   **Time: 15 min**

      1. Inspect and then run the code ``exercises/14-multi-gpu/multi_gpu.cu``.
      2. How will you compare the time taken to complete an H2D transfer and a D2D transfer?

Build with CMake
----------------

From the repository root on Gadi, configure CMake and build this exercise's
executables manually. CMake 3.18 or newer must be available.

.. code-block:: bash

   module load gcc/14.2.0 cuda/12.9.0 intel-mkl/2024.2.1
   cmake -S exercises -B exercises/build -DCMAKE_BUILD_TYPE=Release
   cmake --build exercises/build --target ex_14_multi_gpu_multi_gpu --parallel 4

The executable is written directly into ``exercises/build``:

.. code-block:: text

   exercises/build/ex_14_multi_gpu_multi_gpu

Run the executable inside a GPU allocation, or submit the batch job below.
Rebuild after changing the source; the job script does not compile it.

Submit a batch job
------------------

Build the executables in ``exercises/build`` first (see :doc:`../../job_submission`).
From the repository root,
change to ``job-script`` and submit the prebuilt program on Gadi:

.. code-block:: bash

   cd job-script
   qsub 14-multi-gpu.pbs

If you are already in ``job-script``, run only the ``qsub`` command.
See :doc:`../../job_submission` for setup, monitoring and output locations.
