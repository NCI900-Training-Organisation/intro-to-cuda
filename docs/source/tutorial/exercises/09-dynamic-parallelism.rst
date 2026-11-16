Exercise 09: Dynamic Parallelism
================================

.. admonition:: Exercise
   :class: todo

   **Time: 20 min**

      1. Run the code ``exercises/09-dynamic-parallelism/dynamic_parallelism.cu`` and check the output.
      2. What happens if try calling ``cudaDeviceSynchronize();`` in the ``parentKernel()`` function?

Build with CMake
----------------

From the repository root on Gadi, configure CMake and build this exercise's
executables manually. CMake 3.18 or newer must be available.

.. code-block:: bash

   module load gcc/14.2.0 cuda/12.9.0 intel-mkl/2024.2.1
   cmake -S exercises -B exercises/build -DCMAKE_BUILD_TYPE=Release
   cmake --build exercises/build --target ex_09_dynamic_parallelism_dynamic_parallelism --parallel 4

The executable is written directly into ``exercises/build``:

.. code-block:: text

   exercises/build/ex_09_dynamic_parallelism_dynamic_parallelism

Run the executable inside a GPU allocation, or submit the batch job below.
Rebuild after changing the source; the job script does not compile it.

Submit a batch job
------------------

Build the executables in ``exercises/build`` first (see :doc:`../../job_submission`).
From the repository root,
change to ``job-script`` and submit the prebuilt program on Gadi:

.. code-block:: bash

   cd job-script
   qsub 09-dynamic-parallelism.pbs

If you are already in ``job-script``, run only the ``qsub`` command.
See :doc:`../../job_submission` for setup, monitoring and output locations.
