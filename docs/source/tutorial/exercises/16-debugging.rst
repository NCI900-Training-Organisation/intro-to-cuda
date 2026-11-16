Exercise 16: Debugging with cuda-gdb
====================================

.. admonition:: Exercise
   :class: todo

   **Time: 15 min**

      1. Debug the code ``exercises/16-debugging/vector_add.cu`` using `cuda-gdb`.
      2. Rectify the error in the code.

Build with CMake
----------------

From the repository root on Gadi, configure CMake and build this exercise's
executables manually. CMake 3.18 or newer must be available.

.. code-block:: bash

   module load gcc/14.2.0 cuda/12.9.0 intel-mkl/2024.2.1
   cmake -S exercises -B exercises/build -DCMAKE_BUILD_TYPE=Release
   cmake --build exercises/build --target ex_16_debugging_vector_add --parallel 4

The executable is written directly into ``exercises/build``:

.. code-block:: text

   exercises/build/ex_16_debugging_vector_add

Run the executable inside a GPU allocation, or submit the batch job below.
Rebuild after changing the source; the job script does not compile it.

Submit a batch job
------------------

Build the executables in ``exercises/build`` first (see :doc:`../../job_submission`).
From the repository root,
change to ``job-script`` and submit the prebuilt program on Gadi:

.. code-block:: bash

   cd job-script
   qsub 16-debugging.pbs

If you are already in ``job-script``, run only the ``qsub`` command.
See :doc:`../../job_submission` for setup, monitoring and output locations.
