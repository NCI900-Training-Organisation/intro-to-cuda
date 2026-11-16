Exercise 05: Asynchronous CUDA Calls
====================================

.. admonition:: Exercise
   :class: todo

   **Time: 20 min**

      1. Run the code ``exercises/05-asynchronous/vector_add.cu`` and check the output.
      2. Run the code ``exercises/05-asynchronous/vector_add_async.cu`` and check the output.
      3. What is the performance difference between the two codes?

Build with CMake
----------------

From the repository root on Gadi, configure CMake and build this exercise's
executables manually. CMake 3.18 or newer must be available.

.. code-block:: bash

   module load gcc/14.2.0 cuda/12.9.0 intel-mkl/2024.2.1
   cmake -S exercises -B exercises/build -DCMAKE_BUILD_TYPE=Release
   cmake --build exercises/build --parallel 4 --target \
     ex_05_asynchronous_vector_add \
     ex_05_asynchronous_vector_add_async

The executables are written directly into ``exercises/build``:

.. code-block:: text

   exercises/build/ex_05_asynchronous_vector_add
   exercises/build/ex_05_asynchronous_vector_add_async

Run the executables inside a GPU allocation, or submit the batch job below.
Rebuild after changing the source; the job script does not compile it.

Submit a batch job
------------------

Build the executables in ``exercises/build`` first (see :doc:`../../job_submission`).
From the repository root,
change to ``job-script`` and submit the prebuilt program on Gadi:

.. code-block:: bash

   cd job-script
   qsub 05-asynchronous.pbs

If you are already in ``job-script``, run only the ``qsub`` command.
See :doc:`../../job_submission` for setup, monitoring and output locations.
