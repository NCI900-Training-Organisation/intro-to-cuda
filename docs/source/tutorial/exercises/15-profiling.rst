Exercise 15: Profiling with Nsight Systems
==========================================

.. admonition:: Exercise
   :class: todo

   **Time: 20 min**

      1. Run the code ``exercises/15-profiling/vector_add_unified.cu`` using `make & make profile`.
      2. Install `nsys <https://developer.nvidia.com/nsight-systems/get-started>`_ .
      3. Copy the file `vector_add_unified_profile.nsys-rep` to your laptop and inspect profile using `nsys`.

Build with CMake
----------------

From the repository root on Gadi, configure CMake and build this exercise's
executables manually. CMake 3.18 or newer must be available.

.. code-block:: bash

   module load gcc/14.2.0 cuda/12.9.0 intel-mkl/2024.2.1
   cmake -S exercises -B exercises/build -DCMAKE_BUILD_TYPE=Release
   cmake --build exercises/build --target ex_15_profiling_vector_add_unified --parallel 4

The executable is written directly into ``exercises/build``:

.. code-block:: text

   exercises/build/ex_15_profiling_vector_add_unified

Run the executable inside a GPU allocation, or submit the batch job below.
Rebuild after changing the source; the job script does not compile it.

Submit a batch job
------------------

Build the executables in ``exercises/build`` first (see :doc:`../../job_submission`).
From the repository root,
change to ``job-script`` and submit the prebuilt program on Gadi:

.. code-block:: bash

   cd job-script
   qsub 15-profiling.pbs

If you are already in ``job-script``, run only the ``qsub`` command.
See :doc:`../../job_submission` for setup, monitoring and output locations.
