Exercise 16: Profiling with Nsight Systems
==========================================

.. admonition:: Exercise
   :class: todo

   **Time: 20 min**

      1. Run the code ``exercises/16-profiling/vector-add-unified.cu`` using the CMake build command below, then submit the profiling job.
      2. Install `nsys <https://developer.nvidia.com/nsight-systems/get-started>`_ .
      3. Copy the file `vector-add-unified-profile.nsys-rep` to your laptop and inspect profile using `nsys`.

Submit a batch job
------------------

From the repository root, change to ``job-script`` and submit the prebuilt program on Gadi:

.. note::

   Before submitting, load the GCC, CUDA and MKL modules and run this single
   command from the repository root to compile all exercises. Executables are
   written to ``exercises/build``:

   .. code-block:: bash

      cmake -S exercises -B exercises/build -DCMAKE_BUILD_TYPE=Release && cmake --build exercises/build --parallel 4

.. code-block:: bash

   cd job-script
   qsub 16-profiling.pbs

If you are already in ``job-script``, run only the ``qsub`` command.
See :doc:`../../job-submission` for setup, monitoring and output locations.
