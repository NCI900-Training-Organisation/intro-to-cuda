Exercise 04: Warps
==================

**Time:** 20 min

Colour pixels by the GPU timestamp recorded by each thread.

#. Open ``notebooks/04-warps.ipynb``.
#. Compile and run ``exercises/04-warps/warp_timing.cu`` using the notebook cells.
#. Compare the warp-ID image with the timestamp image. One block of 256 threads forms eight warps on one streaming multiprocessor.
#. Repeat the measurement. Does warp 0 always record its timestamp first?

The timestamp is a measurement point, not a guarantee of warp execution order. Plotting uses NumPy and Matplotlib.

Build with CMake
----------------

From the repository root on Gadi, configure CMake and build this exercise's
executables manually. CMake 3.18 or newer must be available.

.. code-block:: bash

   module load gcc/14.2.0 cuda/12.9.0 intel-mkl/2024.2.1
   cmake -S exercises -B exercises/build -DCMAKE_BUILD_TYPE=Release
   cmake --build exercises/build --target ex_04_warps_warp_timing --parallel 4

The executable is written directly into ``exercises/build``:

.. code-block:: text

   exercises/build/ex_04_warps_warp_timing

Run the executable inside a GPU allocation, or submit the batch job below.
Rebuild after changing the source; the job script does not compile it.

Submit a batch job
------------------

Build the executables in ``exercises/build`` first (see :doc:`../../job_submission`).
From the repository root,
change to ``job-script`` and submit the prebuilt program on Gadi:

.. code-block:: bash

   cd job-script
   qsub 04-warps.pbs

If you are already in ``job-script``, run only the ``qsub`` command.
See :doc:`../../job_submission` for setup, monitoring and output locations.
