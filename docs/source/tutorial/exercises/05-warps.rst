Exercise 05: Warps
==================

**Time:** 20 min

Colour pixels by the GPU timestamp recorded by each thread.

#. Open ``notebooks/05-warps.ipynb``.
#. Compile and run ``exercises/05-warps/warp-timing.cu`` using the notebook cells.
#. Compare the warp-ID image with the timestamp image. One block of 256 threads forms eight warps on one streaming multiprocessor.
#. Repeat the measurement. Does warp 0 always record its timestamp first?

The timestamp is a measurement point, not a guarantee of warp execution order. Plotting uses NumPy and Matplotlib.

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
   qsub 05-warps.pbs

If you are already in ``job-script``, run only the ``qsub`` command.
See :doc:`../../job-submission` for setup, monitoring and output locations.
