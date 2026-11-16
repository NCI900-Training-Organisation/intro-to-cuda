Exercise 04: Warps
==================

**Time:** 20 min

Colour pixels by the GPU timestamp recorded by each thread.

#. Open ``notebooks/04-warps.ipynb``.
#. Compile and run ``exercises/04-warps/warp_timing.cu`` using the notebook cells.
#. Compare the warp-ID image with the timestamp image. One block of 256 threads forms eight warps on one streaming multiprocessor.
#. Repeat the measurement. Does warp 0 always record its timestamp first?

The timestamp is a measurement point, not a guarantee of warp execution order. Plotting uses NumPy and Matplotlib.

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
