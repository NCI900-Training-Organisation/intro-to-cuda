Exercise 05: Warps
==================

**Time:** 20 min

Visualize warp membership and when each warp reaches a clock-reading instruction.

#. Open ``notebooks/05-warps.ipynb``.
#. Build once in ``02-gpu.ipynb``, then submit the warp-timing job from the notebook.
#. Compare the warp-membership image with the timestamp dot plot. One block of 256 threads forms eight warps on one streaming multiprocessor.
#. Repeat the measurement. Does warp 0 always record its timestamp first?

Each dot marks a warp's earliest clock reading; its horizontal line extends to the
latest reading among that warp's 32 threads. Equal readings collapse to one dot.
These are clock-read offsets, not execution durations or a guarantee of warp
execution order. Plotting uses standard Python and inline SVG.

Submit a batch job
------------------

From the repository root, change to ``job-script`` and submit the prebuilt program on Gadi:

.. note::

   Before submitting, ensure CMake and Ninja are available, load the GCC, CUDA
   and MKL modules, and run this single
   command from the repository root to compile all exercises. Executables are
   written to ``exercises/build``:

   .. code-block:: bash

      cmake -G Ninja -S exercises -B exercises/build -DCMAKE_BUILD_TYPE=Release && cmake --build exercises/build --parallel 4

.. code-block:: bash

   cd job-script
   qsub 05-warps.pbs

If you are already in ``job-script``, run only the ``qsub`` command.
See :doc:`../../job-submission` for setup, monitoring and output locations.
