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

One block with 1,024 threads
----------------------------

Next, inspect ``exercises/05-warps/thread-timing.cu``. It launches exactly one
block of 1,024 threads, with the same barrier and clock-read operation as the
256-thread example. With 32 threads per warp, this gives 32 full warps.
The program checks that the GPU supports the requested block size.

The program writes all 1,024 readings to ``thread-timestamps.csv``. The notebook
plots one dot per thread in four panels of 256 threads, so the individual dots
remain readable. Horizontal position is the thread index, vertical position is
its clock-reading offset from the earliest reading in the entire block, and
colour identifies its warp. Every panel uses the same vertical scale.

Rebuild using the shared build cell in ``02-gpu.ipynb``, then submit from
``job-script``:

.. code-block:: bash

   qsub 05-thread-timing.pbs
   qstat -u "$USER"

After the job finishes, run the additional plotting cell in ``05-warps.ipynb``.
Does every thread in a warp record the same value? Repeat the job to see whether
the pattern changes. More warps can expose timing differences, but equal clock
readings remain possible. These offsets are not thread execution durations.

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
