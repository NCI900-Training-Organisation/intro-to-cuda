Exercise 07: Shared Memory
==========================

**Time:** 10 min

Four threads in one block exchange values through a shared array. Each thread
writes its own slot, waits at ``__syncthreads()``, then reads the next slot.
Thread 3 wraps around and reads slot 0.

#. Open ``notebooks/07-shared-memory.ipynb`` and inspect the kernel in
   ``exercises/07-shared-memory/swap-neighbours.cu``.
#. Predict which value each thread reads, then run the program and check the PBS
   output. The read values in thread order should be 20, 30, 40, and 10.
#. Explain why each thread must reach the barrier before any thread reads its
   neighbour's slot.

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
   qsub 07-shared-memory.pbs

If you are already in ``job-script``, run only the ``qsub`` command.
See :doc:`../../job-submission` for setup, monitoring and output locations.
