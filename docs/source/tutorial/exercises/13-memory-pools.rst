Exercise 13: Memory Pools
=========================

.. admonition:: Exercise
   :class: todo

   **Time: 30 min**

      1. Inspect and then run the code ``exercises/13-memory-pools/default-mempool.cu`` to see you CUDA graphs work.
      2. Inspect and then run the code ``exercises/13-memory-pools/custom-mempool.cu`` to comare the performance of the CUDA graphs with normal kernel laucnhes.

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
   qsub 13-memory-pools.pbs

If you are already in ``job-script``, run only the ``qsub`` command.
See :doc:`../../job-submission` for setup, monitoring and output locations.
