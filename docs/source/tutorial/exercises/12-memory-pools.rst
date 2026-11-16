Exercise 12: Memory Pools
=========================

.. admonition:: Exercise
   :class: todo

   **Time: 30 min**

      1. Inspect and then run the code ``exercises/12-memory-pools/default_mempool.cu`` to see you CUDA graphs work.
      2. Inspect and then run the code ``exercises/12-memory-pools/custom_mempool.cu`` to comare the performance of the CUDA graphs with normal kernel laucnhes.

Submit a batch job
------------------

Compile this exercise separately with ``make`` first. From the repository root,
change to ``job-script`` and submit the prebuilt program on Gadi:

.. code-block:: bash

   cd job-script
   qsub 12-memory-pools.pbs

If you are already in ``job-script``, run only the ``qsub`` command.
See :doc:`../../job_submission` for setup, monitoring and output locations.
