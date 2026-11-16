Exercise 09: Dynamic Parallelism
================================

.. admonition:: Exercise
   :class: todo

   **Time: 20 min**

      1. Run the code ``exercises/09-dynamic-parallelism/dynamic_parallelism.cu`` and check the output.
      2. What happens if try calling ``cudaDeviceSynchronize();`` in the ``parentKernel()`` function?

Submit a batch job
------------------

Compile this exercise separately with ``make`` first. From the repository root,
change to ``job-script`` and submit the prebuilt program on Gadi:

.. code-block:: bash

   cd job-script
   qsub 09-dynamic-parallelism.pbs

If you are already in ``job-script``, run only the ``qsub`` command.
See :doc:`../../job_submission` for setup, monitoring and output locations.
