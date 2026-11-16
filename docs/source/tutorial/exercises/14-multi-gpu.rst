Exercise 14: Multi-GPU Workflow
===============================

.. admonition:: Exercise
   :class: todo

   **Time: 15 min**

      1. Inspect and then run the code ``exercises/14-multi-gpu/multi_gpu.cu``.
      2. How will you compare the time taken to complete an H2D transfer and a D2D transfer?

Submit a batch job
------------------

Compile this exercise separately with ``make`` first. From the repository root,
change to ``job-script`` and submit the prebuilt program on Gadi:

.. code-block:: bash

   cd job-script
   qsub 14-multi-gpu.pbs

If you are already in ``job-script``, run only the ``qsub`` command.
See :doc:`../../job_submission` for setup, monitoring and output locations.
