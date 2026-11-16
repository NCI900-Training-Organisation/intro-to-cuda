Exercise 07: CUDA Events
========================

.. admonition:: Exercise
   :class: todo

   **Time: 20 min**

      1. Run the code ``exercises/07-events/vector_add_async.cu`` and check the output.
      2. Do all the kernels have the same execution time?

Submit a batch job
------------------

Compile this exercise separately with ``make`` first. From the repository root,
change to ``job-script`` and submit the prebuilt program on Gadi:

.. code-block:: bash

   cd job-script
   qsub 07-events.pbs

If you are already in ``job-script``, run only the ``qsub`` command.
See :doc:`../../job_submission` for setup, monitoring and output locations.
