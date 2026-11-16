Exercise 16: Debugging with cuda-gdb
====================================

.. admonition:: Exercise
   :class: todo

   **Time: 15 min**

      1. Debug the code ``exercises/16-debugging/vector_add.cu`` using `cuda-gdb`.
      2. Rectify the error in the code.

Submit a batch job
------------------

Compile this exercise separately with ``make`` first. From the repository root,
change to ``job-script`` and submit the prebuilt program on Gadi:

.. code-block:: bash

   cd job-script
   qsub 16-debugging.pbs

If you are already in ``job-script``, run only the ``qsub`` command.
See :doc:`../../job_submission` for setup, monitoring and output locations.
