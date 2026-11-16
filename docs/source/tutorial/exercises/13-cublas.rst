Exercise 13: cuBLAS
===================

.. admonition:: Exercise
   :class: todo

   **Time: 30 min**

      1. Inspect and then run the code ``exercises/13-cublas/sgemm_cublas.cu`` to see you how cuBLAS works.
      2. What happens when when ``CUBLAS_OP_T`` is changed to ``CUBLAS_OP_N``?

Submit a batch job
------------------

Build the executables in ``exercises/build`` first (see :doc:`../../job_submission`).
From the repository root,
change to ``job-script`` and submit the prebuilt program on Gadi:

.. code-block:: bash

   cd job-script
   qsub 13-cublas.pbs

If you are already in ``job-script``, run only the ``qsub`` command.
See :doc:`../../job_submission` for setup, monitoring and output locations.
