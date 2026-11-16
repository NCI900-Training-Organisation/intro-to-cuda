Exercise 16: Debugging with cuda-gdb
====================================

.. admonition:: Exercise
   :class: todo

   **Time: 15 min**

      1. Debug the code ``exercises/16-debugging/vector_add.cu`` using `cuda-gdb`.
      2. Rectify the error in the code.

Submit a batch job
------------------

From the repository root, change to ``job-script`` and submit the prebuilt program on Gadi:

.. note::

   Before submitting, load the GCC, CUDA and MKL modules and run this single
   command from the repository root to compile all exercises. Executables are
   written to ``exercises/build``:

   .. code-block:: bash

      cmake -S exercises -B exercises/build -DCMAKE_BUILD_TYPE=Release && cmake --build exercises/build --parallel 4

.. code-block:: bash

   cd job-script
   qsub 16-debugging.pbs

If you are already in ``job-script``, run only the ``qsub`` command.
See :doc:`../../job_submission` for setup, monitoring and output locations.
