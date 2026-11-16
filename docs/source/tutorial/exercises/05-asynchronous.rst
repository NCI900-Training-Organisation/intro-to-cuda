Exercise 05: Asynchronous CUDA Calls
====================================

.. admonition:: Exercise
   :class: todo

   **Time: 20 min**

      1. Run the code ``exercises/05-asynchronous/vector_add.cu`` and check the output.
      2. Run the code ``exercises/05-asynchronous/vector_add_async.cu`` and check the output.
      3. What is the performance difference between the two codes?

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
   qsub 05-asynchronous.pbs

If you are already in ``job-script``, run only the ``qsub`` command.
See :doc:`../../job_submission` for setup, monitoring and output locations.
