Exercise 06: Shared Memory
==========================

**Time:** 30 min

Adjust a grayscale image using its average brightness.

#. Open ``notebooks/06-shared-memory.ipynb`` and inspect the before-and-after previews and reduction diagram.
#. Compile and run ``exercises/06-shared-memory/brightness.cu`` using the notebook cells.
#. Check that every GPU block sum and adjusted pixel matches the CPU reference.
#. Change ``darkness``, ``target_brightness``, and ``threads_per_block`` and compare results.
#. Try image dimensions that produce an incomplete final block. Explain why unused threads contribute zero but still reach every barrier.

Threads cooperate through a shared-memory reduction. The CPU combines block totals, and another kernel applies the brightness gain. ``exercises/06-shared-memory/image_tools.py`` generates the image and CPU reference without extra Python dependencies.

Submit a batch job
------------------

Build the executables in ``exercises/build`` first (see :doc:`../../job_submission`).
From the repository root,
change to ``job-script`` and submit the prebuilt program on Gadi:

.. code-block:: bash

   cd job-script
   qsub 06-shared-memory.pbs

If you are already in ``job-script``, run only the ``qsub`` command.
See :doc:`../../job_submission` for setup, monitoring and output locations.
