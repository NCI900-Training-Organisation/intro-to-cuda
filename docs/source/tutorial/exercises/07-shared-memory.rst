Exercise 07: Shared Memory
==========================

**Time:** 30 min

Adjust a grayscale image using its average brightness.

#. Open ``notebooks/07-shared-memory.ipynb`` and inspect the before-and-after previews and reduction diagram.
#. Compile and run ``exercises/07-shared-memory/brightness.cu`` using the notebook cells.
#. Check that every GPU block sum and adjusted pixel matches the CPU reference.
#. Change ``darkness``, ``target_brightness``, and ``threads_per_block`` and compare results.
#. Try image dimensions that produce an incomplete final block. Explain why unused threads contribute zero but still reach every barrier.

Threads cooperate through a shared-memory reduction. The CPU combines block totals, and another kernel applies the brightness gain. ``exercises/07-shared-memory/image-tools.py`` generates the image and CPU reference without extra Python dependencies.

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
   qsub 07-shared-memory.pbs

If you are already in ``job-script``, run only the ``qsub`` command.
See :doc:`../../job-submission` for setup, monitoring and output locations.
