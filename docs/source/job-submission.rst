Submitting exercise jobs on Gadi
================================

Build the exercises manually with CMake in ``exercises/build`` before submitting. The PBS
scripts only run prebuilt executables; they do not compile them. Use the same
compiler and CUDA modules when building as the scripts load at runtime.
CMake and Ninja must also be available when compiling.

For example, starting from the repository root on Gadi:

.. code-block:: bash

   module load gcc/14.2.0 cuda/12.9.0
   module load intel-mkl/2024.2.1
   cmake -G Ninja -S exercises -B exercises/build -DCMAKE_BUILD_TYPE=Release
   cmake --build exercises/build --parallel 4
   cd job-script
   qsub 03-gpu-exec-model.pbs

.. note::

   If ``exercises/build`` was configured with a different generator, move that
   directory aside and start with a fresh ``exercises/build`` before using Ninja.
   An existing CMake cache cannot switch generators.

Executables are read directly from ``exercises/build/`` with unique names such
as ``ex-04-gpu-workflow-vector-add``. Names include the exercise folder to
distinguish programs with repeated source filenames. Jobs still write generated data in
the corresponding source exercise folder.

All submission commands below must be run from the ``job-script`` directory.
Choose the job for the exercise you want to run.

.. list-table:: Job submission commands
   :header-rows: 1
   :widths: 35 65

   * - Exercise
     - Command (from ``job-script``)
   * - 01-performance-basics
     - ``qsub 01-performance-basics.pbs``
   * - 02-gpu
     - ``qsub 02-gpu.pbs``
   * - 03-gpu-exec-model
     - ``qsub 03-gpu-exec-model.pbs``
   * - 03-01-print-thread-index
     - ``qsub 03-01-print-thread-index.pbs``
   * - 03-02-double-thread-index
     - ``qsub 03-02-double-thread-index.pbs``
   * - 03-03-print-global-index
     - ``qsub 03-03-print-global-index.pbs``
   * - 03-04-double-global-index
     - ``qsub 03-04-double-global-index.pbs``
   * - 04-gpu-workflow
     - ``qsub 04-gpu-workflow.pbs``
   * - 05-warps
     - ``qsub 05-warps.pbs``
   * - 05: Per-thread timing, one block of 1,024 threads
     - ``qsub 05-thread-timing.pbs``
   * - 06-asynchronous
     - ``qsub 06-asynchronous.pbs``
   * - 07-shared-memory
     - ``qsub 07-shared-memory.pbs``
   * - 08-events
     - ``qsub 08-events.pbs``
   * - 09-unified-memory
     - ``qsub 09-unified-memory.pbs``
   * - 10-dynamic-parallelism
     - ``qsub 10-dynamic-parallelism.pbs``
   * - 11-occupancy
     - ``qsub 11-occupancy.pbs``
   * - 12-cuda-graphs
     - ``qsub 12-cuda-graphs.pbs``
   * - 13-memory-pools
     - ``qsub 13-memory-pools.pbs``
   * - 14-cublas
     - ``qsub 14-cublas.pbs``
   * - 15-multi-gpu
     - ``qsub 15-multi-gpu.pbs``
   * - 16-profiling
     - ``qsub 16-profiling.pbs``
   * - 17-debugging
     - ``qsub 17-debugging.pbs``

Resources and modules
---------------------

The scripts request project ``vp91``, queue ``gpuvolta``, 15 minutes, one GPU,
12 CPUs and 48 GB host memory. Exercise 15 requests two GPUs, 24 CPUs and 96 GB.
Change the PBS project and storage lines if using another allocation or filesystem.

The scripts load ``gcc/14.2.0`` and ``cuda/12.9.0``. Exercises 03 and 07 also load
``python3/3.11.0``. Exercise 01 needs ``intel-mkl/2024.2.1`` for both manual
compilation and execution. Exercise 16 requires ``nsys`` on the job's PATH.
Exercise 17 uses ``cuda-gdb`` in batch mode to capture the intentional bug and
backtrace; use an interactive GPU allocation for manual debugger stepping.

Monitor jobs and find results
-----------------------------

.. code-block:: bash

   qstat -u "$USER"

PBS writes the merged output/error log in ``job-script``, for example
``03-gpu-exec-model.o123456``. Generated CSV, SVG, raw-image and profiling files appear in
the corresponding exercise directory. Repeated runs overwrite generated files;
run one job per exercise at a time. After editing CUDA source, rebuild manually
before submitting again.
