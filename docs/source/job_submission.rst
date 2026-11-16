Submitting exercise jobs on Gadi
================================

Compile each exercise manually with ``make`` before submitting its job. The PBS
scripts only run prebuilt executables; they do not compile them. Use the same
compiler and CUDA modules when building as the scripts load at runtime.

For example, starting from the repository root on Gadi:

.. code-block:: bash

   module load gcc/14.2.0 cuda/12.9.0
   make -C exercises/02-gpu-exec-model
   cd job-script
   qsub 02-gpu-exec-model.pbs

All submission commands below must be run from the ``job-script`` directory.
Choose the job for the exercise you want to run.

.. list-table:: Job submission commands
   :header-rows: 1
   :widths: 35 65

   * - Exercise
     - Command (from ``job-script``)
   * - 00-performance-basics
     - ``qsub 00-performance-basics.pbs``
   * - 01-gpu
     - ``qsub 01-gpu.pbs``
   * - 02-gpu-exec-model
     - ``qsub 02-gpu-exec-model.pbs``
   * - 02a: Thread and block information only
     - ``qsub 02a-print-launch.pbs``
   * - 02b: Thread and block information plus global index
     - ``qsub 02b-print-global-index.pbs``
   * - 02c: Double an array using the local index (intentional bug)
     - ``qsub 02c-double-thread-index.pbs``
   * - 02d: Double an array using the global index
     - ``qsub 02d-double-global-index.pbs``
   * - 03-gpu-workflow
     - ``qsub 03-gpu-workflow.pbs``
   * - 04-warps
     - ``qsub 04-warps.pbs``
   * - 05-asynchronous
     - ``qsub 05-asynchronous.pbs``
   * - 06-shared-memory
     - ``qsub 06-shared-memory.pbs``
   * - 07-events
     - ``qsub 07-events.pbs``
   * - 08-unified-memory
     - ``qsub 08-unified-memory.pbs``
   * - 09-dynamic-parallelism
     - ``qsub 09-dynamic-parallelism.pbs``
   * - 10-occupancy
     - ``qsub 10-occupancy.pbs``
   * - 11-cuda-graphs
     - ``qsub 11-cuda-graphs.pbs``
   * - 12-memory-pools
     - ``qsub 12-memory-pools.pbs``
   * - 13-cublas
     - ``qsub 13-cublas.pbs``
   * - 14-multi-gpu
     - ``qsub 14-multi-gpu.pbs``
   * - 15-profiling
     - ``qsub 15-profiling.pbs``
   * - 16-debugging
     - ``qsub 16-debugging.pbs``

Resources and modules
---------------------

The scripts request project ``vp91``, queue ``gpuvolta``, 15 minutes, one GPU,
12 CPUs and 48 GB host memory. Exercise 14 requests two GPUs, 24 CPUs and 96 GB.
Change the PBS project and storage lines if using another allocation or filesystem.

The scripts load ``gcc/14.2.0`` and ``cuda/12.9.0``. Exercises 02 and 06 also load
``python3/3.11.0``. Exercise 00 needs ``intel-mkl/2024.2.1`` for both manual
compilation and execution. Exercise 15 requires ``nsys`` on the job's PATH.
Exercise 16 uses ``cuda-gdb`` in batch mode to capture the intentional bug and
backtrace; use an interactive GPU allocation for manual debugger stepping.

Monitor jobs and find results
-----------------------------

.. code-block:: bash

   qstat -u "$USER"

PBS writes the merged output/error log in ``job-script``, for example
``02-gpu-exec-model.o123456``. Generated CSV, SVG, raw-image and profiling files appear in
the corresponding exercise directory. Repeated runs overwrite generated files;
run one job per exercise at a time. After editing CUDA source, rebuild manually
before submitting again.
