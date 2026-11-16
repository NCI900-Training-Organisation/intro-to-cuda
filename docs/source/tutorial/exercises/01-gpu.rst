Exercise 01: GPU Architecture
=============================

.. admonition:: Exercise
   :class: todo

   **Time: 20 min**

   Request a GPU on Gadi using the command

   ``module load gcc/14.2.0 openmpi/4.1.7 intel-vtune/2024.2.1 cmake/3.8.2``

   ``module load cuda/12.9.0 cudnn/9.5.0-cuda12 papi/7.1.0 intel-mkl/2024.2.1``
   
   ``qsub -I -q gpuvolta  -P vp91 -l walltime=02:00:00,ncpus=12,ngpus=1,mem=48GB``

   and run the following code to check the GPU architecture.

    1. Check the GPU architecture of your system using the command ``nvidia-smi``.
    2. Run the code ``exercises/01-gpu/architecture.cu`` and check the output.

Build with CMake
----------------

From the repository root on Gadi, configure CMake and build this exercise's
executables manually. CMake 3.18 or newer must be available.

.. code-block:: bash

   module load gcc/14.2.0 cuda/12.9.0 intel-mkl/2024.2.1
   cmake -S exercises -B exercises/build -DCMAKE_BUILD_TYPE=Release
   cmake --build exercises/build --target ex_01_gpu_architecture --parallel 4

The executable is written directly into ``exercises/build``:

.. code-block:: text

   exercises/build/ex_01_gpu_architecture

Run the executable inside a GPU allocation, or submit the batch job below.
Rebuild after changing the source; the job script does not compile it.

Submit a batch job
------------------

Build the executables in ``exercises/build`` first (see :doc:`../../job_submission`).
From the repository root,
change to ``job-script`` and submit the prebuilt program on Gadi:

.. code-block:: bash

   cd job-script
   qsub 01-gpu.pbs

If you are already in ``job-script``, run only the ``qsub`` command.
See :doc:`../../job_submission` for setup, monitoring and output locations.
