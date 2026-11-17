Exercise 02: GPU Architecture
=============================

.. admonition:: Exercise
   :class: todo

   **Time: 20 min**

   Ensure CMake 3.18 or newer is available. Request a GPU on Gadi using the command

   ``module load gcc/14.2.0 openmpi/4.1.7 intel-vtune/2024.2.1``

   ``module load cuda/12.9.0 cudnn/9.5.0-cuda12 papi/7.1.0 intel-mkl/2024.2.1``
   
   ``qsub -I -q gpuvolta  -P vp91 -l walltime=02:00:00,ncpus=12,ngpus=1,mem=48GB``

   and run the following code to check the GPU architecture.

    1. Check the GPU architecture of your system using the command ``nvidia-smi``.
    2. Run the code ``exercises/02-gpu/architecture.cu`` and check the output.

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
   qsub 02-gpu.pbs

If you are already in ``job-script``, run only the ``qsub`` command.
See :doc:`../../job-submission` for setup, monitoring and output locations.
