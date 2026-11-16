Exercise 02: GPU Execution Model
================================

**Time:** 25 min

Print thread and block information
----------------------------------

Inspect ``exercises/02-gpu-exec-model/print_launch.cu``. Its kernel launches
**two blocks with four threads per block**:

.. code-block:: cuda

   print_launch_info<<<2, 4>>>();

Each thread prints ``threadIdx.x``, ``blockIdx.x``, ``blockDim.x`` and ``gridDim.x``.
Compile manually using the **Build with CMake** instructions below.

Run the executable inside an allocated GPU session:

.. code-block:: bash

   ./exercises/build/ex_02_gpu_exec_model_print_launch

To submit only this example, run from ``job-script``:

.. code-block:: bash

   qsub 02a-print-launch.pbs

Alternatively, submit the batch job below; it runs this example before the
indexing exercise. Expect eight lines: thread indices 0–3 in each of blocks 0 and 1.
Every line reports block dimension 4 and grid dimension 2. The order of printed
lines is not guaranteed. ``cudaDeviceSynchronize()`` waits for the kernel and
allows its output to be displayed before the program exits.

#. Why do thread indices repeat in the second block?
#. Change the launch to three blocks with two threads. Predict the output before running.

Print a global thread index
---------------------------

Next, inspect ``exercises/02-gpu-exec-model/print_global_index.cu``. This separate
example also launches two blocks with four threads each and prints the same four
values, plus a unique global thread index:

.. code-block:: cuda

   unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;

The same CMake build command builds both examples. In an allocated GPU session, run:

.. code-block:: bash

   ./exercises/build/ex_02_gpu_exec_model_print_global_index

To submit only this example, run from ``job-script``:

.. code-block:: bash

   qsub 02b-print-global-index.pbs

The batch job also runs this example. Expect eight output lines in any order:
thread indices repeat from 0 to 3 within each block, while global indices run
from 0 to 7 across the whole grid. Block dimension is 4 and grid dimension is 2.

#. Why does thread 0 in block 1 have global index 4?
#. Change the launch to ``<<<4, 2>>>`` and predict all five printed values.
   Does the number of threads or the range of global indices change?

Double an array: local index versus global index
------------------------------------------------

These two separate programs start with ``{1, 2, 3, 4, 5, 6, 7, 8}`` and launch
two blocks of four threads. Each prints the input, actual output and expected
output. The same CMake build command builds both programs.

First, inspect ``double_thread_index.cu``, which intentionally uses the wrong index:

.. code-block:: cuda

   int i = threadIdx.x;
   if (i < count) {
       values[i] *= 2;
   }

Both blocks address indices 0–3; neither addresses indices 4–7. The overlapping
read/modify/write operations also create a **data race**. Do not expect a fixed
result for the first four elements or assume that they are reliably doubled twice.
``cudaDeviceSynchronize()`` on the host waits for completion; it does not prevent
these conflicting accesses within the kernel.

Next, inspect ``double_global_index.cu``. It changes the index calculation to:

.. code-block:: cuda

   int i = blockIdx.x * blockDim.x + threadIdx.x;

Now the first block handles indices 0–3 and the second handles indices 4–7.
Every element is doubled exactly once, producing ``2 4 6 8 10 12 14 16``.
The program verifies all eight results against the CPU expectation.

Run in an allocated GPU session, from the repository root:

.. code-block:: bash

   ./exercises/build/ex_02_gpu_exec_model_double_thread_index
   ./exercises/build/ex_02_gpu_exec_model_double_global_index

Or, after compiling manually, submit the individual jobs from ``job-script``:

.. code-block:: bash

   qsub 02c-double-thread-index.pbs
   qsub 02d-double-global-index.pbs

The main ``02-gpu-exec-model.pbs`` job also runs both programs.

#. Predict which elements each block modifies before running either program.
#. Why does ``if (i < count)`` fail to fix the local-index version?
#. Try one block with eight threads. Why does the local-index bug disappear for that launch?

Explore thread indexing
-----------------------

Generate labelled thread images for different grid and block dimensions. Each pixel shows its global ID, local ID, block coordinates, and thread coordinates.

#. Open ``notebooks/02-gpu-exec-model.ipynb`` and compare the four reference images.
#. Inspect ``exercises/02-gpu-exec-model/thread_indexing.cu``.
#. Compile and run the CUDA cells, then compare GPU-recorded indices with the formulas.
#. Change the notebook's ``grid`` and ``block`` variables and regenerate the image.
#. Explain why local IDs repeat between blocks while global IDs remain unique.

The helper ``exercises/02-gpu-exec-model/draw_indices.py`` can generate formula-based images without a GPU.

To view the grid as plain text without Jupyter, run this command from the repository root:

.. code-block:: bash

   source /g/data/vp91/Training-Venvs/intro-to-cuda/bin/activate
   
   python3 exercises/02-gpu-exec-model/draw_indices.py 2 2 2 2 --terminal

The four numbers are ``grid_x grid_y block_x block_y``. Change them to explore
different configurations; each dimension must be from 1 to 16. The output shows
global IDs (``G``), local IDs (``L``), block coordinates (``B``), and thread
coordinates (``T``). Heavy solid lines separate blocks. Only Python is required;
these indices are calculated from formulas without GPU execution.

Build with CMake
----------------

From the repository root on Gadi, configure CMake and build this exercise's
executables manually. CMake 3.18 or newer must be available.

.. code-block:: bash

   module load gcc/14.2.0 cuda/12.9.0 intel-mkl/2024.2.1
   cmake -S exercises -B exercises/build -DCMAKE_BUILD_TYPE=Release
   cmake --build exercises/build --parallel 4 --target \
     ex_02_gpu_exec_model_double_global_index \
     ex_02_gpu_exec_model_double_thread_index \
     ex_02_gpu_exec_model_print_global_index \
     ex_02_gpu_exec_model_print_launch \
     ex_02_gpu_exec_model_thread_indexing

The executables are written directly into ``exercises/build``:

.. code-block:: text

   exercises/build/ex_02_gpu_exec_model_double_global_index
   exercises/build/ex_02_gpu_exec_model_double_thread_index
   exercises/build/ex_02_gpu_exec_model_print_global_index
   exercises/build/ex_02_gpu_exec_model_print_launch
   exercises/build/ex_02_gpu_exec_model_thread_indexing

Run the executables inside a GPU allocation, or submit the batch job below.
Rebuild after changing the source; the job script does not compile it.

Submit a batch job
------------------

Build the executables in ``exercises/build`` first (see :doc:`../../job_submission`).
From the repository root,
change to ``job-script`` and submit the prebuilt program on Gadi:

.. code-block:: bash

   cd job-script
   qsub 02-gpu-exec-model.pbs

If you are already in ``job-script``, run only the ``qsub`` command.
See :doc:`../../job_submission` for setup, monitoring and output locations.
