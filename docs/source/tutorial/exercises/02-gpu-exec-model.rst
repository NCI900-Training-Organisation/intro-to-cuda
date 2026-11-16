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
Compile manually from the repository root:

.. code-block:: bash

   make -C exercises/02-gpu-exec-model

Run the executable inside an allocated GPU session:

.. code-block:: bash

   ./exercises/02-gpu-exec-model/print_launch

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

The same ``make`` command builds both examples. In an allocated GPU session, run:

.. code-block:: bash

   ./exercises/02-gpu-exec-model/print_global_index

To submit only this example, run from ``job-script``:

.. code-block:: bash

   qsub 02b-print-global-index.pbs

The batch job also runs this example. Expect eight output lines in any order:
thread indices repeat from 0 to 3 within each block, while global indices run
from 0 to 7 across the whole grid. Block dimension is 4 and grid dimension is 2.

#. Why does thread 0 in block 1 have global index 4?
#. Change the launch to ``<<<4, 2>>>`` and predict all five printed values.
   Does the number of threads or the range of global indices change?

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

Submit a batch job
------------------

Compile this exercise separately with ``make`` first. From the repository root,
change to ``job-script`` and submit the prebuilt program on Gadi:

.. code-block:: bash

   cd job-script
   qsub 02-gpu-exec-model.pbs

If you are already in ``job-script``, run only the ``qsub`` command.
See :doc:`../../job_submission` for setup, monitoring and output locations.
