Exercise 03: GPU Execution Model
================================

**Time:** 30 min

Work through these four programs in order. Every program launches **two blocks
with four threads per block**. First use the local thread index, observe its
limitations, then use a global index to address all eight elements correctly.

1. Print the thread index
-------------------------

Inspect ``exercises/03-gpu-exec-model/01-print-thread-index/print-thread-index.cu``:

.. code-block:: cuda

   print_launch_info<<<2, 4>>>();

Each thread prints its thread index, block index, block dimension and grid
dimension. Expect eight lines: thread indices 0–3 repeat in blocks 0 and 1;
every line reports block dimension 4 and grid dimension 2. Output order is not
guaranteed. ``cudaDeviceSynchronize()`` waits for completion and flushes output.

After building with the command in the note below, run inside a GPU allocation
from the repository root:

.. code-block:: bash

   ./exercises/build/ex-03-gpu-exec-model-01-print-thread-index

Or submit only this step from ``job-script``:

.. code-block:: bash

   qsub 03-01-print-thread-index.pbs

Why does each block have a thread with index 0?

2. Double using the thread index
--------------------------------

Inspect ``exercises/03-gpu-exec-model/02-double-thread-index/double-thread-index.cu``. It starts with
``{1, 2, 3, 4, 5, 6, 7, 8}`` and intentionally uses the wrong array index:

.. code-block:: cuda

   int i = threadIdx.x;
   if (i < count) {
       values[i] *= 2;
   }

Both blocks address elements 0–3; neither addresses elements 4–7. Their overlapping
read/modify/write operations create a **data race**, so the first four results
are not reliable. Do not assume those elements are consistently doubled twice.
The host's ``cudaDeviceSynchronize()`` waits for completion; it does not prevent
conflicting accesses inside the kernel. The bounds check does not make the
indices unique.

.. code-block:: bash

   ./exercises/build/ex-03-gpu-exec-model-02-double-thread-index

Or submit this step from ``job-script``:

.. code-block:: bash

   qsub 03-02-double-thread-index.pbs

Predict which array elements each block accesses. Why would this bug disappear
with one block of eight threads?

3. Print the global thread index
--------------------------------

Inspect ``exercises/03-gpu-exec-model/03-print-global-index/print-global-index.cu``. It prints the same
four values as step 1, plus a global index calculated from both block and thread:

.. code-block:: cuda

   unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;

Thread indices still repeat from 0 to 3, but global indices are unique from
0 to 7. Thread 0 in block 1 has global index 4.

.. code-block:: bash

   ./exercises/build/ex-03-gpu-exec-model-03-print-global-index

Or submit this step from ``job-script``:

.. code-block:: bash

   qsub 03-03-print-global-index.pbs

Change the launch to ``<<<4, 2>>>``. Predict all five printed values before running.
Does the range of global indices change?

4. Double using the global thread index
---------------------------------------

Inspect ``exercises/03-gpu-exec-model/04-double-global-index/double-global-index.cu``. It starts with the
same eight values as step 2, changing the indexing expression to:

.. code-block:: cuda

   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < count) {
       values[i] *= 2;
   }

Block 0 handles elements 0–3 and block 1 handles elements 4–7. Each element has
one writer and is doubled exactly once. The program checks the result against
``2 4 6 8 10 12 14 16`` and reports ``PASS``.

.. code-block:: bash

   ./exercises/build/ex-03-gpu-exec-model-04-double-global-index

Or submit this step from ``job-script``:

.. code-block:: bash

   qsub 03-04-double-global-index.pbs

Explain why the same bounds check now covers the full array without overlapping
writes. Compare this output with step 2.

Explore thread indexing
-----------------------

Generate labelled thread images for different grid and block dimensions. Each pixel shows its global ID, local ID, block coordinates, and thread coordinates.

#. Open ``notebooks/03-gpu-exec-model.ipynb`` and compare the four reference images.
#. Inspect ``exercises/03-gpu-exec-model/05-thread-indexing/thread-indexing.cu``.
#. Compile and run the CUDA cells, then compare GPU-recorded indices with the formulas.
#. Change the notebook's ``grid`` and ``block`` variables and regenerate the image.
#. Explain why local IDs repeat between blocks while global IDs remain unique.

The helper ``exercises/03-gpu-exec-model/05-thread-indexing/draw-indices.py`` can generate formula-based images without a GPU.

To view the grid as plain text without Jupyter, run this command from the repository root:

.. code-block:: bash

   source /g/data/vp91/Training-Venvs/intro-to-cuda/bin/activate
   
   python3 exercises/03-gpu-exec-model/05-thread-indexing/draw-indices.py 2 2 2 2 --terminal

The four numbers are ``grid_x grid_y block_x block_y``. Change them to explore
different configurations; each dimension must be from 1 to 16. The output shows
global IDs (``G``), local IDs (``L``), block coordinates (``B``), and thread
coordinates (``T``). Heavy solid lines separate blocks. Only Python is required;
these indices are calculated from formulas without GPU execution.

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
   qsub 03-gpu-exec-model.pbs

If you are already in ``job-script``, run only the ``qsub`` command.
See :doc:`../../job-submission` for setup, monitoring and output locations.
