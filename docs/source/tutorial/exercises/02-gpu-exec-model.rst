Exercise 02: GPU Execution Model
================================

**Time:** 25 min

Generate labelled thread images for different grid and block dimensions. Each pixel shows its global ID, local ID, block coordinates, and thread coordinates.

#. Open ``notebooks/02-gpu-exec-model.ipynb`` and compare the four reference images.
#. Inspect ``exercises/02-gpu-exec-model/thread_indexing.cu``.
#. Compile and run the CUDA cells, then compare GPU-recorded indices with the formulas.
#. Change the notebook's ``grid`` and ``block`` variables and regenerate the image.
#. Explain why local IDs repeat between blocks while global IDs remain unique.

The helper ``exercises/02-gpu-exec-model/draw_indices.py`` can generate formula-based images without a GPU.

To view the grid as plain text without Jupyter, run this command from the repository root:

.. code-block:: bash

   python3 exercises/02-gpu-exec-model/draw_indices.py 2 2 2 2 --terminal

The four numbers are ``grid_x grid_y block_x block_y``. Change them to explore
different configurations; each dimension must be from 1 to 16. The output shows
global IDs (``G``), local IDs (``L``), block coordinates (``B``), and thread
coordinates (``T``). Four-line borders separate blocks. Only Python is required;
these indices are calculated from formulas without GPU execution.
