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
