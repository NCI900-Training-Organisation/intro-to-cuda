Exercise 12: CUDA Graphs
========================

.. admonition:: Exercise
   :class: todo

   **Time: 45 min**

      1. Inspect and then run the code ``exercises/12-cuda-graphs/vector-add-async.cu`` to see you CUDA graphs work.
      2. Inspect and then run the code ``exercises/12-cuda-graphs/compare-graphs.cu`` to comare the performance of the CUDA graphs with normal kernel laucnhes.
      3. Inspect and then run the code ``exercises/12-cuda-graphs/custom-graph.cu`` to see how to create a custom graph.
      4. Inspect and then run the code ``exercises/12-cuda-graphs/update-node-args.cu`` to see how to update the arguments of a node in a graph.

Submit a batch job
------------------

From the repository root, change to ``job-script`` and submit the prebuilt program on Gadi:

.. note::

   Before submitting, load the GCC, CUDA and MKL modules and run this single
   command from the repository root to compile all exercises. Executables are
   written to ``exercises/build``:

   .. code-block:: bash

      cmake -S exercises -B exercises/build -DCMAKE_BUILD_TYPE=Release && cmake --build exercises/build --parallel 4

.. code-block:: bash

   cd job-script
   qsub 12-cuda-graphs.pbs

If you are already in ``job-script``, run only the ``qsub`` command.
See :doc:`../../job-submission` for setup, monitoring and output locations.
