Exercise 11: CUDA Graphs
========================

.. admonition:: Exercise
   :class: todo

   **Time: 45 min**

      1. Inspect and then run the code ``exercises/11-cuda-graphs/vector_add_async.cu`` to see you CUDA graphs work.
      2. Inspect and then run the code ``exercises/11-cuda-graphs/compare_graphs.cu`` to comare the performance of the CUDA graphs with normal kernel laucnhes.
      3. Inspect and then run the code ``exercises/11-cuda-graphs/custom_graph.cu`` to see how to create a custom graph.
      4. Inspect and then run the code ``exercises/11-cuda-graphs/update_node_args.cu`` to see how to update the arguments of a node in a graph.

Build with CMake
----------------

From the repository root on Gadi, configure CMake and build this exercise's
executables manually. CMake 3.18 or newer must be available.

.. code-block:: bash

   module load gcc/14.2.0 cuda/12.9.0 intel-mkl/2024.2.1
   cmake -S exercises -B exercises/build -DCMAKE_BUILD_TYPE=Release
   cmake --build exercises/build --parallel 4 --target \
     ex_11_cuda_graphs_compare_graphs \
     ex_11_cuda_graphs_custom_graph \
     ex_11_cuda_graphs_update_node_args \
     ex_11_cuda_graphs_vector_add_async

The executables are written directly into ``exercises/build``:

.. code-block:: text

   exercises/build/ex_11_cuda_graphs_compare_graphs
   exercises/build/ex_11_cuda_graphs_custom_graph
   exercises/build/ex_11_cuda_graphs_update_node_args
   exercises/build/ex_11_cuda_graphs_vector_add_async

Run the executables inside a GPU allocation, or submit the batch job below.
Rebuild after changing the source; the job script does not compile it.

Submit a batch job
------------------

Build the executables in ``exercises/build`` first (see :doc:`../../job_submission`).
From the repository root,
change to ``job-script`` and submit the prebuilt program on Gadi:

.. code-block:: bash

   cd job-script
   qsub 11-cuda-graphs.pbs

If you are already in ``job-script``, run only the ``qsub`` command.
See :doc:`../../job_submission` for setup, monitoring and output locations.
