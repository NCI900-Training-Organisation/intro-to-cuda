Final Project: Image Blur
=========================

**Time:** 60–90 min

Build a CUDA program that blurs a batch of grayscale images. This project brings
together shared memory, asynchronous calls, CUDA events and unified memory.
Start with the skeleton in ``exercises/00-project/image-blur.cu`` and complete
the numbered TODOs. You do not need Jupyter or external image libraries.

The problem
-----------

Each output pixel is the average of a **3 × 3 neighbourhood**: the original pixel
and its eight neighbours. Read all nine values from the original image, add them
as integers, divide by nine, and round down. Treat coordinates outside the image
as zero; still divide by nine at image edges. Write into a separate output image
so one thread cannot overwrite data another thread needs to read.

The skeleton generates four different ``35 × 27`` images and provides a working
CPU reference and PGM file writer. Your GPU results must match the CPU output
exactly. The deliberately incomplete image dimensions test partial thread blocks.

What you must implement
-----------------------

.. list-table:: Required CUDA features
   :header-rows: 1
   :widths: 20 80

   * - Feature
     - Your task
   * - Shared memory
     - Cooperatively load an image tile and its one-pixel halo, synchronize the
       block, and reuse those shared pixels for the blur.
   * - Asynchronous calls
     - Queue prefetches and kernel launches for independent images on CUDA streams.
   * - Events
     - Measure each image's processing interval and wait for completion before
       the CPU reads its result.
   * - Unified memory
     - Allocate input/output buffers with ``cudaMallocManaged`` and prefetch them
       between CPU and GPU.

Complete the kernel first
-------------------------

#. Calculate each thread's output coordinates using the global thread index.
#. Start with ``16 × 16`` threads per block. Declare a dynamic shared-memory tile
   of ``18 × 18`` bytes: the output block plus a one-pixel border on all sides.
#. Share the loading work among all block threads. Some threads load several
   pixels. Include the four halo corners, and load zero for out-of-image pixels.
#. Call ``__syncthreads()`` after loading and before reading neighbours. Threads
   outside the output image must still help load and reach the barrier.
#. For valid output coordinates, sum nine shared values and write their integer
   average. Use ceiling division when calculating the grid dimensions.

Complete the batch pipeline
---------------------------

Implement ``gpu_blur_batch`` using the comments in TODOs 6–10:

#. Allocate separate managed input/output buffers for every image. Initialize
   all inputs on the CPU before submitting GPU work.
#. Begin with one stream, then use two. Give each image its own start/stop events
   and assign image ``i`` to stream ``i % stream_count``.
#. Queue the following operations in the image's assigned stream, in order:

   * Record the start event.
   * Prefetch its managed input and output buffers to the current GPU.
   * Launch the shared-memory blur and check for launch errors.
   * Prefetch the output to ``cudaCpuDeviceId``.
   * Record the stop event.

#. Queue all images before waiting. Synchronize each stop event, obtain elapsed
   time with ``cudaEventElapsedTime``, and copy the completed managed result into
   the supplied output image.
#. Free all managed buffers and destroy streams/events after work finishes.
   Use ``CUDA_CHECK`` for every CUDA API call.

Per-image event times include the queued prefetches and the blur. Keep allocation,
initialization, file writing and CPU verification outside that interval. Do not
sum overlapping image times to estimate batch time. An optional wall-clock batch
timer should measure the same work for every stream configuration.

Build and run
-------------

.. note::

   With GCC, CUDA, CMake and Ninja available, run this single command from the
   repository root. The project does not need MKL. Use a fresh build directory
   when switching from a different CMake generator.

   .. code-block:: bash

      cmake -G Ninja -S exercises -B exercises/build -DCMAKE_BUILD_TYPE=Release -DEXERCISES_BUILD_MKL=OFF && cmake --build exercises/build --target ex-00-project-image-blur --parallel 4

Run from the project directory, using an allocated GPU session for GPU work:

.. code-block:: bash

   cd exercises/00-project
   ../build/ex-00-project-image-blur --reference-only
   ../build/ex-00-project-image-blur

The reference-only run writes four input images and four CPU-blurred images
without GPU operations. Open the ``.pgm`` files in an image viewer supporting PGM.

The default run intentionally reports **GPU solution incomplete** and exits
unsuccessfully until you implement the TODOs. Once finished, replace the final
``return false`` in ``gpu_blur_batch`` with ``return true`` to enable validation.
Changing the return value alone is not a solution: all four images must report
``PASS``. GPU image files are saved only for results that pass the checks.

What to submit
--------------

* Your completed ``image-blur.cu`` and four matching GPU-blurred images.
* Correctness results for ``1 × 1``, ``16 × 16`` and ``35 × 27`` images. Change
  the constants in ``main`` and rebuild for each test.
* Timing results comparing one and two streams on the same larger images.
* A short explanation of the halo, the block barrier, and when the CPU can safely
  read a managed output. Explain your measurements even if two streams are slower.

The default small images are for debugging, not a promise of a speedup. Establish
correctness before experimenting with larger images or different block sizes.
