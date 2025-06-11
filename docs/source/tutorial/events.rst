CUDA Events
===========

.. admonition:: Overview
   :class: Overview

    * **Time:** 30 min

    #. Learn about CUDA events and their usage.
    #. Understand how to measure elapsed time using CUDA events.
    
CUDA events are a powerful feature that allows you to measure the time taken by operations on the GPU. 
They can be used to synchronize between different streams and to profile the performance of your CUDA 
applications. Events are lightweight and provide a way to track the completion of operations without 
blocking the CPU.

CUDA Event Basics
----------------------------
CUDA events are used to mark points in time in your CUDA application. They can be created, recorded, and 
queried to determine when certain operations have completed. Events can be used to measure the elapsed time 
between two points in your code, which is useful for performance profiling. CUDA events are created using 
the `cudaEventCreate` function, and you can record an event using `cudaEventRecord`.

.. code-block:: c
    :linenos:

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    // Perform some GPU operations here

    // Record the stop event
    cudaEventRecord(stop, 0);

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Elapsed time: %f ms\n", milliseconds);


Cross stream Synchronization
----------------------------

CUDA events can also be used for cross-stream synchronization. When you record an event in one stream,
it can be waited on in another stream. This allows you to synchronize operations across different streams without blocking the CPU.

.. code-block:: c
    :linenos:

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaEvent_t event;
    cudaEventCreate(&event);

    // Record an event in stream1
    cudaEventRecord(event, stream1);

    // Perform some operations in stream1
    kernel1<<<blocks, threads, 0, stream1>>>(...);

    // Wait for the event in stream2
    cudaStreamWaitEvent(stream2, event, 0);

    // Perform some operations in stream2 that depend on the completion of stream1
    kernel2<<<blocks, threads, 0, stream2>>>(...);

    // Cleanup
    cudaEventDestroy(event);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);


.. admonition:: Explanation
   :class: attention

    ``cudaStreamWaitEvent`` is used to make a stream wait for an event recorded in another stream.

Tmiming events across streams
------------------------------------------------

When timing events across streams, you can record events in one stream and then wait for those events 
in another stream. This allows you to measure the time taken by operations in different streams without 
blocking the CPU.

.. code-block:: c
    :linenos:

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event in stream1
    cudaEventRecord(start, stream1);

    // Perform some operations in stream1
    kernel1<<<blocks, threads, 0, stream1>>>(...);

    // Record the stop event in stream2
    cudaEventRecord(stop, stream2);

    // Wait for the stop event to complete
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Elapsed time across streams: %f ms\n", milliseconds);


.. important::

    #. You cannot use an event recorded in one stream to measure the execution time of operations that occur in another stream unless you enforce proper synchronization.
    #. If streams are non-blocking and concurrent, incorrect usage may lead to race conditions or invalid timings.


.. admonition:: Key Points
   :class: hint
   
    #. CUDA events are used to measure elapsed time and synchronize operations across streams.
    #. Events can be created, recorded, and queried to determine the completion of operations.
    #. Cross-stream synchronization is achieved using `cudaStreamWaitEvent`.
    #. Timing events across streams requires careful synchronization to ensure accurate measurements.
    #. Events are lightweight and do not block the CPU, making them suitable for performance profiling.
