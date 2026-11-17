# Final project: blur a batch of images

Build a CUDA program that softens four grayscale images with a **3 × 3 box blur**.
Each output pixel is the average of its original value and its eight neighbours.
Treat pixels outside the image as zero and always divide by nine, rounding down.
Use a separate output buffer so one thread cannot change another thread's input.

The supplied `image-blur.cu` is a **skeleton, not a completed GPU solution**.
It generates synthetic images, writes PGM image files, provides a working CPU blur,
and checks your eventual GPU results. You implement the CUDA parts marked TODO.
No notebook, image downloads, Python packages, or image libraries are required.

## What you must implement

| Feature | Your task | Why it is needed |
| --- | --- | --- |
| Shared memory | Load each output block's pixels and a one-pixel halo into a shared tile. | Neighbouring threads reuse many of the same input pixels. |
| Asynchronous calls | Queue prefetches and kernels on multiple CUDA streams. | Independent images can have work in flight together. |
| CUDA events | Record start/stop events per image; wait on completion and report elapsed time. | Measure work and ensure results are ready before the CPU reads them. |
| Unified memory | Allocate input/output buffers with `cudaMallocManaged`; prefetch to the GPU and return outputs to the CPU. | The same managed allocation is accessible from CPU and GPU code. |

Start with `16 × 16` threads per block. Its shared tile is `18 × 18` bytes:
the central `16 × 16` pixels plus a halo, including all four corners.
The default image size is `35 × 27`, so some blocks are incomplete. Threads
outside the output image still participate in loading and the block barrier.

## Build

From the repository root, with GCC, CUDA, CMake and Ninja available:

```bash
cmake -G Ninja -S exercises -B exercises/build -DCMAKE_BUILD_TYPE=Release -DEXERCISES_BUILD_MKL=OFF && cmake --build exercises/build --target ex-00-project-image-blur --parallel 4
```

MKL is not needed for this project. If the existing build directory uses another
generator, move it aside before configuring with Ninja.

Run from the project directory inside an allocated GPU session:

```bash
cd exercises/00-project
../build/ex-00-project-image-blur --reference-only
../build/ex-00-project-image-blur
```

`--reference-only` writes the four inputs and four CPU reference images and exits
successfully, without GPU operations. The default run intentionally reports
**GPU solution incomplete** and exits unsuccessfully until you complete the TODOs
and replace the final `return false` in `gpu_blur_batch` with `return true`.
Do not simply change that return value: the supplied checks must pass too.

## Suggested approach

1. Inspect the generated `image-0-input.pgm` and `image-0-cpu-blur.pgm` in a viewer
   supporting PGM. Read `cpu_blur` to understand the exact boundary rule.
2. Complete TODOs 1–5: calculate indices, cooperatively load the shared tile,
   synchronize the block, and calculate the nine-pixel average.
3. Complete allocation, launch, synchronization and cleanup for **one stream**
   first. Use the `CUDA_CHECK` macro for every CUDA API call and check launch errors.
4. Add event timing and asynchronous prefetching. Follow the same-stream order in
   TODO 8 so an operation cannot overtake the data it depends on.
5. Set `stream_count` to 2. Queue all images before waiting on their stop events.
   Each image must have independent buffers and its own event pair.
6. Enable the checks. All four images must report `PASS` and produce GPU images
   identical to the CPU references, including the image edges and partial blocks.

## Experiments and deliverables

- Submit your completed source and four matching GPU output images.
- Test `1 × 1`, `16 × 16`, and `35 × 27` images to exercise boundary handling.
- Try larger images and compare one stream with two streams using the same inputs.
- Record per-image event times. They include GPU prefetch, blur, and return prefetch.
  They are not kernel-only timings, and overlapping times must not be added to
  estimate total batch time. A separate wall-clock batch timer is an optional extension.
- Explain the halo, why every block thread reaches the barrier, and why the CPU
  must wait before reading a managed output. Multiple streams do not guarantee a
  speedup, particularly for these small default images.

Keep input generation, allocations, file writing and correctness checking outside
the per-image event interval. For a fair comparison, keep timing boundaries the same.
