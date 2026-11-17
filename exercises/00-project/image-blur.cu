// Final project: blur a batch of grayscale images with CUDA.
// Complete the TODOs. The CPU reference and image-file writer are supplied.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(error)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

struct Image {
    int width;
    int height;
    std::vector<unsigned char> pixels;
};

// Supplied: create repeatable inputs without downloading images or using Python.
Image make_image(int width, int height, int image_number)
{
    Image image{width, height, std::vector<unsigned char>(width * height)};
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            image.pixels[y * width + x] =
                static_cast<unsigned char>(20 + (17 * x + 29 * y + 31 * image_number) % 200);
        }
    }
    return image;
}

// Supplied: the definition of correctness for this project.
// Average the centre and its eight neighbours. Outside-image pixels are zero.
// Divide by NINE even at an edge; integer division rounds down.
Image cpu_blur(const Image &input)
{
    Image output{input.width, input.height,
                 std::vector<unsigned char>(input.pixels.size())};
    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            int sum = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int xx = x + dx, yy = y + dy;
                    if (xx >= 0 && xx < input.width && yy >= 0 && yy < input.height)
                        sum += input.pixels[yy * input.width + xx];
                }
            }
            output.pixels[y * input.width + x] = static_cast<unsigned char>(sum / 9);
        }
    }
    return output;
}

// Supplied: PGM is a simple grayscale image format (header followed by bytes).
bool save_pgm(const std::string &filename, const Image &image)
{
    std::ofstream file(filename, std::ios::binary);
    file << "P5\n" << image.width << ' ' << image.height << "\n255\n";
    file.write(reinterpret_cast<const char *>(image.pixels.data()), image.pixels.size());
    file.close();
    return bool(file);
}

__global__ void blur_shared(const unsigned char *input, unsigned char *output,
                            int width, int height)
{
    // TODO 1: Calculate this thread's output pixel (x, y) using blockIdx,
    // blockDim and threadIdx. Use a 2D block, initially 16 by 16 threads.

    // TODO 2: Declare dynamic shared memory: extern __shared__ unsigned char tile[];
    // A Bx by By block needs a (Bx + 2) by (By + 2) tile. The extra border is
    // the halo: neighbouring pixels required by threads on the block's edge.
    // Include the four halo corners: this is a 3x3 blur, not a cross stencil.

    // TODO 3: Cooperatively load ALL tile slots, including the halo.
    // Hint: flatten the local thread index and tile index. Thread t loads
    // slots t, t + threads_per_block, ... until the whole tile is filled.
    // Map shared (sx, sy) to global:
    //   gx = blockIdx.x * blockDim.x + sx - 1
    //   gy = blockIdx.y * blockDim.y + sy - 1
    // Store zero for out-of-image coordinates. Never read input out of bounds.

    // TODO 4: Call __syncthreads() after loading and BEFORE neighbour reads.
    // Do NOT return early for out-of-image output threads before this barrier.
    // A partial block still needs all its threads to load the tile and cooperate.

    // TODO 5: For valid output pixels, sum the nine corresponding shared values
    // into an int, divide by 9, and write to output[y * width + x].
    // Keep input and output separate so every thread reads the original image.

    // Placeholder only: remove these casts when implementing the kernel.
    (void)input;
    (void)output;
    (void)width;
    (void)height;
}

bool gpu_blur_batch(const std::vector<Image> &inputs, int stream_count,
                    std::vector<Image> &outputs, std::vector<float> &elapsed_ms)
{
    // TODO 6: Allocate TWO cudaMallocManaged buffers per image: input and output.
    // Copy the supplied CPU inputs into the managed inputs. Initialize all images
    // before launching GPU work. Allocate/create resources outside timed regions.
    // The managed buffers must outlive every kernel/prefetch that accesses them.

    // TODO 7: Get the current device ID. Create stream_count CUDA streams and
    // a start/stop event pair for EACH image. Use CUDA_CHECK for every CUDA call.
    // Assign image i to stream i % stream_count; each image has independent buffers.

    // TODO 8: Enqueue this sequence for each image on its assigned stream:
    //   a. Record its start event.
    //   b. cudaMemPrefetchAsync both managed buffers to the selected GPU.
    //   c. Launch blur_shared with a ceiling-divided 2D grid and dynamic shared
    //      memory bytes: (block.x + 2) * (block.y + 2) * sizeof(unsigned char).
    //   d. Check launch errors using cudaGetLastError().
    //   e. cudaMemPrefetchAsync the OUTPUT to cudaCpuDeviceId on the same stream.
    //   f. Record its stop event on the same stream.
    // Queue ALL images before waiting. Waiting inside this loop serializes work.

    // TODO 9: Synchronize each stop event before the CPU reads its managed output.
    // Use cudaEventElapsedTime to fill elapsed_ms[i]. This measures the stream
    // interval including the queued prefetches and blur, not just kernel time.
    // Copy each result into outputs[i].pixels. Do not sum overlapping image times
    // and call that the batch time; use a separate wall clock for that extension.

    // TODO 10: Destroy events and streams and cudaFree every managed buffer,
    // after completion. Avoid cudaDeviceSynchronize after each launch: use the
    // per-image completion events so multiple streams can have work queued.

    // Keep this false until ALL stages above are implemented. Once complete,
    // return true to enable the supplied correctness checks and GPU output files.
    (void)inputs;
    (void)stream_count;
    (void)outputs;
    (void)elapsed_ms;
    return false;
}

int main(int argc, char **argv)
{
    bool reference_only = argc == 2 && std::string(argv[1]) == "--reference-only";
    if (argc > 1 && !reference_only) {
        std::fprintf(stderr, "Usage: %s [--reference-only]\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Intentionally not multiples of 16: your kernel must handle partial blocks.
    // After correctness passes, try larger images to explore performance.
    constexpr int width = 35, height = 27, image_count = 4, stream_count = 2;
    std::vector<Image> inputs, references, outputs;
    for (int i = 0; i < image_count; ++i) {
        inputs.push_back(make_image(width, height, i));
        references.push_back(cpu_blur(inputs.back()));
        // Sentinel values expose missing output writes. Inputs never exceed 219.
        outputs.push_back(Image{width, height, std::vector<unsigned char>(width * height, 255)});
        std::string prefix = "image-" + std::to_string(i);
        if (!save_pgm(prefix + "-input.pgm", inputs.back()) ||
            !save_pgm(prefix + "-cpu-blur.pgm", references.back())) {
            std::fprintf(stderr, "Cannot write image files in the current directory.\n");
            return EXIT_FAILURE;
        }
    }
    std::printf("Prepared %d images (%d x %d) and their CPU blur references.\n",
                image_count, width, height);
    if (reference_only) return EXIT_SUCCESS;

    std::vector<float> elapsed_ms(image_count, -1.0f);
    if (!gpu_blur_batch(inputs, stream_count, outputs, elapsed_ms)) {
        std::fprintf(stderr, "GPU solution incomplete: implement TODOs 1-10 in image-blur.cu.\n");
        return EXIT_FAILURE;
    }
    if (outputs.size() != references.size() || elapsed_ms.size() != references.size()) {
        std::fprintf(stderr, "FAIL: incorrect number of output images or timings.\n");
        return EXIT_FAILURE;
    }
    bool all_passed = true;
    for (int i = 0; i < image_count; ++i) {
        bool passed = outputs[i].width == width && outputs[i].height == height &&
                      outputs[i].pixels == references[i].pixels && elapsed_ms[i] >= 0.0f;
        std::printf("Image %d: %s | prefetch + blur + return: %.3f ms\n",
                    i, passed ? "PASS" : "FAIL", elapsed_ms[i]);
        all_passed = all_passed && passed;
        if (passed && !save_pgm("image-" + std::to_string(i) + "-gpu-blur.pgm", outputs[i]))
            return EXIT_FAILURE;
    }
    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
