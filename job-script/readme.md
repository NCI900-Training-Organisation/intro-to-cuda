# Gadi PBS job scripts

With CMake and Ninja available, build separately from the repository root,
then submit from `job-script`:

```bash
cmake -G Ninja -S exercises -B exercises/build -DCMAKE_BUILD_TYPE=Release
cmake --build exercises/build --parallel 4
cd job-script
qsub 03-gpu-exec-model.pbs
```

There is one script for every exercise, numbered 01–17. Each script loads the
modules, changes to `../exercises/<exercise>`, and runs the prebuilt executable(s) from
`../exercises/build/` (relative to `job-script`). Names include the exercise
folder, for example `ex-04-gpu-workflow-vector-add`.
There are no build steps or shared shell helpers.

The defaults are project `vp91`, queue `gpuvolta`, 15 minutes, one GPU, 12 CPUs and
48 GB memory. Exercise 15 requests two GPUs, 24 CPUs and 96 GB. Change the PBS
project and storage lines if using another allocation or filesystem.

PBS logs appear in `job-script`, for example `03-gpu-exec-model.o123456`. Generated CSV,
SVG, raw-image and profiling files are saved in the corresponding exercise folder.
Repeated runs overwrite those generated files; run one job per exercise at a time.

- **01:** Requires the MKL module when compiling and running the CPU benchmark.
- **03:** Edit the four dimensions in both commands together.
  To run just the print exercises, submit `qsub 03-01-print-thread-index.pbs` or
  `qsub 03-03-print-global-index.pbs` from this directory after compiling manually.
  The array-doubling comparison has individual jobs `03-02-double-thread-index.pbs`
  (intentional indexing bug and data race) and `03-04-double-global-index.pbs`
  (correct global indexing). The main Exercise 03 job runs all four examples.
- **05:** Writes `timestamps.csv` and prints its first 16 thread records.
  Use `qsub 05-thread-timing.pbs` for a separate one-block, 1,024-thread example;
  its `thread-timestamps.csv` is plotted per thread in `05-warps.ipynb`.
- **07:** Runs the four-thread shared-memory exchange and prints the values read by each thread.
- **09:** Runs the version you compiled. Edit prefetch calls and rebuild to compare.
- **16:** Requires `nsys`; writes `vector-add-unified-profile.nsys-rep`.
- **17:** Uses `cuda-gdb` in batch mode to capture the intentional bug and backtrace.
