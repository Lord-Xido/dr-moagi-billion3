# Dr Moagi / Jarvis-X: DrMoagi3D

A compact Windows x64 C++17 reference runtime. The 1000 × 1000 × 1000 namespace consists of one billion decimal 1 GB cells: exactly 10^18 logical bytes. It is virtual addressing, not additional physical VRAM. Tile data is scalar FP32, 64 × 64 × 64 voxels (1 MiB). No model weights were supplied: E_theta and D_phi are deterministic block-average / nearest-block reconstruction operators, not trained neural networks.

## Build and verify on LORD-XIDO

Compilation has NOT been run in the delivery environment: Windows is present but no CMake, MSVC, GCC or Clang toolchain was found. There is no compiled executable in this delivery. Install Visual Studio 2022 with **Desktop development with C++**, the Windows SDK and CMake tools. Open a Developer PowerShell for VS 2022 in this folder and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-and-verify.ps1
```

This configures x64, builds `build\Release\DrMoagi3D.exe`, runs CTest invariants and CPU end-to-end tests, then executes an 8-tile hardware-auto demo and saves `verification.log`. To force CPU for the final demo:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-and-verify.ps1 -CPUOnly
```

Run directly:

```powershell
.\build\Release\DrMoagi3D.exe --demo --tiles 8 --file dm3d_1GB.vmem --host-mib 256 --gpu-mib 64
.\build\Release\DrMoagi3D.exe --self-test
```

## Addressing and storage

Coordinates x,y,z each range from 0 to 999; the in-cell byte offset ranges from 0 to 999,999,999. Checked 64-bit arithmetic maps the final byte to 999,999,999,999,999,999. A sparse region → brick → tile map stores only present keys. Regions are geometric 100 × 100 × 100 cell blocks; bricks identify individual 1 GB cells within their region; tiles identify aligned 1 MiB chunks within a brick. Each cell has 953 complete tiles and a 707,072-byte unused tail. Unmaterialized reads return zeros.

The backing file has a decimal 1 GB logical length, a 64 KiB header containing magic/version and up to 953 address records, and fixed 1 MiB payload slots. Arbitrary virtual cells can occupy these slots. It does not reserve an exabyte on disk. Sparse holes represent zero-filled materialized capacity; populated tiles occupy real disk blocks. The 1 GB file is not 1 GB of populated nonzero data. Its header and seeded first tile are initially populated. NTFS sparse semantics are requested explicitly; sparse support and physical disk usage depend on the destination filesystem.

The format is little-endian, local to x64 Windows, and single-process/single-writer. Writes are flushed and checked but are not journaled or crash-atomic. Unknown existing formats and incorrect file lengths are rejected. Keep important data elsewhere; this is a reference test substrate, not a production database. Copying a sparse file may expand its disk usage.

## Computation and feedback

The encoder averages each 4³ block into a 16³ latent. Phi repeatedly contracts halfway toward the encoded target for bounded K (8 or 12). The decoder expands the latent; CTR Generates candidates, Contrasts residuals, Reckons RMSE, Verifies finite data, Corrects with the full residual, then verifies reconstruction and backing readback. Omega is an exponential moving average of the tile residual RMSE (rho=0.9), used to select subsequent refinement depth. Omega is runtime-local scalar feedback, not persistent voxelwise learned state. The correction residual consumes memory and provides no compression claim.

CPU always works. On Windows, a Shader Model 5 D3D11 compute backend executes Phi on hardware with feature level 11.0 or later, compatible with the GTX 1650 target and Windows 11. DirectX 11 is used for practical SDK-only compute deployment; this is not a D3D12 implementation. It queries the WDDM local-memory budget through DXGI, checks room for two device buffers and staging readback, and falls back on GPU initialization/allocation/dispatch failure. Device removal or a stalled driver remains subject to Windows driver behavior. Only refinement is GPU accelerated; encoding, decoding, CTR and I/O execute on CPU. GPU startup compilation uses the Windows SDK D3DCompiler library, with no CUDA dependency.

## Resource policy and telemetry

The runtime streams one tile at a time, with a guarded pipeline footprint below 8 MiB plus process/device overhead. Host and GPU limits default to 256 MiB / 64 MiB and may be set between 8 and 2048 MiB. They constrain owned payload allocations, not total process RSS or driver memory. Available host RAM is checked before tile work. Device-local allocations require both configured and live DXGI headroom. No shared-VRAM oversubscription is attempted.

The scheduler reads available RAM, measured I/O time, measured pipeline time and residual error. It adjusts K and computes a bounded admission-window recommendation (1..8); execution remains serial with a one-tile resident working set. Parallel batching/prefetch and adaptive precision are extension points, not implemented features. Diagnostics report timings, raw residual RMSE, corrected maximum error, Omega, K, admission recommendation, available GPU budget and fallback reasons. Pipeline time includes correction writes; it is not a pure GPU timing measurement.

## Verification status

`DELIVERY_STATUS.txt` records delivery checks. The backing length, sparse flag and deterministic seed were verified using Windows file APIs / PowerShell. C++ compilation, CTest and CPU/GPU end-to-end execution remain to be run on Windows with the toolchain above. Do not interpret delivered source or a sparse backing file as a passed runtime verification.

Tests cover boundary addressing, invalid coordinates, deterministic generation, constant-block encoder, refinement convergence and memory-pressure scheduling. The CPU end-to-end CTest checks storage round trip, finite residuals, corrected reconstruction and persistence. Hardware GPU execution is exercised by the build script's final demo; CPU fallback is reported rather than hidden. Compare CPU and GPU logs on LORD-XIDO before performance conclusions.

## GitHub integration

This runtime is added under `windows-runtime/` in `Lord-Xido/dr-moagi-billion3`. It complements the existing Python and `cpp/` implementations; it does not connect their in-memory state or claim numerical equivalence. The Windows Actions workflow builds x64, runs CPU tests and uploads the compact executable and diagnostics. Hosted CI does not verify GTX 1650 acceleration; run the hardware-auto demo on LORD-XIDO for that check.

The 1 GB backing file is excluded from Git. To recreate the initial seeded delivery file, run `powershell -ExecutionPolicy Bypass -File .\scripts\create-substrate.ps1`, then `scripts\verify-delivery.ps1`. The demo also creates a new sparse substrate automatically when its file path does not exist.
