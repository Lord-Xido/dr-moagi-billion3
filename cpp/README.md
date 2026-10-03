# DM3D Native Runtime and Mathematical Verifier

This directory adds a finite C++17 execution layer for the Dr Moagi Billion³
architecture. It is intentionally falsifiable: hyperscale quantities remain
logical/symbolic, while only a bounded active working set is physically
evaluated.

## Closed 3D operator

The native execution path is

```text
INGEST
  -> LIFT_3D
  -> DIFFUSE
  -> ENCODE_3D
  -> COUPLE_6N
  -> FOLD_IN^K
  -> DECODE_3D
  -> COMPARE
  -> REENCODE_ERR
  -> LATADD
  -> RECUR
```

The inner fold is

```text
F_R(Z) = lambda Z + (1-lambda) R + mu Delta Z
```

on a periodic 3D lattice. For the configured values `lambda=0.70` and
`mu=0.02`, the 7-point toroidal Laplacian has spectrum in `[-12,0]`, hence

```text
rho(J_F) <= max(|lambda|, |lambda - 12 mu|) = 0.70 < 1.
```

This proves contractivity of the **inner fixed-reference fold**. It does not,
by itself, prove that the full neural/memory/parameter-update outer machine is
globally contractive.

## Verified invariants

`dm3d_verify` checks:

1. periodic T^3 boundary identity;
2. `Delta(constant)=0`;
3. zero global integral of the toroidal Laplacian;
4. latent codec identity `E(D(Z))=Z` for the block-average/replicate pair;
5. mean conservation of six-neighbour latent coupling;
6. analytical and empirical contraction of `F_R`;
7. decay of the fixed-point residual;
8. reconstruction-loss descent under residual re-encoding + latent addition;
9. deterministic multimodal fusion;
10. finite end-to-end recurrence;
11. symbolic handling of `(10^6)^(10^6)=10^6,000,000`.

## Boundary semantics

The existing Python `GeometryOps.laplacian` currently uses replicate padding.
The native verifier intentionally uses periodic/toroidal boundaries to match the
3D bytecode/fold specification. These are different boundary conditions.
Cross-language numerical equality should therefore only be asserted after the
Python boundary semantics are made selectable or aligned.

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/dm3d_mmc
./build/dm3d_verify
```

For local host-specific optimization:

```bash
cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release -DDM3D_ENABLE_NATIVE_OPT=ON
cmake --build build-native --parallel
```

## Scale contract

The million-cubed substrate and `10^6,000,000` operation space are represented
as sparse/logical address spaces. They are not claims of dense allocation or
physical throughput. Production scaling requires sparse bricks, Morton/octree
indexing, sharding, asynchronous I/O, and measured accelerator/distributed
backends.
