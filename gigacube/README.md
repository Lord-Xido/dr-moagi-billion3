# JARVIS-X Gigacube — Verified 3D Bit-Lattice Extension

This directory adds a **separate executable** reversible 3D Boolean bit engine to the Dr Moagi Billion³ repository. It does not alter the existing `core/`, `vm/`, `cpp/`, or `ui/` engines.

## Model and limits

- Logical coordinate set: `[0, 8_000_000_000)^3`, interpreted as 8 billion **bit positions per axis** (one decimal GB equivalent of bit count on each axis).
- Logical cell count: `(8e9)^3 = 5.12e29` bits; dense packed storage would be `6.4e28 bytes = 64 ronnabytes`. **This is never allocated.**
- Address: exactly reversible 99-bit 3D Morton interleaving of three 33-bit coordinate fields.
- Physical storage: only allocated `side^3` bit bricks (`side=64` by default), packed to `side^3 / 8` bytes per brick, persisted transactionally with SQLite.
- Unallocated region: frozen zeros. The active brick set is fixed for a database instance; changes to the set require a versioned reinitialization.
- Primitive: the six-neighbor Boolean stencil over even/odd checkerboard parity classes. This is classical reversible computation, **not quantum entanglement**, learned neural compression, or a convergence guarantee.

Forward cycle:

```text
A_next = A XOR F(B)
B_next = B XOR G(A_next)
F = xp XOR xm XOR yp XOR ym XOR zp XOR zm XOR (xp AND yp)
G = xp XOR ym XOR zm XOR (xm AND zp)
```

Inverse cycle applies `G` to the odd cells first, then `F` to the even cells. Each cycle commits bit state, iteration counter, SHA-256 state digest, and journal in one SQLite transaction.

## Install and run

Python 3.10+ and NumPy are required.

```bash
python -m pip install -r gigacube/requirements.txt
python gigacube/jarvisx_gigacube.py selftest
python gigacube/tests/test_cli.py
python gigacube/jarvisx_gigacube.py init --input README.md --db example.sqlite --brick 64
python gigacube/jarvisx_gigacube.py iterate --db example.sqlite --count 10
python gigacube/jarvisx_gigacube.py status --db example.sqlite
python gigacube/jarvisx_gigacube.py render --db example.sqlite --output view.html --points 4000
python gigacube/jarvisx_gigacube.py recover --db example.sqlite --output recovered.md
python -c "import hashlib; a=lambda p:hashlib.sha256(open(p,'rb').read()).hexdigest(); assert a('README.md')==a('recovered.md')"
```

`recover` runs inversions in a temporary SQLite copy, leaving the active database at its iteration level. `reverse` instead rewinds the main database. The generated visualization is an HTML **state snapshot**; Three.js is loaded from a CDN only when that file is opened in a browser. Processing is performed in Python/NumPy, not in the rendered torus.

The default ingest limit is **64 MiB**; `--max-mib` can change it explicitly, but this does not promise large-scale performance or allocation. Do not commit runtime `.sqlite`, recovered data, or private input files. Generated data may contain sensitive source information.

## Verification

```bash
python gigacube/jarvisx_gigacube.py selftest
python gigacube/tests/test_cli.py
```

The unit self-test currently exercises 37 assertions, including Morton inversion, bit-exact roundtrips, X/Y/Z brick-face coupling, restart determinism, transactional rollback, and corruption detection. CI runs these checks and a separate CLI file-ingest/iteration/recover/render smoke test on Python 3.10 and 3.12. The local result is not automatically a GitHub Actions result: inspect the PR checks for independent confirmation.

### Scope boundary

The mapping of logical states into 3D coordinates is not an allocation of the entire theoretical cube. For `B` allocated bricks of side `s`, one forward sweep updates `B*s^3` bit cells; `K` sweeps update `K*B*s^3` cells. Throughput depends on CPU, memory, tile count, and database I/O. Sparse storage alone is not an algorithmic acceleration proof. The neural fixed-point path and this reversible Boolean engine have **different invariants** and must not be conflated.
