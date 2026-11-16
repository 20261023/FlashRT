# Experimental DM05 ApxInf: PUBLIC source-only handoff

Actual full native/public formal result exists: job329543 COMPLETED0:0. The
requested four-case output proxy passed, with149.352156ms public-policy median
versus author277.657942ms. This remains experimental, not general release or
deployment acceptance. No tests, smoke/kernel gates, extra reviews or robot
success evaluation were performed.

## Text-only contents
- `dm05-apxinf-experimental.patch`: curated unified delta against the complete archived private
  Hy foundation, including its already-uncommitted content.
- `SOURCE_MANIFEST.json`: base/public/measured source hashes, build and binary
  identities, wrapper boundaries and post-measurement NOT-RUN labels.
- `SOURCE_FILES.txt`: curated repository-relative source path list.
- `BUILD_LICENSE_NOTES.md`: configured build identity, licenses and provenance.
- `RESULT_SUMMARY.json`: aggregate actual results and hashes, no raw tensors.
- `SHA256SUMS`: checksums of all other delivered text files.

## Foundation boundary: not a clean7f37642 rebuild
This is a separately identified PUBLIC delta, not a dump of the private dirty
foundation. It is **NOT self-contained from clean7f37642**, and the dirty
foundation must not be blindly published. The required complete foundation is
supplied/reused separately and is not embedded here.

Foundation archive SHA256:
`deee639fe413e4dc6e1a6fbd2c32224342b41c63bd49d83f714be8e71f07519a`.
Foundation manifest SHA256:
`f432b4795f784ca0a941cb7d8074be498b689f0ab12ce05ac66a547817cafb21`.
Base Git HEAD: `7f37642dd6c5896bca66d7640d9944c2a3fe04b4`; HEAD alone does not
reproduce the archived uncommitted content. Main decides publication; no commit
or push was performed by this packaging task.

The patch has normal Git unified-diff repository-relative paths. Its owner may
apply it to a separately owned extraction of that exact foundation. **No patch
application/check was run during packaging.** No build, target import, model
inference, GPU job, tests or reviewer was run during this follow-up.

## Measured bytes versus new PUBLIC adapters
All 23 runtime Rust/CUDA/binding/policy/benchmark source files
retain their measured hashes. Of the 26 recorded snapshot files,
25 remain byte-identical
in the public delta. `doc/dm05.md` is deliberately updated post-measurement;
new portable helpers, notices and license copies are individually labelled.
All public helper copies are **post-measurement NOT RUN** and are not relabelled
as job329543 private script bytes. Private task/environment/submission scripts,
raw logs with machine locations, raw inputs/actions, assets, binaries and archives
are excluded. Only artifact/log hashes and sanitized aggregate metadata are given.

After applying, read `doc/dm05.md` and
`python/apxinf/examples/dm05_experimental/README.md`. The adapters accept explicit
caller-supplied source/asset/output locations. They do not install dependencies,
download assets, submit jobs or run on import. This handoff authorizes no new GPU
work. Retain the separately supplied foundation's existing licenses/notices.

## Observed result and limits
Forty samples: native full-policy median149.352156ms, mean149.3726382ms,
p95150.10012425ms. Author median277.657942ms. Nominal median ratio1.85908225x
must disclose saved-noise upload on native versus CUDA RNG on author. The
139.0278145ms canonical-CPU-to-normalized-CPU scope is not pure graph replay and
is not ratio-compared with author's GPU-inner number.

Worst public14 relL2 .009324395491569158, minimum cosine .9999608551974092,
max absolute error .0156019926071167. All saved outputs finite and exactly
repeat-stable; core-decoded and public outputs agree. Unchanged limits are0.05
and0.999. Raw arrays are not bundled. Main independently recomputed these facts.

The recorded GPU is A800-SXM4-80GB/SM80/108SMs. **Driver version is NR** because
job329543's saved log does not record it; no older/other-node driver is inferred.
Cold public setup45.151212521s is slower than author17.097278s. Arena reservation
is16GiB with at most9,393,563,008bytes used, not a peak-memory claim.

Only unpadded no-history three448-image BF16/E10/H50/public14 is covered. Eager
FP32 arithmetic, generated public RNG, general shapes, operator gates, transfer
profiling and graph/eager parity are not accepted by this result. Closed-loop
success is NR. No thresholds were loosened or intermediate bitwise gates imposed.
