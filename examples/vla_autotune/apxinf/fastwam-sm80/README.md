# Experimental FastWAM ApxInf: source-only handoff

**Actual first full native result exists; experimental, not deployable, not
functionally/release accepted, and not a latency winner.** This package contains
text source/metadata only. It does not contain environments, checkpoints, raw
inputs/actions, credentials, binaries or archives. Main decides later Git
publication; this implementer did not commit/push or edit main FlashRT docs.

## Contents

- `fastwam-apxinf-experimental.patch`: unified source patch against the complete
  archived foundation content, including its pre-existing uncommitted changes.
- `SOURCE_MANIFEST.json`: exact base/current/measurement file hashes, patch and
  foundation identities. Measured runtime bytes are distinguished from later
  documentation/reproduction additions.
- `SOURCE_FILES.txt`: curated source path list.
- `BUILD_LICENSE_NOTES.md`: native library/build/source and license identities.
- `RESULT_SUMMARY.json`: sanitized aggregate actual timings, saved-output
  arithmetic and hashes only; no raw inputs or actions.
- `SHA256SUMS`: text delivery checksums (excluding the checksum file itself).

## Apply to the correct foundation

This is a separately identified **PUBLIC curated source delta**, not the
measured private archive or a dump of its dirty foundation. The measured25-file
implementation delta depends on the archived **private Hy foundation**, including
its pre-existing uncommitted changes. It is **not a self-contained rebuild from
clean `7f37642`**, and the private foundation must not be blindly published.
The public delta contains no machine/account/old-environment paths or asset data.
No path-only runtime parameterization was needed in the curated product files:
their measured bytes were already path-free. Private task build/export scripts
are excluded. A separately identified PUBLIC `build_native.sh` replaces private
task/build/toolchain/environment paths with explicit environment variables;
portable export/benchmark helper adapters are also included. All these later
copies are **not newly run** and are not relabelled as job329256 script bytes.

Foundation archive SHA256:
`deee639fe413e4dc6e1a6fbd2c32224342b41c63bd49d83f714be8e71f07519a`.
Foundation manifest SHA256:
`f432b4795f784ca0a941cb7d8074be498b689f0ab12ce05ac66a547817cafb21`.
Its base Git HEAD is `7f37642dd6c5896bca66d7640d9944c2a3fe04b4`, but HEAD alone
does **not** reproduce the full archived content. The foundation source archive
is supplied/reused separately, not embedded in this text handoff.

After the owner extracts that exact archive into a separate owned checkout,
the patch is in normal Git unified-diff form with repository-relative paths:

```bash
git apply /path/to/fastwam-apxinf-experimental.patch
```

No patch-application test/check, model test, new inference, rebuild, extra review
or GPU submission was run during this packaging follow-up. Source bytes and
hashes were compared with the saved measured payload as packaging provenance,
not as a code-correctness gate. Runtime Rust/CUDA/binding/policy bytes remain
exactly those used by job329256. Later edits are limited to `doc/fastwam.md` and
portable helpers under `python/apxinf/examples/fastwam_experimental/`; those
adapters have not been executed and are not relabelled as the measured scripts.

## Reproduction entry points

After application, read:

```text
doc/fastwam.md
python/apxinf/examples/fastwam_experimental/README.md
```

The README supplies CPU-only export, build-only and separately-authorized actual
full-inference commands using caller-supplied asset/output locations. It does
not install dependencies, download assets, submit jobs or run anything on import.
The source benchmarks require the exact H32/A7/proprio8/cachedT5128/current224x448/
BF16/Euler20shift5 profile. Model computation is allowed only within an allocated
GPU step when independently authorized; **no additional GPU work is authorized
by this source handoff**.

## Actual evidence and limitations

Job329256 COMPLETED exit0 on A800-SXM4-80GB,2026-11-15,
23:49:42->23:51:05+08:00. Native and public policy each saved80outputs.

| Scope | Median ms | Mean ms | P95 ms |
|---|---:|---:|---:|
| Whole graph dispatch/completion wall |212.5182835|212.6541041875|213.09412235|
| Prepared-input upload + graph + output wall |212.869315|213.018002175|213.75278815|
| Public AutoPolicy wall |215.3436585795|215.5860788247|216.6182828136|

Saved native/public outputs versus saved author/project compiled20/job327380:
maxrelL2 0.011609512347092753,mincos0.9999326109645393,maxabs0.05078125,
within the original0.05/0.999 proxy. No new model/testing gate or robot success
evaluation was performed. Independent operators, graph-vs-eager parity, transfer
profiling and release gates were not run under the user's no-tests waiver.

Public CPU FP32 NHWC layout is prepared before timing; historical project input
is NCHW. No pure-backend speedup ratio is calculated. New latency is slower than
historical author142.4363065ms/project137.6502975ms: **not a winner**. Source-only
performance-debt discussion is in `doc/fastwam.md`, with no measured stage shares
or promised optimization gains.
