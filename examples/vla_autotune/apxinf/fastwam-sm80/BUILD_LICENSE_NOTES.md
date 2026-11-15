# Build, source identity and licenses

## Measured source identity

The measured source delta archive (retained privately, not delivered here) has
SHA256 `e8d71b823a70760b77da12abf5468494b381e0d7ab3fd7ef0dd3a56500c6da79`.
The complete foundation archive/manifest hashes are in `SOURCE_MANIFEST.json`.
This public source delta depends on the archived private Hy foundation with
uncommitted content. It is not independently reproducible from clean7f37642,
and the dirty foundation itself is not part of this publication handoff.
The author reference revision is `7faa71108368fbb3b6885649f112af607427a2d4`.
The source patch's per-file hashes bind runtime Rust, CUDA, Python binding and
policy bytes to the measured payload. Post-measurement helper/docs additions
are individually labelled and have not been executed or rebuilt.
The PUBLIC `build_native.sh` is an explicit path-parameterized copy of the
private task build commands, not the measured private script hash. Runtime
product files needed no path parameterization and retain their measured hashes.

## Observed build (not a fresh build claim)

Rust1.95.0, GCC11.4.0, CUDA12.8, SM80; Python3.11 PyO3 extension.
`cargo build --release --offline --locked -p apxinf-model --features cuda
--example fastwam_bench` completed27.48seconds. `cargo build --release --offline
--locked -p apxinf-py --features cuda,extension-module` completed1m28seconds.
The A800 driver for the measured **job329256** was **580.178.04**, from
`logs/full-native-329256.log:5`:

```text
NVIDIA A800-SXM4-80GB, 580.178.04, 8.0
```

That actual run log has SHA256
`b6cc273d4e026a868c183110e08e2e4517e283af49288ebe20d538a9f762eecc`.
This is the inference-job driver identity, not an inferred driver from an older
allocation or native-library build. No tests/import smoke scripts were run.

Complete measured artifacts (identities only; no binaries are bundled):

| Artifact | SHA256 |
|---|---|
| native `fastwam_bench` | `0d7aaa805aaf75bc790bb9c3e0eabd681ecf12c14bb9afcbd69033114dfbe638` |
| `apxinf_py.so` | `b3e00e93d6dc28dbfee499bff6eb227bcb5d87e4abe90c1411991ff660d8ed7b` |
| read-only foundation `libapxinf_kernels.a` | `8018fc360d71208da11a618d1d0ea85bde294705df1a7922f937bfb69744f51b` |
| read-only foundation `libapxinf_gemm_native.a` | `45d132ec4e483674945e347a59df72d752da33f8687fe2e07ac1a1803380c3ac` |

Existing operator build IDs from the reused foundation's metadata:

```text
legacy APXINF_KERNEL_BUILD_ID=kb1-fa4c9c72d878e786e5240bc8a863c0b5
new APXINF_KERNEL_BUILD_ID=gemm-kb1-83ac197031ecb5e9f8be11cbd5793308
APXINF_TENSOR_OPS_BUILD_ID=tensor-kb1-63097efc82c452bd80e9c87f638b4db6
```

These IDs identify the **unchanged reused operators**, not the added spatial
kernel. Added spatial implementation identities are source SHA256:

```text
spatial_ops.cu 298446c522bc0fa834b349c2dfd8df655fac7358e7b1df9a3dd89a3fe700cbd3
spatial_ops.rs 960534c71eec2a7441d0649d112ed6d2ba8c0db30cbb08546b18beb453986c15
```

The added spatial CUDA object was compiled into the owned target and linked
alongside read-only foundation archives. `APXINF_FOUNDATION_CUDA_BUILD` and
`APXINF_FOUNDATION_CUDA_NEW_BUILD` are explicit optional build inputs, not model
backends. Only use matching source/content-addressed native artifacts; arbitrary
prebuilt directories are not equivalent to this recorded build. Leaving both
unset takes the full source build path, which is provided but not the observed
build and was not rerun. Build output and environments are intentionally absent.

## Asset identity — no asset data included

| Reused asset | SHA256 |
|---|---|
| `libero_uncond_2cam224.pt` | `1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579` |
| `Wan2.2_VAE.pth` | `20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36` |

The checkpoint exporter allows only those digests, performs one-time CPU weight
mapping and preserves transformation provenance. It constructs no model and
does not perform native/reference inference. The source package conveys no
checkpoint redistribution permission; obtain/reuse assets under their own terms.

## Licenses / provenance

- Foundation ApxInf root `LICENSE` (Apache-2.0) remains unchanged and is not
  replaced by this patch. The user must retain it with the patched foundation.
- `crates/apxinf-model/src/fastwam/FASTWAM_LICENSE.txt` preserves the FastWAM
  authors' MIT license verbatim. `NOTICE` records author revision and source
  semantic files. Wan implementation provenance remains through the referenced
  FastWAM/Wan2.2 source and is not superseded by this notice.
- Existing third-party/vendor source, licenses and notices are unchanged; no
  vendor trees, binary archives or checkpoint files are added to the patch.
- New family code composes safe ApxInf CUDA operators. No Torch, TensorRT or
  external inference engine is relabelled as native ApxInf inference.
- The portable reproduction helpers and result notes do not establish acceptance
  of unexecuted paths. Experimental/not-deployable status is intentional.
