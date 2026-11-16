# Build identity, license and wrapper notes

## Recorded builds, not new build claims
Configured Rust1.95.0, GCC11.4.0, CUDA12.8, CMake3.31.12, explicitSM80.
Native Cargo command completed2m45s; Python command completed16m22s, both exit0.
Exact commands/log hashes are in SOURCE_MANIFEST.json. No rebuild/import was
performed while preparing this package. Only binary identities, not binaries:

|Artifact|Bytes|SHA256|
|---|---:|---|
| `dm05_benchmark` |63217368|`445ebb12e1e1439c96dd9f4db66fd66c888e8852eea023798e2ae1703a52a113`|
| `libapxinf_py.so` |74773576|`575b96f869fad10f0abbe04eb762b611e9ae016a8b4c0a0c45823c4dcc174d52`|
| `apxinf_py.so` |74773576|`575b96f869fad10f0abbe04eb762b611e9ae016a8b4c0a0c45823c4dcc174d52`|

Actual job329543 log SHA256: `88446bf5f7e5b6b3f0b89b84ba9d352ccee5d41f67cc6535fd6cdadf4cb05522`. It records A800-SXM4-80GB,
compute8.0,108SMs and whole-model Graph preparation/replay. It does **not** record
the driver version: NR. A driver from an older foundation build, another model
or another allocation must not be substituted. The log remains private because
it contains machine locations; public metadata preserves its original hash.

## Source identity
The foundation base is7f37642 plus disclosed uncommitted private content. The
archive/manifest hashes bind that complete base. This is not a self-contained
clean7f37642 rebuild. Only the curated delta is delivered; do not publish the
dirty foundation. Runtime source SHA256 matches the successful build receipts
used by329543. This packaging comparison is byte provenance, not a patch-apply
or correctness test. All new public helpers/docs are post-measurement NOT RUN.

## License/provenance
OpenDM revision03f799120d425ead51d9482bb1da950b6dabade9 is Apache-2.0,
Copyright2026Dexmal. Transformers5.3.0 Gemma3/SigLIP semantic references are
Apache-2.0; Gemma3 carries Copyright2025Google Inc./HuggingFace Inc. team,
and SigLIP carries Copyright2024Google AI/The HuggingFace Team.
The new family NOTICE records exact source URLs/files and preserved notices.

- `crates/apxinf-model/src/dm05/APXINF_LICENSE.txt` SHA256 `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` (verbatim).
- `crates/apxinf-model/src/dm05/OPENDM_LICENSE.txt` SHA256 `71816671f1537b7e616c5ff005fa4696adae4323549437913a45132e9c3f19c4` (verbatim).
- `crates/apxinf-model/src/dm05/TRANSFORMERS_LICENSE.txt` SHA256 `77fd4710def9ec3c0f6225800e0235f15a425abd4a8b03559127fcd782612049` (verbatim).

The ApxInf root license and existing PyTorch LayerNorm, FlashAttention, CUTLASS
and other vendor notices remain unchanged in the separately supplied foundation
and must be retained. No vendor source trees or private dirty base are embedded
in this public delta. Checkpoint revision5cd18734814abb075a9ccfd9ad6d16777b5cf10e
and its model SHA256 are identity references only, not asset redistribution.

## Wrapper boundary
Public policy performs author CPU image/token/state preprocessing and action
reconstruction, then invokes the existing native ModelRunner with action mask
and provided noise. No author Torch model is constructed for native inference.
Portable serializer uses CPU tensor layout conversion; fixture IDs use exact
F32 integer storage before native I32 upload. Public build helper supplies
explicit caller-owned/toolchain variables instead of private environment paths.
Public helper source copies are not newly run and are not measured private
script identities; per-origin hashes and status are in the manifest. No jobs
are submitted by these helpers or by packaging. No new model, tests or reviews.
