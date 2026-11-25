# RDNA3 padded GQA-6 tile flash-attention fold (optional 0102 candidate)

**Status: local r359 rework candidate, not published and not registered in `release.json`.** The standalone
artifact applies strictly on the r359 default tree. The cumulative apply after optional 0101 is an unresolved
packaging conflict described below; do not claim `scripts/validate-set.sh` is green and do not apply this candidate
with `--include-optional-all`.

## What it does

For the supported GQA-6 geometry, the ordinary tile launcher selects `ncols2 = 2` and traverses each K/V head
three times per query row. The fold pins `ncols1 = 1, ncols2 = 8`, placing all six real heads in one padded block.
Columns 6 and 7 receive zero Q and are never stored. Warps owning only padding still participate in cooperative
K/V loads and barriers but skip KQ, softmax, and VKQ arithmetic.

The r359 implementation addresses PR #362 review R1-R6:

- `fold_gqa6` is a template parameter. Every pre-existing instantiation uses `fold_gqa6 = false`; fold-only
  validity branches and effective-column calculations are `if constexpr` or compile-time constants.
- `fattn-tile-gqa6-fold-dkq256-dv256-q8_0.cu` is the only folded TU. It instantiates the pinned 1x8 Q8_0 geometry
  for the two existing logit-softcap forms. Existing tile TUs do not contain the folded device kernel.
- `GGML_HIP_FA_TILE_GQA6_FOLD=1` is an explicit opt-in. Unset and `0` leave folding off. The value is read once
  per process.
- A thread-safe one-shot INFO census at `-lv 4` proves that the folded arm launched.
- The architecture gate is `GGML_CUDA_CC_IS_RDNA3_0(cc)`. RDNA3.5 and RDNA4 are excluded.
- `tests/test-backend-ops.cpp` adds head-256/GQA-6 widths 9..16, KV 32768, and derived-mask coverage.
- The old wide-prefill KLD claim and the claim that `=0` restored the original binary path are withdrawn.

## Identities

| item | identity |
|---|---|
| Base release | `v18-919794500-r359` |
| Canonical upstream base | `919794500475862f6768c353255469dc661cb734` |
| Delivery default tip | `003c88e1f2eb6297fc28fe99559a3c6a7c7545cc` |
| Delivery default tree | `3910c43846612385fbf120d6b8ae0f1b4ecf867a` |
| Local patch commit | `fc0596bd7798ae8422adebb5e657f7e324a7d96a` |
| Applied tree | `ab25e49ec2da77871877543f01d7ba97e6222cfa` |
| Patch | `0102-gqa6-fold-rdna3.patch` |
| Patch SHA-256 | `bbb86d65ad5e1bcdfa5522da67f7bc08a1891eba6c37ebb4527697c3ba51b191` |

The local source branch is `gqa6-fold-rdna3-r359`. No new validation tag is claimed.

## Applicability and fallback

Folding can engage only when all existing GQA-optimization requirements hold and:

- the architecture is RDNA3_0;
- K/Q and V head dimensions are both 256;
- the GQA ratio is exactly 6;
- the instantiation is symmetric native Q8_0 K/V;
- query width is 1..8;
- the environment opt-in is set before process start.

Everything else uses the ordinary r359 selection. Unaligned KV and unsupported mask/type/geometry cases do not
fold. The width condition also means a small prefill with `-ub <= 8` can fold; this is not a decode-only code path.

Disabling or omitting the environment variable selects the ordinary aligned GQA-6 `ncols2 = 2` launcher. The fold
changes `ntiles_z_gqa`, occupancy, and therefore the launcher's `parallel_blocks` reduction grouping when enabled.
Greedy text and MTP acceptance can move at near-ties. Fold-on text is a new numerics reference, not a byte-identity
promise against fold-off or an unpatched build.

## Checks completed on this r359 candidate

No full llama.cpp rebuild and no GPU execution have been performed for this candidate yet.

- The existing head-256/Q8_0 TU and the dedicated folded TU compile in isolation with the established ROCm 10.0
  HIP flags for gfx1101 and gfx1102.
- The same two TUs cross-compile for gfx1201. The fold object contains only the pinned 1x8/Q8_0 kernels for
  `use_logit_softcap` false and true; no 2x8 or 4x8 fold variants are instantiated.
- Baseline r359 and `fold=false` objects each contain 34 pre-existing `flash_attn_tile` kernels. Their VGPR, SGPR,
  private/group segment, maximum dynamic group size, and wavefront metadata are identical on gfx1101, gfx1102,
  and gfx1201. This is structural evidence, not a substitute for the required runtime OFF-neutrality check.
- `git format-patch` replay with strict `git am` on `003c88e1f` reproduces tree
  `ab25e49ec2da77871877543f01d7ba97e6222cfa`; `git diff --check` passes.
- The generator reproduces the existing generated files and adds only the dedicated fold TU.

Still required before resubmission: a full build, the complete `FLASH_ATTN_EXT` suite with default/off/on controls,
Q8_0 decode-width `--ub 1` exact-logit checks, identical default/OFF text and unchanged tile timing against an
unpatched r359 binary, fold-on census checks, production A/B with both MTP acceptance values, and the maintainer's
gfx1100/gfx1201 runtime acceptance. The local hardware covers gfx1101/gfx1102 only.

## Unresolved optional-0101 cumulative conflict

Optional 0101 (`mixed-kv`) adds `type_V` to several template declarations that this patch changes with `fold_gqa6`.
It also edits the same generator region. The standalone 0102 patch therefore applies on the default tree but fails
strict application after `0100 + 0101`, first at `fattn-tile.cuh:927` and then in `generate_cu_files.py`.

A version cut specifically on top of 0101 would fail the equally required standalone application on the default
tree. A three-way merge is not an acceptable workaround because optional validation requires strict application.
The current clean alternatives need a maintainer decision about optional-block composition or the 0101/R1 source
structure. For that reason this candidate is intentionally absent from `release.json`; no cumulative optional tree
or green `validate-set.sh` result is claimed.

## Historical r353 measurements (old runtime-branch implementation)

These numbers are context for the fold's value, not validation of the r359 implementation. The old implementation
had the codegen-isolation defect fixed here. Each production cell ran once on RX 7800 XT + RX 7600 XT with
DaisyRose-27B-Q3_K_XL, Q8_0 target K/V, and Q4_0 draft K/V.

| depth | plain off | plain on | gain | MTP-3 off | MTP-3 on | gain | acceptance off/on |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | 26.061 | 26.103 | 1.0016x | 47.604 | 47.378 | 0.9953x | 0.7881 / 0.7742 |
| 32K | 22.924 | 23.833 | 1.0397x | 36.498 | 40.169 | 1.1006x | 0.7590 / 0.7941 |
| 96K | 18.114 | 20.062 | 1.1075x | 24.586 | 28.557 | 1.1615x | 0.7675 / 0.7616 |

MTP throughput includes any acceptance re-roll caused by changed reduction order; it is not a kernel-only speedup.
The 32K acceptance increase is particularly relevant. Plain throughput is the cleaner performance indicator.

The old wide-prefill Q8_0 KLD result is withdrawn as evidence because its wide query batches did not reach the
fold. The maintainer's Q8_0 decode-width measurement on gfx1100 was mean KLD -0.000001, same-top-p 100.000%, and
maximum 0.000054. That is evidence about the old fold's rounding-level behavior, not the required r359 OFF-arm
neutrality check.

## Standalone application (candidate review only)

Reconstruct and verify the r359 default tree first, then apply this artifact alone:

```bash
git checkout 919794500475862f6768c353255469dc661cb734
bash /path/to/llama-cpp-rdna-boosts/scripts/apply-all.sh . /path/to/llama-cpp-rdna-boosts
test "$(git rev-parse HEAD^{tree})" = 3910c43846612385fbf120d6b8ae0f1b4ecf867a
git am /path/to/optional/gqa6-fold-rdna3/0102-gqa6-fold-rdna3.patch
test "$(git rev-parse HEAD^{tree})" = ab25e49ec2da77871877543f01d7ba97e6222cfa
git diff --check
```

Do not use `--include-optional-all` while the 0101 cumulative conflict is unresolved.
