# LocateAnything attention routing validation

Source: NVlabs/Eagle `783f656d127ee498137b5ff52603ce36c292d317`.
Model: `nvidia/LocateAnything-3B` revision
`c32291ca5e996f5a7a485845b4f57a233936bba0`.

The standard worker now passes an explicitly selected attention backend into the
model configuration before model construction. Its default leaves the saved model
configuration intact. Text and vision settings are independent; the batch runtime
keeps its existing `la_flash` default.

## End-to-end result

The A100 host ran the original checkpoint on CPU in bfloat16 using a synthetic
224×224 RGB image and `max_new_tokens=16`. Task-scoped compatibility copies of
the model code and Transformers 5.10.4 handled the checkpoint's legacy
`_check_and_adjust_attn_implementation`, tied-weight, RoPE and cache APIs.
Those compatibility edits are not part of this Eagle change. The ordinary worker
requested `attn="sdpa"` and `vision_attn="sdpa"`.

| Worker path | Effective text / vision backend | 16-token prediction time | Speed relative to baseline |
|---|---|---:|---:|
| Original worker with explicit SDPA request | Request not forwarded; no comparable run | N/A | N/A |
| Patched worker with explicit SDPA request | SDPA / SDPA; `Qwen2SdpaAttention` loaded | 14.067 s | N/A |

Model load took 1.402 s in the measured process. The answer contained
`<ref>$$$$$` repetitions for this synthetic image; this run verifies loading and
generation, not semantic grounding quality. A one-token smoke took 11.379 s.
The checkpoint files were removed from the test host after validation.

## Focused regression

`python -m unittest Embodied/tests/test_attention_routing.py -v`: 6 passed.
The tests cover default standard and batch behavior, explicit independent text
and vision routing, invalid backend rejection, and effective-backend mismatch.
Before the change, the explicit SDPA request and unknown-backend tests failed.

No GPU timing comparison is claimed: the 5090 host had no locally obtainable
checkpoint, and the A100 GPUs were occupied by existing workloads.
