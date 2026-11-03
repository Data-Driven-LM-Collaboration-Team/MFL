# L40S signal and prune/merge experiment — measured costs

120/120 calls completed, all strict-valid. Official BF16, identical twenty I3 inputs. This table is cost evidence; semantic evaluation is incomplete.

| Configuration | Actual visual tokens | Visual reduction | Mean case seconds | All LLM tokens |
|---|---:|---:|---:|---:|
| full_native | 361,032 | 0.00% | 32.022 | 425,216 |
| select70 | 110,817 | 69.31% | 20.794 | 172,938 |
| merge70 | 110,817 | 69.31% | 18.753 | 172,077 |
| prune_merge70 | 110,817 | 69.31% | 19.811 | 172,495 |
| norm_merge70 | 110,817 | 69.31% | 18.605 | 171,933 |
| random_merge70 | 110,817 | 69.31% | 19.041 | 172,143 |

All twenty select/merge/prune+merge plans have identical representatives, all norm/random controls have equal counts and actually different representatives. All120 raw token streams decoded on independent CPU and matched the original RAW; map accounting and SHA readback passed. Same fixed seed, rotating configuration order.

Case time includes actual preparation, encoding/compression and generation as recorded; common video decode, model load and engineering gates are excluded. Output lengths differ. This is one pass, not a latency confidence interval. All frames still entered the vision encoder.

Only the first clip has a completed fresh six-arm visual review at present. Full_native and merge70 identify the true white-shirt initiator but additionally label the affected blue-shirt person abnormal. The other methods reverse the directed action or hallucinate a fall. This single example cannot select a signal or establish cohort accuracy.

No adoption or expansion. Aggregate quality metrics remain null, human_verified=false, training_eligible=false. The high-compression successor tests a changed scene/locality policy; do not attribute cross-policy differences solely to retention.

Evidence: evaluation_only/cost_snapshots/20261102T123622.json; actual full-vocabulary GPU proof plus independent local CPU readback; paired_visual_adjudications/clip_0bcffd8678ef479a12e9f485.v001.json.