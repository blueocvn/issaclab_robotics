# Frozen1798 camera-framing A/B experiment

The agreed test completed without training or changing repository defaults. Only the image sampler focal-scale override differed. The selected1798 checkpoint hash stayed unchanged.

## Inputs

- A: current64x48 center crop from80x60 raw render; focal scale[1.0,1.0].
- B: full80x60 render resized to64x48; focal scale[0.8,0.8].
- Numerical transform verification matched exact crop/bilinear resize within0.00000221.

![Input comparison](input_comparison.png)

## Matched-start results

| Population | A: current crop | B: full-frame resize |
| --- | ---: | ---: |
| nominal | 229/256 (89.45%) | 58/256 (22.66%) |
| wide | 63/256 (24.61%) | 33/256 (12.89%) |

Both cohorts had exact initial-pose matches and identical reset provenance. The command lines differed only in focal_scale_range. The selected checkpoint SHA256 is `92e88f27e5f8e9b5817e42742d779f6c41d96fe50b983e9585212cf5529aebf6`.

## Decision

Keep A for the current frozen1798 policy. B provides wider geometric coverage but reduced frozen-policy task success on both cohorts. This is evidence against using B as a drop-in improvement; it does not test whether training/adaptation on B would improve performance. These256-start cohorts are screening results, not final acceptance.

Fresh A scored229/256 nominal and63/256 wide, versus earlier cached231/256 and54/256. Use the fresh paired reference for this A/B comparison; do not interpret a single small cohort as a precise general accuracy estimate.

## Next useful change

The frozen framing hypothesis did not improve performance. Retain1798/current crop as baseline. The previously measured scarcity of post-lift data remains a supported hypothesis. Next training experiments should address teacher-assisted successful-phase coverage and retention of teacher-labelled examples, or explicitly test adaptation to the full-frame representation against an equally trained A control. Either requires its own controlled experiment; neither was implemented or trained here.

No per-run launch script was created. plan.json, transform_verification.json, manifests, episode JSON files and comparison.json preserve the evidence.
