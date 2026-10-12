# Train diagnostics, 2026-10-10

Both single-GPU runs completed 5000 actor updates. This report records the successful remote read of `single-gpu-train-summary.json`; subsequent attempts to copy that JSON timed out during SSH banner exchange.

## Training logs

Last-100-update means before the restart at update1000 versus final update5000:

| Metric | Control 1000 → 5000 | Faults 1000 → 5000 |
|---|---:|---:|
| Reward/second | 7.0398 → 7.5469 | 6.8984 → 7.4748 |
| Window displacement error, m | 0.100799 → 0.090064 | 0.099338 → 0.089809 |
| Value loss | 0.017126 → 0.015001 | 0.018147 → 0.016177 |
| Critic explained variance | 0.992787 → 0.994137 | 0.992392 → 0.994113 |

## Train stage05 runtime probe

This is a MuJoCo evaluation of the stage05 train subset, not the whole training mixture. Comparisons use baseline policy111260-1000, matched prefixes with 23752 samples and 231.92 seconds of gait observations. The baseline differs from the seed checkpoint used to restart these runs.

| Metric | Baseline | Control5000 | Faults5000 |
|---|---:|---:|---:|
| Gait trips detected | 66 | 49 | 49 |
| Window displacement error, cm | 11.0487 | 7.7334 | 8.3896 |
| slide_rate | 0.066773 | 0.106833 | 0.097465 |
| slip_m_s | 0.098145 | 0.086209 | 0.081100 |
| Key body RMSE, m | 0.019612 | 0.018828 | 0.019213 |
| Tilt, degrees | 4.0848 | 6.3921 | 7.6895 |

Both candidates have no new falls relative to baseline and both fail the existing acceptance criteria solely for sliding_regression in this probe. slide_rate worsens by approximately60.0%/46.0%, while slip_m_s improves; these are different diagnostics and must not be conflated. Legacy stumble=0 is not evidence of zero trips.

Tracking and detected trips improve on this train subset, but sliding already regresses there. Held-out validation trips at5000 are86/103 versus baseline73, so the train subset improvement does not establish generalization. These observations alone do not prove overfitting or identify a causal failure mechanism. No new policy was deployed to Quest.
