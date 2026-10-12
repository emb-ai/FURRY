# HOMIE / SONIC on FURRY clips

All numbers below are counterfactual MuJoCo simulations, with catch-up enabled. Baseline: TWIST2 quest-111586-500 at 50 Hz.

## validation

| Mode | Root XY mean / p95, cm | Slip, cm/s | Low moving foot, % | Wrist world / root-local, cm | Torso rotation, deg | Upward contact candidates | Falls |
|---|---:|---:|---:|---:|---:|---:|---:|
| TWIST2 +500 | 76.3 / 243.6 | 5.07 | 25.2 | 77.1 / 4.1 | 16.8 | 3 | 0/12 |
| HOMIE v2 | 44.4 / 182.8 | 4.83 | 15.6 | 47.2 / 5.0 | 7.7 | 0 | 0/12 |
| SONIC VR3 + planner | 37.0 / 127.9 | 2.68 | 16.2 | 39.7 / 3.5 | 17.0 | 0 | 0/12 |

Duration per mode: 174.12 s. Frame counts match clip-by-clip; no fall-conditioned truncation.

## live

| Mode | Root XY mean / p95, cm | Slip, cm/s | Low moving foot, % | Wrist world / root-local, cm | Torso rotation, deg | Upward contact candidates | Falls |
|---|---:|---:|---:|---:|---:|---:|---:|
| TWIST2 +500 | 39.2 / 100.2 | 4.78 | 28.3 | 39.6 / 2.4 | 18.8 | 4 | 0/2 |
| HOMIE v2 | 12.2 / 23.9 | 3.43 | 12.9 | 14.2 / 3.2 | 10.3 | 0 | 0/2 |
| SONIC VR3 + planner | 19.7 / 46.2 | 2.81 | 17.5 | 23.6 / 2.4 | 21.8 | 0 | 0/2 |

Duration per mode: 58.44 s. Frame counts match clip-by-clip; no fall-conditioned truncation.

## Metric definitions

- Root XY: error against the retargeted reference path, without post-hoc alignment. Live clips additionally report ego-camera XY error against recorded HMD goals in results.json.
- Slip: load-weighted tangential speed at actual sole-floor contact points, including angular foot motion; zero when unloaded. Compare together with contact and low-foot diagnostics.
- Low moving foot: fraction of foot-samples with exact sphere-sole clearance <1cm and horizontal foot-body speed >0.15m/s; normal landing/take-off may contribute.
- Braking contact: new foot load >10N, previous speed >0.15m/s, reduction >0.08m/s in 20ms, 150ms event debounce. This includes normal touchdowns.
- Upward contact candidate: braking contact plus previous upward foot-body speed >0.04m/s and sole gap >3mm. It is a narrow detector, not a complete or validated count of stumbles.
- Upper-body metrics: common GMR targets, world and full root-local positions; quaternion geodesic wrist/torso angles and 17 upper-body joint RMSE.

## Limitations

- Diagnostic previously reused validation, not independent test.
- Two live recordings are new counterfactual simulations, not headset screen captures.
- No measurement of true human leg intent. Three-point targets come from common robot-scaled GMR FK.
- HOMIE and SONIC retain respective author PD and motor dynamics; common scene, contact geometry, 50Hz control, 1000Hz physics.
- Trip candidates are unlabelled 50Hz contact proxies. Ordinary braking touch-downs must not be interpreted as confirmed stumbles.
- Main table uses planner seed11; seed22 checks planner sampling sensitivity, not training variance.
- SONIC wrapper replans synchronously at 5Hz with world-frame commands; this is a desktop adapter, not a timing-identical reproduction of asynchronous author deployment.
- No Quest installation or Android performance acceptance.

## SONIC planner seed sensitivity

| Set | Seed | Root XY mean, cm | Slip, cm/s | Upward contact candidates | Falls |
|---|---:|---:|---:|---:|---:|
| validation | 11 | 37.0 | 2.68 | 0 | 0 |
| validation | 22 | 38.1 | 2.62 | 0 | 0 |
| live | 11 | 19.7 | 2.81 | 0 | 0 |
| live | 22 | 18.0 | 2.48 | 0 | 0 |

Contact-candidate snapshots were inspected. They include edge contacts/rocking and do not establish a human-labelled stumble count; see contact-candidates.png. Event thresholds were clarified during this pilot, not pre-registered acceptance thresholds.
