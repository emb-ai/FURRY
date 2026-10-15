# Quest camera legs — kicks (dataset-v3-kicks, 2026-10-14)

Push-T kicks from the camera skeleton, in the same format as dataset-v3
(`quest-g1-motion-v1` clips plus a command bank), tagged
`source: quest-camera-kicks`. The separate tag lets `author_dataset.py
--quest-shares` give kicks their own share of the Quest mass. Inside
dataset-v3, kicks would get only about 1/34 of the camera share each.

- Session ep15 (block B, 2026-10-14 17:32–17:45):
  - smooth app, Push-T scene, observer view;
  - policy quest-camv2-500, catch-up OFF, so `mimic.csv` holds the operator's
    command without any headset correction;
  - camera legs live 100% of the time.
  - 18 of 21 episodes were kept; three shorter than 10 s were dropped.
  - Kicks are straight-leg, about every 4–6 s: hip flexion 25–45° and knee
    10–25° at the peak. There were 166 kicks across the episodes plus the ep13
    kick episode. Meta legs show 4 of them.
- ep13 1791982452458398 (15:52, camv2-250, catch-up off) holds deeper kicks
  from a static lunge (hip 60–80°, knee up to 117°). On the headset the robot
  ran away after one of them. It is used for validation.
- Stages 201–219.
  - Train: 30 clips, 584 s of originals plus mirrors.
  - Validation: stages 201 (ep13 1791982452458398) and 217 (ep15
    1791988970896362), 91 s.
- Usable share is 0.86–0.97 per episode. Live and replayed legs differ by
  ≤ 1.2° median.
- Cleaning keeps the kicks: 160 of 166 raw kick peaks are in clean3, and the
  peak differs by 1.1° median (p90 2.1°). The leg speed maximum is 11.3 rad/s,
  below the 12 rad/s curation ceiling, so no kick was cut.
- Shin bias: the afternoon camera placement, `--shin -5.2 -2.0 -4.4 3.8`. See
  `../dataset-v3/README.md`.
- The bank equals the live `mimic.csv` exactly on all 30 original train clips.

RGB-D for the session was recorded but is not included.
