# Quest locomotion: результаты продолжений и абляций, 12 октября 2026

Восемь независимых обучений завершили по 1000 новых actor updates, но ни один из 80 checkpoints не прошёл заранее заданные совместные критерии походки. Ослабление tracking ног уменьшило диагностические спотыкания относительно контрольного дообучения в обоих training seeds; одновременно контактное скольжение ухудшилось. Это частичный эффект, а не исправленная походка.

## Финал новой матрицы

Общий baseline — checkpoint `111260/1000`. Таблица показывает финальные 1000 новых updates для training seeds 11 / 22. Первичная оценка: одна физика MuJoCo, actor 50 Гц, 12 Quest held-out клипов × evaluation seeds 0 / 1, matched prefixes. Baseline имеет 73 trips и 16 braking trips на этих префиксах.

| Вариант | Trips, seed 11 / 22 | Изменение slide rate, seed 11 / 22 | Новые падения, seed 11 / 22 |
|---|---:|---:|---:|
| Контроль | 93 / 98 | −1,68% / +19,17% | 0 / 0 |
| Lower-body errors × 0,25 | 87 / 68 | +22,66% / +39,02% | 0 / 0 |
| Airtime при yaw-only | 90 / 97 | +23,17% / +24,77% | 0 / 1 |
| Оба изменения | 83 / 79 | +31,55% / +27,34% | 0 / 0 |

Для yaw seed 22 новый failure — `stage14-run01-003000-part00__seed0`; matched prefix сокращён до 323,77 с против 324,44 с у остальных финалов. Подробные paired показатели сохранены для каждого checkpoint отдельно.

Ранний yaw seed 11 / update 200 дал trips 73→61, braking 16→15 и slide rate −2,86%, но directed progress ratio упал на 30,20%; кандидат не принят. Ratio не означает долю успешно пройденного маршрута. Выбор только по финальному checkpoint или одной метрике скрывает этот компромисс.

## Постановка и проверки

Каждый run использовал одну H100 80GB и 4096 сред; это восемь отдельных PPO runs, не распределённый PPO. Actor/std/frozen normalization инициализированы одним полным PT checkpoint, SHA256 `5bbfbddf34b19d8be8c7b37ecfbe61cdef83f03c759f22305fb64d0ee0e06a7f`; critic/normalization/Adam новые у всех условий. Critic readiness — существующая проверка на 500–1000 updates, далее 1000 actor updates и оценка каждые 100. Trainer завершился 11 октября около 00:36–00:40 МСК, pipelines — до 00:55 МСК.

Общие условия: nonlooping clips, sphere soles и существующий window displacement/turn reward. Actor input leg faults в этой матрице выключены. Source DaggerPPO/std/LR/KL/dropout/DR сохранены; source commit TWIST2 — `b06178f19a22f2138cbd31f60c6d494bc263f67d`. Новые whole-run early stops или автоматическая установка политик не добавлялись.

Coefficient 0,25 применяется при locomotion к 12 leg DOF position/velocity errors, четырём нижним key-body local/global errors и pose termination. Верхние суставы, руки/корпус и safety/root/timeouts сохраняют прежние компоненты. Coefficient 1 не заменяет исходные формулы. Yaw-only отдельно расширяет условие активности исходного неположительного airtime penalty. CPU source checks и реальный GPU smoke 4096 envs / 2 critic + 1 actor update прошли; полные восемь конфигураций сверены с запуском и source PPO.

Validation ранее многократно использовалась и не является независимым test. Trip detector использует reference swing phase, которая может расходиться с корректирующим шагом; `stumble=0` не означает `trips=0`. Ни один новый кандидат не установлен на Quest. Clearance reward, contact-point sliding reward, recovery и новая программа action/noise disturbances этой матрицей ещё не проверены.

## Файлы для разбора

- [Результаты всех 80 checkpoints](results-summary.json): counts, причины diagnostic rejection, selected matched gait/progress/upper-body metrics. Это компактная сводка, не NPZ traces или полные training logs.
- [Подробный итог и ранние сигналы](results.md), [план и формулы](experiment.md), [время завершения и train last50](timing.json).
- [Общие исходные веса, параметры восьми условий и SHA256 кода](provenance.json), [ancestry/split audit](ancestor-check.json), [проверка diagnostic gate](gait-gate-test.json).
- Код использованных изменений: [reward/termination hooks](code/locomotion_ablation.py), [trainer](code/locomotion_finetune.py), [pipeline](code/run_locomotion_pipeline.py), [runtime evaluation](code/locomotion_eval.py), [diagnostic gate](code/locomotion_verdict.py), [source checks](code/test_locomotion_source.py), [launcher](code/launch_locomotion.py), [status reader](code/status_locomotion.py). Это архив фактически использованного кода с историческими server paths, а не изменение штатного runtime/trainer в main.
- Предыдущие продолжения до 5000: [итог](previous-5k/report.md), [16 checkpoint comparisons](previous-5k/results-summary.json), [train-stage05](previous-5k/train-diagnostics.md), [loss curves](previous-5k/loss-curves.png) и [данные графиков](previous-5k/loss-curves.json).
- Историческая проверка 1 vs 4 GPU: [аудит](previous-5k/migration-audit.md), [paired summary](previous-5k/migration-audit-summary.json), [source PPO fixed-batch check](previous-5k/source-ppo-fixed-batch-check.json). Датированный RUNNING в историческом аудите не является актуальным статусом.
- [SHA256 и размеры опубликованных файлов](files.json).

Веса и полные traces остаются в собственном серверном каталоге `/home/jovyan/vlad/FURRY/twist2-distributed-20261010/runs/`. Чат экспортирован отдельно для пользователя и не публикуется вместе с результатами.
