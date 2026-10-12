# Locomotion ablations v1, 2026-10-10

Пользователь явно запросил новые обучения по предложениям присланного отчёта. Это отдельная серия после завершённых control/faults5000; результаты прежних веток не опровергают ещё не проверенные гипотезы.

## Матрица

| Условие | Lower-body error coefficient | Airtime при yaw-only | Training seeds | GPU |
|---|---:|---|---|---|
| control | 1.0 | исходное условие | 11/22 | 0/1 |
| relax | 0.25 во время locomotion | исходное условие | 11/22 | 2/3 |
| yaw | 1.0 | включён | 11/22 | 4/5 |
| relax-yaw | 0.25 во время locomotion | включён | 11/22 | 6/7 |

Каждый run — одна H10080GB,4096envs,50Hz,24steps,98304samples/iteration и исходные20PPO minibatch updates. Бюджет1000 новых actor updates, сохранение/evaluation каждые100. Это восемь независимых запусков, не distributed PPO. `uv run --no-project` использует существующие изолированные env/eval-env; исходный server path `/home/jovyan/vlad/FURRY/twist2-distributed-20261010`, новый code directory `quest-locomotion-ablations-20261010/`. Чужие процессы, каталоги и system environment не изменяются.

Общая исходная политика — полный checkpoint111260/1000, `comparison-baseline.pt`, SHA256 `5bbfbddf34b19d8be8c7b37ecfbe61cdef83f03c759f22305fb64d0ee0e06a7f`. Actor, frozen normalization и learnable std берутся из него; critic/critic normalization/Adam новые у всех четырёх условий. Critic-only prefit использует уже существующую readiness-проверку500..1000updates. Это новый reward experiment с общим actor, не точный resume старого optimizer. Исходные source DaggerPPO/KL/LR/dropout/std/DR сохранены; whole-run KL stop или stop-on-validation-regression не добавлялись.

Общие изменения — nonlooping motion handling, sphere soles и существующий1swindow displacement/turn reward. Поэтому контроль означает продолжение общей экспериментальной постановки без двух новых изменений, а не полностью неизменённый source trainer. Actor-only input leg faults в первой матрице выключены, чтобы отделить эффект свободы шага от robustness curriculum.

## Формулы и границы изменений

Locomotion определяется по референсному XY-speed>0.05m/s или абсолютной yaw angular velocity>0.15rad/s. В relax коэффициент умножает компоненты squared errors12leg DOFs и4lower key bodies в существующих position/velocity и local/global rewards; upper-body components и root rewards прежние. Те же нижние key-body squared distances масштабируются перед максимумом pose termination. Верхние key-body thresholds, safety/root/timeouts остаются исходными. L1 diagnostics также согласованы. Вне locomotion coefficient1.0; при coefficient1.0 source reward/termination methods вообще не заменяются.

Yaw ablation меняет только условие активности исходного airtime: добавляет yaw-only к исходной линейной активности. Исходная формула остаётся неположительным штрафом за короткий перенос, не новой положительной наградой за подъём. Изменения выполнены process-local до создания среды и binding reward functions; исходный TWIST2 на диске не правился.

Новых входов actor или сигнала доверия ногам нет. Эта серия предназначена для locomotion; intentional kick control не считается решённым. Ничего не установлено на Quest.

## Проверки перед запуском

- CPU tests на реально закреплённых source methods: identity coefficient1, неизменность idle и upper body, lower pose termination relaxation только при locomotion, сохранение upper/safety termination, airtime для yaw обоих знаков — passed.
- Реальный Isaac Gym GPU smoke с4096envs, coefficient0.25+yaw,2critic+1actor update завершён. Actor неизменен во время critic warmup по встроенной проверке; source PPO выполнил update и сохранил final.pt. Это тест исполнения, не результат улучшения походки.
- Upper-body/gait evaluation smoke на отдельном MuJoCo процессе прошёл. Новые metrics не меняют физику или вход политики.
- Ancestor dataset audit:1090train identities сохранены, validation12/stages2,9,14,ancestor_train_in_validation0. Исходный manifest hash `e887731e2cb3c8c50c9de77634ef2cdf0640cc2cd2e8f98ad57f2d3850732b55` совпал с checkpoint.
- Diagnostic gate tests: одинаковая политика не считается улучшением; ухудшение trips или upper-body marker error отражается в reasons.
- Перед launcher все8GPU имели1MiB/0%util, compute processes отсутствовали. Launcher повторно проверил это непосредственно перед dispatch.

## Оценка

Одинаковый MuJoCo runtime50Hz,12Quest held-out clips×2evaluation seeds0/1. Это многократно использованный holdout, не независимый test. Train-stage05 и compatibility100Hz оцениваются отдельно в конце pipeline; input-fault probes остаются отдельными диагностическими прогонами.

Сохраняются legacy comparison.json и отдельный gait-verdict.json. Новый диагностический gate заранее фиксирует отсутствие новых падений, trips/braking rate≤baseline+5%,contact sliding≤baseline+2%,window displacement/yaw/lateral errors≤baseline+5%,upper-body RMSE≤baseline+5%,directed progress не хужеbaseline более5%; также прежние checks more_falls/less_survival. Улучшение хотя бы одного trips/braking/sliding/window показателя≥5% требуется для gait_accepted. Это конвенция отбора для проверки гипотез, не доказательство статистической значимости. Gate не останавливает PPO и не устанавливает веса автоматически. All-body key RMSE остаётся диагностикой, но не подменяет отдельную проверку рук при намеренном ослаблении tracking ног.

Trip detector использует фазу из референса и имеет прежние ограничения; stumble0 не означаетtrip0. Upper-key metrics относятся к двум wrists, elbows и torso runtime-маркерам, не анатомической голове. Сравнения кандидатов используют одинаковые префиксы; полная survival также сохраняется. Training seeds11/22 отличаются от evaluation seeds0/1.

## Следующая часть отчёта

Clearance reward, контактный slip reward с вращением/нагрузкой, явные finetune action/noise schedules и recovery/силы через руки не объявляются проверенными этой матрицей. Clearance требует верифицированной swing phase; contact-point approximation в Isaac требует сравнения с точной MuJoCo геометрией. Эти пункты остаются отдельными абляциями следующего этапа, а не отклоняются по результатам текущих двух изменений.

Актуальные server artifacts: `submitted.json`, `status.json`, `ancestor-check.json`, smoke-combined/ и evaluation-smoke.* в новом code directory; training/validation artifacts — в `runs/locomotion-v1-*/`. Первоначальный dispatch подтверждён для всех восьми условий. На первом статусе все живы, baseline выполняется; начало trainer проверяется отдельно.

## Подтверждённый старт trainer

После baseline все8pipeline живы и выполняют critic updates: на срезе14..16critic iterations, actor_updates0. GPU memory10.84GB,util69..80%. В7runs посещены все1090motions, вrelax-s22 пока1089. Yaw-only samples составляют около7% всех текущих samples; это подтверждение воздействия абляции на посещаемые состояния, не результат улучшения поворотов.
