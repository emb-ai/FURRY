# Итог1GPUдо5000, 2026-10-10

На проверке22:57:45МСК обаtrainer и обеevaluation pipelines завершены. Control/faults:5000actorupdates, phase complete/pipeline_complete. Продолжения1000→5000 заняли2:35:50/2:37:17 чистого trainer walltime. Все промежуточные checkpoints1500..5000 сохранены в собственных серверных run directories. Deployment=false.

**Ни один из16одногпу checkpoint-кандидатов не прошёл все прежние primary50Hzvalidationcriteria.** Обучение улучшило перемещение, но gait-regressions сохранились.

Baseline — прежняяpolicy111260-1000, не новый pre-migrationseed1000. Primaryvalidation:12Questheld-outclips×seeds0/1,50Hz,matchedprefixes. Этот holdout уже многократно использовался, не независимыйtest и не живаяQuestприёмка.

| Финальный5000 | Контроль | Actor leg faults |
|---|---:|---:|
| Accepted | false | false |
| Причины | key_marker/sliding regression | key_marker/sliding regression |
| Новые падения относительноbaseline | 0 | 0 |
| Slide rate vsbaseline | +17.83% | +3.70% |
| Slip velocity vsbaseline | +6.16% | +3.48% |
| Детектированныеtrips | 86vs73 | 103vs73 |
| Key-marker RMSE vsbaseline | +6.93% | +9.98% |
| Window displacement error | 7.80см vs9.33см | 7.94см vs9.33см |
| RootXYerror | 1.475м vs3.594м | 1.747м vs3.594м |

Одинаковая продолжительность финальныхgaitprefixes324.44с; trips — детектор поreplay, не человек-наблюдатель. Legacy stumble=0 не означает отсутствиеtrips. Нет новыхпадений не означает нулевое общее число падений. Уfaults1500/2000/2500 были новыепадения; в3000..5000 их относительноbaseline нет.

Более ранниеcheckpointне дают принятого кандидата:control3500 имеет81trip vs73 иslide+5.75%;faults4000 имеетslide+1.32%, но key-markerRMSE+15.29% и86trips vs73. Их нельзя назвать принятыми по выбору удобной метрики. Они остаются диагностическимикандидатами.

Last100trainmeans:criticloss0.01500/0.01618,EV0.99414/0.99411,std0.41075/0.42657,trainreward7.54693/7.47477. Этиtrainloss/rewardне подтверждают качествопоходки. Увеличение продолжительности до5000 не устранило расхождениеtrainobjectiveиruntimegaitmetrics; причинность конкретных reward/physics/data решений не установлена.

Результаты подтверждают завершение запрошенного обучения, а не исправлениеgait. Требуется разбор соответствия train/replay reward и контактов, а не обещание улучшения от дополнительныхiterations. Исходныеweights сохранены, новаяполитика не устанавливалась наQuest.

Свидетельства: `single-gpu-status.json`, `single-gpu-evaluation-summary.json`; [итоговыеlossкривые](loss-curves.png). Серверныеweights: `/home/jovyan/vlad/FURRY/twist2-distributed-20261010/runs/control-h100x1-5k/training/checkpoint_005000.pt` и `runs/faults-h100x1-5k/training/checkpoint_005000.pt` в том же корне; также final.pt. Дополнительныеcompatibility/stressпробы завершились вместе сpipeline, но их количественные итоги в этот отчёт не извлечены.
