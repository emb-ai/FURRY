from pathlib import Path
import json,runpy
O=Path(__file__).resolve().parent
runpy.run_path(str(O/'make_report.py'))
s=json.loads((O/'summary.json').read_text());manifest=json.loads((O.parent/'cup-training-plan-20261015/training-manifest.json').read_text());valid={x['episode'] for x in manifest['splits']['valid']}
r=s['rollouts'];keys=[(x['label'],x['episode'],x['execution_steps'],x['seed']) for x in r];assert len(set(keys))==len(keys)
counts={}
for h in (48,8):
 rows=[x for x in r if x['label']=='latest' and x['execution_steps']==h and x['episode'] in valid];assert len(rows)==4
 counts[str(h)]={'success':sum(x['success'] for x in rows),'total':len(rows)}
s['final_validation_counts']=counts;s['complete']=True
(O/'summary.json').write_text(json.dumps(s,indent=2))
report=O/'REPORT.html';html=report.read_text();lead=f'<p><strong>Итог: финальные веса, четыре validation-старта, seed 0: блок 48 — {counts["48"]["success"]}/4, блок 8 — {counts["8"]["success"]}/4. Один train-старт пройден в обоих режимах.</strong> Выбор ранних весов и длинный интервал между наблюдениями ухудшали результат. Это малая диагностика, не приёмка качества или обобщения.</p>'
report.write_text(html.replace('</h1>','</h1>'+lead,1))
entry=f'''

## Проверка pipeline чашки, 2026-10-15

- По запросу «сделай проверку» изолирован outputs/cup-pipeline-check-20261015. Production source/APK, исходные train checkpoints и DexWeave не изменялись. Это диагностика, деплоя нет.
- Записанные команды100Hz: cold HOMIE history2/4 validation-success; восстановление истории по первым .5с записанных физических состояний даёт3/4. В первые1с после префикса ошибка root<2.31e-6м; остающийся эпизод1792017999001122 не проходит и при .8/1с префиксе. Это teacher-forced initialization prefix, далее свободная физика; не автономность от t0 и не полная точность динамического replay. Native MJB vs тот же архивный episode отличается только body_invweight0 на3.64e-12; режим теней renderer отдельноOFF.
- MIPT diagnostic118481 COMPLETED5:00exit0:0: fresh R3M/U-Net, одна train-demo1792014836827567, без аугментаций/commandnoise,3000updates. На6учебныхокнах raw6.35°/EMA7.24°; лучший generated-MSE checkpoint rawstep2800. Closed-loop48 coldseed0 не прошёл90с. Это недостаточно сошедшийся короткий overfitcheck, не доказательство сломанного обучения.
- MIPT checkpointcomparison118485 COMPLETED10с exit0:0.12окон на первых4train и12 на4valid, t0/1/2с,48commands. Epoch15EMA armsMAE13.35°train/15.90°valid,Grip.1377/.1380; epoch300EMA .990°/11.013°,Grip.00288/.02229; raw аналогичен. Noise-MSE selection пропустил более точные по действиям финальные веса. Эти числа — команды, не physicaltrackingerror и не rolloutsuccess.
- Финальная EMA compact checkpoint SHA2323b53f8ece620f2d0bbe9d0aa1495a3dcd0e296010079ab76ecd72629a0ffd; oneepisodebest SHA cc593ae007062ba69ad2eb145f26f499083ffb7d8ff6ebe4c49e105216e7afed. Локальные SHA совпали сMIPT. Originalbest не перезаписан.
- Autonomous cold-start MuJoCo/HOMIE,30Hzcommands,DDIM10,seed0,90с: финальные веса validation execute48={counts['48']['success']}/4, execute8={counts['8']['success']}/4. Первый train-start проходит при48 за8.7с/XY1.74см и8 за9.5с/2.18см; epoch15 на том же train не проходит ни48,ни8. Первыйvalid1792015107773946 проходит только final+8 за37.4с/3.79см (demo6.87с); final+48 иearly+8 не проходят. После первогоphysical snapshot никакие будущие demo state/action не подставляются. Observer previousimage/command сохранён t-1,t при смене execution. Один seed и одна capture-сессия, latency симуляцией не учитывается.
- Все4final+48 validation закончились чашкой на боку (uprightcos~.001); у3 не выполнен lift>4.5см. Успех — штатный collection_task: подъём, положениевцели,vertical>.94,отсутствиеконтактарук,lowvel и .5сsettling. Нельзя трактовать один успех как готовую обобщающую политику.
- Samplercheck раннихвесов на12validwindows: DDIM10arms15.16°,DDIM50 14.37°,trailing10 15.42°,DDPM50 11.55°. Это другой seedbatch/device, напрямую не сравнивать с GPU checkpointcomparison; DDPMclosedloop не запускался.
- REPORT.html/summary.json содержат все результаты. early-vs-final-train.mp4 —11с/20fps/1280x460,3D replay двух autonomous train-rollouts; после8.7с успеха справа freeze. Preview и полноеffmpegdecode проверены. Проверенные изменения — диагностика выбора весов и execution; новой общедатасетной тренировки и установкиAPK не было.
'''
ag=O.parent.parent/'AGENTS.md'
if '## Проверка pipeline чашки, 2026-10-15' not in ag.read_text():
 with ag.open('a') as f:f.write(entry)
print(json.dumps(counts))
