# Unitree G1: MuJoCo на Quest 3 и macOS

**Автономный Quest 3:** `android/build/g1-quest.apk`, приложение **G1 Quest Lab**. MuJoCo, ONNX Runtime и OpenGL ES работают внутри шлема; компьютер и сеть после установки не нужны. Сборка и управление установкой описаны в [android/README.md](android/README.md).

Ранее на Quest после исправления индексов визуальных сеток измерено 72 FPS, в том числе с passthrough, при 68 091 треугольнике на глаз. Новая геометрия содержит **89 629 треугольников на глаз**; лимит 100 000 проверяется сборкой и рендерером. FPS новой версии на шлеме ещё не измерен. Подробности визуальных сеток и гибридных коллизий: [Geometry](assets/g1/README.md).

Реальная комната видна через цветной passthrough; виртуальный пол скрыт. **A** калибрует движения контроллеров и включает управление руками через нативную IK и политику TWIST2; **B** приостанавливает/продолжает трекинг. Grip каждого контроллера сжимает соответствующую кисть, меню левого контроллера сбрасывает сцену. **X** перезапускает симуляцию, **Y** начинает/заканчивает отдельную запись движений человека; стики не назначены. Поднятие чашки, Kick-T, полный PICO/GMR-трекинг тела ещё не реализованы. Соответствие движений рук пока неудовлетворительное по проверке оператора; запись предназначена для его диагностики.

## Настольная версия

Локальная симуляция G1 с готовой политикой TWIST2 на CPU. Проверена на MacBook Air M4, 24 ГБ, macOS 26.5.2. Unity, CUDA, Redis и PyTorch для этого запуска не нужны.

## Запуск

Откройте `run_sim.command` двойным щелчком в Finder. Или из папки проекта:

```sh
./run_sim.command
./run_sim.command --scene cup
./run_sim.command --scene push_t
./run_sim.command --scene stand --demo
```

По умолчанию открывается `lab`: G1, стол с чашкой, T-коробка и целевая разметка. Физика — 1000 Гц, политика — 100 Гц, синхронизация окна запрашивается с частотой 60 Гц. Робот свободно стоит на полу: основание не закреплено и не поддерживается скрытой силой. Приводы тела получают моменты от PD-контроллера, целевые углы выдаёт обученная политика TWIST2.

| Клавиша | Действие |
|---|---|
| F8 | Сброс робота, предметов и истории контроллера |
| F9 | Пауза / продолжение |
| F10 | Плавная демонстрация движений рук / завершить текущий цикл |
| F11 | Закрыть / открыть пальцы обеих кистей |
| F12 | Камера от первого лица / общий вид |
| Tab | Стандартная панель MuJoCo |

На клавиатуре Mac для F8–F12 может потребоваться удерживать **fn/🌐**. Вращение камеры — перетаскивание левой кнопкой мыши, масштаб — прокрутка. Выход — закрыть окно. Буквенные команды стандартного MuJoCo меняют настройки отображения; наши команды используют F8–F12, чтобы не конфликтовать с ними. В пассивном viewer управляйте паузой через F9.

## Что готово

- Динамическая модель G1: 29 суставов тела и 14 суставов пальцев.
- Контроллер TWIST2 с CPU inference, устойчивое стояние и небольшие движения рук.
- Отдельные сцены `cup`, `push_t`, `stand` и общая сцена `lab`.
- Чашка с полостью и ручкой, стол, цельная T-коробка из двух коллизионных боксов. Предметы движутся под действием физики.
- Запись фактических состояний и команд; воспроизведение сохранённых состояний.
- Общий и монокулярный вид от первого лица в настольном окне.

Сцены подготовлены для последующей разработки задач. Автоматический захват/поднятие чашки, управление ногами для Kick-T и критерии успеха задач пока не реализованы. Демонстрация рук не является демонстрацией захвата. Политика TWIST2 — контроллер движения, а не готовая политика решения этих задач.

## Запись и воспроизведение

```sh
./run_sim.command --scene cup --demo --seconds 10 --record outputs/episode.npz
./run_sim.command --replay outputs/episode.npz
```

NPZ содержит время симуляции, `qpos`, `qvel`, `ctrl`, 35-мерные референсные команды TWIST2, команду пальцев, XML сцены и метаданные. Частота записи — 100 Гц. Воспроизведение показывает записанные состояния с `mj_forward`; это не повторный физический rollout команд. Во время записи сброс отключён, чтобы время внутри эпизода оставалось монотонным. Пути к исходным mesh-файлам в XML абсолютные: для переноса эпизода на другой компьютер потребуется перенести assets и обновить путь.

## Проверки и измерения

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m g1_sim --headless --seconds 30 --scene lab --report outputs/benchmark.json
./run_sim.command --seconds 20 --demo --report outputs/viewer_benchmark.json --screenshot outputs/g1_lab.png
```

Интеграционные проверки: 60 секунд равновесия с движениями рук и пальцев, ограничение наклона/дрейфа, отсутствие предупреждений MuJoCo, восстановление состояния после reset, падение чашки на стол, физический контакт и сдвиг T-коробки силой.

На этой машине первая проверка сцены с пальцами дала 30 секунд симуляции за 4,68 секунды без окна (6,4× real time). С окном — 20 секунд симуляции за 20,56 секунды с учётом запуска viewer и ограничения скорости; p95 inference около 0,56 мс. Это измерения текущей настольной сцены, не тест VR-задержки и не длительный тепловой тест.

## Установка заново

Нужен Python 3.12 для Apple Silicon. Используйте его явный путь вместо `python3.12`, если команда отсутствует:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
git clone https://github.com/amazon-far/TWIST2.git vendor/TWIST2
git -C vendor/TWIST2 checkout b06178f19a22f2138cbd31f60c6d494bc263f67d
chmod +x run_sim.command
./run_sim.command
```

Launcher учитывает пробелы/кириллицу в пути и расположение `libpython` для установленного через uv Python. Он вызывает официальный `mjpython`, необходимый для нативного окна на macOS.

## Источник и адаптация

- [TWIST2](https://github.com/amazon-far/TWIST2), commit `b06178f19a22f2138cbd31f60c6d494bc263f67d`.
- Политика: `assets/ckpts/twist2_1017_20k.onnx`.
- Модель: `assets/g1/g1_sim2sim_29dof_with_hands.xml`.
- Контроллер адаптирован из `deploy_real/server_low_level_g1_sim.py`: сохранены порядок наблюдений, 10 кадров истории, масштабы, коэффициенты PD и ограничение моментов.
- Адреса суставов ищутся по именам, потому что кисти и свободные предметы меняют раскладку `qpos`.
- Для варианта с пальцами восстановлены параметры `armature` суставов тела из модели `g1_sim2sim_29dof.xml`, соответствующей upstream sim2sim. Исходный вариант с руками пропускает часть инерций приводов; без исправления в проверке возникал заметный дрейф.
- Оригинальные файлы в `vendor/TWIST2` не изменены. Сцены собираются в памяти, адаптации находятся в `g1_sim`.
- `--fixed-hands` использует исходную модель с 29 приводами без подвижных пальцев.

Исходная лицензия TWIST2 — MIT, Copyright (c) 2025 Yanjie Ze; см. `vendor/TWIST2/LICENSE` и `THIRD_PARTY_NOTICES.txt`.

## Review Verification

Locally reproduced on Apple Silicon macOS, Python 3.12, on 2026-10-05 from a
fresh environment with `requirements.lock` and the pinned vendor revisions:

- `python -m unittest discover -s tests -v`: 13 tests, including the 60-second balance/reset/contact checks and synthetic recording-corruption regressions.
- `./android/check_recording_mac.sh`: 500 input/state rows, 20 invalid inputs, two calibrations and one reset; 430 references replayed with zero divergence. The interrupted-recorder check passed.
- `./android/check_tracking_mac.sh`: native controller/IK regression; results are synthetic, not a human-fidelity measurement.
- `python scripts/prepare_android.py`: regenerated assets and XR lifecycle; 68,091 triangles per eye, below the 100,000 budget.
- `tests/check_tracking_space.cpp`: reference-space changes apply at their announced XR time, including delayed, duplicate and multiple pending events.
- Repository Markdown/link checks and all 16 documentation-tool tests passed.

Commands above use `.venv/bin/python`; on this host the native checks use
`DEVELOPER_DIR=/Library/Developer/CommandLineTools` because the selected Xcode
installation requires license acceptance. No license was accepted by automation.
The `Quest G1 Prototype` CI workflow covers the Python checks and native
reference-space timing test independently of headset hardware.

Review adaptations invalidate calibration on headset recenter, stop the worker
on XR exception paths, preserve incomplete recordings after errors, retain the
original receive time of the recording's initial input, and reject inconsistent
recording schemas/validity/completion markers. Recalibrate with A after recenter.

An Android APK rebuild and in-headset controls, recentering, performance and
thermal behavior have **not** been reproduced in this review. Earlier headset
measurements above are the contributor's report, not new review measurements.
The prototype's CSV format does not resolve ADR-005 or complete the research
data contract. Private recordings stay local; no Wiki publication is performed.
