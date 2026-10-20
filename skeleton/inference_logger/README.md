# Сборщик скелета камеры и траекторий Quest

Назначение: собрать **одну связанную сессию** для сопоставления скелета RGB-D / AprilTags
с HMD, контроллерами, Meta-скелетом, reference и фактическим движением робота.
Это отдельный модуль; существующие скрипты, APK, модели и старые записи не изменены.
Обучения нет.

## Что сохраняется

| Файл | Данные и смысл времени |
|---|---|
| `raw.bag` | Нативная запись RGB + depth SDK RealSense до inference, без overlays; calibration в bag |
| `frames.jsonl` | ID кадров RGB/depth, оба device timestamp/domain, host acquisition monotonic/epoch; индекс захваченных framesets |
| `skeleton.jsonl` | Все найденные COCO-17 2D poses/conf, выбранный человек, несглаженные 3D суставы RGB-D, provenance, углы AprilTags и допустимые poses, начало/конец inference |
| `camera_tx.jsonl` | Опубликованные и отправленные пакеты, `frame_id`, `seq`, `session_id`; отправка не означает применение на Quest |
| `quest_rx.jsonl` | Исходный WebSocket payload без изменения, локальное время получения, сегмент соединения |
| `quest.jsonl` | HMD, контроллеры; с патчем — tracking sequence, raw Meta-14, flags, XR/body times и время submit |
| `sync.jsonl` | Четыре времени каждого probe, отдельный сегмент после reconnect |
| `events.jsonl` | Подключения, замены кадров в latest-slot inference, stale downlink, неверные сообщения |
| `bundle/` | Копия модели, исходников сборщика и используемых модулей, переданного layout |
| `quest_episodes/<episode>/` | Полная неизменённая копия локального эпизода, включая дополнительные бинарные потоки/модель, если они есть в экспорте |
| `checksums.json`, `complete.json` | Хеши исходных файлов, завершение и счётчики; при ошибке — `failed.json` |

Inference работает независимо от входящих Quest-пакетов. SDK пишет raw при захвате;
для inference используется один заменяемый кадр без накопления очереди. Пропуски
inference отмечаются отдельно и не выдаются за пропуски raw. Успешную запись bag
подтверждает отдельное полное чтение, а не сам вызов SDK. Очередь JSONL ограничена;
её переполнение прекращает запись с ошибкой. Отключение камеры завершает сессию;
после подключения запускайте новую — ID захвата не переиспользуется.

Скелет: несглаженный RGB-D, без денойзера и без подмены суставов центрами меток.
AprilTags сохраняются отдельно. `window` — измерение поверхности по depth;
`line` — восстановление по кости. При нескольких людях все 2D poses сохраняются,
но 3D выбор человека отключён, чтобы не менять личность незаметно.

## Окружение

Запускайте команды из `FURRY/skeleton`. Нужны Python 3.10+, numpy, OpenCV с aruco,
pyrealsense2, ultralytics, websockets с sync API и backend модели (OpenVINO для
соответствующего экспорта). Используйте зависимости `skeleton/requirements.txt`;
websockets при отсутствии установите отдельно. Веса YOLO задаются через `--model`
или `-Model`; в папку сборщика они не включены. Геометрия тегов включена в пакет,
RGB-D fusion используется из соседнего `skeleton/fusion.py`.

В PowerShell можно явно выбрать интерпретатор:
`./inference_logger/run.ps1 -Python <python.exe> -Model <model-path> -Output <new-session>`.
Калибровочный layout также указывается своим путём; пример из предыдущей машины
не входит в этот пакет и не является подтверждённой калибровкой текущей съёмки.

## 1. Подготовка Quest

**Старый APK совместим с downlink, но его uplink недостаточен для строгого
связывания по времени/sequence.** Патч `quest_sync.patch` добавляет four-timestamp
ответы и поля `input_sequence`, `quest_monotonic_ns`, `xr_time_ns`, `body_time_ns`,
Meta-14. Значения ns и sequence передаются строками. Поведение политики не меняется.

Патч подготовлен для найденного `../prototypes/quest_g1/android/native/camera_stream.cpp`.
Он **не применён**, APK **не собран и не установлен**. Сначала примените патч,
соберите и установите APK штатным способом проекта. Из каталога `FURRY/skeleton`:

```powershell
git -C ../prototypes/quest_g1 apply --check ../../skeleton/inference_logger/quest_sync.patch
git -C ../prototypes/quest_g1 apply ../../skeleton/inference_logger/quest_sync.patch
```

Для другой ревизии генератор проверяет конкретные точки вставки и отказывается
от неизвестного формата, не редактируя исходники:

```powershell
python -m inference_logger quest-patch <camera_stream.cpp> --output <new.patch>
```

В Quest укажите `ws://<адрес ПК>:8765`. Во время съёмки обязательно включите
**локальную запись эпизода Quest**: robot/reference/body не зависят от доставки uplink.
Отдельный старый сервер камеры на этом же порту запускать не нужно.

## 2. Запись на этом компьютере

Закройте другой процесс, использующий RealSense. Команды из каталога `FURRY/skeleton` в активированном Python-окружении:

```powershell
./inference_logger/run.ps1 -Output inference_logger/sessions/take01 -Seconds 60
```

Размер и закрепление меток нужно проверить физически. Только после проверки можно
передать актуальный layout: `-TagLayout tags/body_layout_20261013.json`.
Без layout пишутся 2D detections, метрические poses меток остаются `null`.

Все параметры доступны напрямую:

```powershell
python -m inference_logger record --output inference_logger/sessions/take01 --model models/yolo11s-pose_openvino_model --width 1280 --height 720 --fps 30 --host 0.0.0.0 --port 8765 --seconds 60
```

`--serial` выбирает конкретную RealSense; `--seconds 0` — до Ctrl+C.
Каталог должен быть новым. Существующие записи не перезаписываются.
Консоль показывает подтверждённые JSONL-счётчики, очередь, joints и tags.
Проверьте рост `quest` и `sync`; их отсутствие означает отсутствие Quest/патча.
Статус локальной записи робота проверяется отдельно в шлеме.

Первые 20–30 с: стойка и несколько перемещений корпуса без сильного вращения головы
для независимой проверки временной связи. Затем движения ног, остановка записи Quest
и завершение сборщика. При LAN dropout raw сохраняется; uplink в этот промежуток
отсутствует, полная траектория восстанавливается из локального эпизода.

## 3. Импорт, проверка, экспорт

Экспортируйте/скопируйте эпизод Quest штатным способом. Импорт принимает только
завершённый эпизод, в `camera.jsonl` которого есть **тот же session_id** в payload.
Сходство имени/времени папки не используется. Копируется всё содержимое, затем
проверяются SHA-256. Можно импортировать несколько эпизодов одной сессии.

```powershell
python -m inference_logger import-quest inference_logger/sessions/take01 <путь-к-эпизоду>
python -m inference_logger verify-bag inference_logger/sessions/take01
python -m inference_logger qc inference_logger/sessions/take01
python -m inference_logger export inference_logger/sessions/take01 --output inference_logger/sessions/take01/derived/v1 --max-gap-ms 50
```

`validate` проверяет JSONL, ID-ссылки, хеши, счётчики и монотонность capture;
`qc` дополнительно сохраняет отчёт. Warnings обязательны к чтению: структурно
корректная сессия ещё не означает пригодность всех потоков для обучения.

Экспорт создаёт:

- `clock_segments.json`: affine Quest steady → PC monotonic, drift, RTT, residuals.
  Не менее 3 probes; отбрасываются RTT > 50 мс, drift > 1000 ppm, residual > 5 мс.
  Нет mapping дальше 3 с от ближайшего probe. Это диагностические пороги, а не
  гарантия точности; асимметрия сети может смещать midpoint fit.
- `acquisition_pairs.jsonl`: камера ↔ ближайший Quest tracking submit в пределах
  `max-gap-ms`, иначе `valid=false`. Никакой интерполяции и переноса позы через провал.
- `episode_links.jsonl`: все строки input/body/frames/mimic/events, исходные значения
  без float-конвертации timestamps; связь camera payload → input_sequence → reference,
  qpos/qvel/ctrl, плюс host-время receive при однозначном clock mapping.
  Повторные physics rows сохраняются. `camera_frame_ids_observed_for_input` означает
  наблюдавшиеся приложением пакеты, а не доказанное применение в fusion.
- `report.json`: coverage, параметры, хеши источников, ограничения.

Полные Meta-бинарники, policy telemetry и остальные неизвестные форматы не теряются
при импорте, но универсального декодера для них нет. Имена и смысл суставов берите
из manifest эпизода; `ctrl` **не** обозначается raw actor action.

## Точность и оставшиеся ограничения

Точное сопоставление **по ID** доступно для camera → input → body/reference/robot.
Сопоставление по часам строится на четырёх timestamps и учитывает drift/reconnect.
Но `acquisition_pairs` — **приближённая** синхронизация: host acquisition камеры
не является временем экспозиции; submit Quest не является OpenXR predicted display
или временем измерения тела. XR/body timestamps сохранены без выдуманного пересчёта.
Для clean reference по временам измерений ещё нужны подтверждённые device→host и
XrTime→steady mappings (например, через расширение OpenXR на стороне APK).

Координаты скелета остаются camera optical, Quest — STAGE, робота — как в manifest.
Общая временная ось не выполняет пространственную калибровку. Camera→STAGE, floor и
tag→joint offsets нужно измерить отдельно; автоматически они не выдумываются.
Не записанные старым APK exact policy tensors/raw actor outputs восстановить нельзя.

## Проверки на 2026-10-14

```powershell
python -m unittest inference_logger.test_logger -v
```

Проверены синтетические offset/drift (включая timestamps > 2^53), reconnect,
WebSocket round-trip, повреждённый JSONL, fail marker, source links и сохранение
повторных robot rows. Патч прошёл `git apply --check` на доступном исходнике.
OpenVINO-модель запущена на кадре старого видео; это только проверка inference,
без настоящей глубины или метрической калибровки.

RealSense SDK обнаружил **0 устройств**. Реальная синхронная запись с Quest,
работа нативного patch в APK и playback нового bag **не проверены**.
Следующий аппаратный шаг — после установки патча снять 60 с, импортировать эпизод,
выполнить `verify-bag`, `qc`, `export` и проверить фазу движений обеих траекторий.
