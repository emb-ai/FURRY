# Установка FURRY на Quest по Wi-Fi

[Home](Home.md) | [Android prototype](../prototypes/quest_g1/android/README.md)

Несколько участников могут по очереди обновлять один Quest со своих компьютеров.
Каждый компьютер отдельно авторизуется в шлеме. Для установки готового APK не нужны
Android Studio, NDK, MuJoCo или ключ подписи: достаточно Python и ADB.

## Настройка компьютера один раз

1. Установите Python 3.8+ с [python.org](https://www.python.org/downloads/).
   На Windows включите добавление Python в PATH; примеры ниже используют `py -3`.
2. Скачайте [Android SDK Platform Tools](https://developer.android.com/tools/releases/platform-tools),
   распакуйте архив и добавьте каталог `platform-tools` в PATH.
   Откройте новый терминал и проверьте `adb version`.
   Альтернатива PATH — переменная `G1_ADB` с полным путём к `adb` / `adb.exe`.
3. Получите актуальный checkout [FURRY](https://github.com/emb-ai/FURRY).
   Скрипт расположен в `prototypes/quest_g1/scripts/quest_wifi.py`.
4. На общем Quest должен быть включён Developer Mode.
   Компьютер и шлем должны иметь доступ друг к другу в локальной сети роутера:
   компьютер может быть подключён по Ethernet. Guest/AP client isolation мешает ADB.
5. Подключите Quest кабелем USB с передачей данных. В шлеме подтвердите
   **Allow debugging / Разрешить отладку** и **Always allow from this computer**.
   Проверьте `adb devices -l`: состояние должно быть `device`, не `unauthorized`.

У каждого участника остаётся собственный ADB-ключ. Копировать чужой `adbkey` не нужно.
Разрешение одного компьютера не отменяет разрешения остальных; отзыв всех
авторизаций в настройках Quest потребует подтвердить компьютеры заново.

## Подключение одной командой

Запускайте из корня checkout FURRY.

macOS / Linux:

```sh
python3 prototypes/quest_g1/scripts/quest_wifi.py
```

Windows PowerShell:

```powershell
py -3 prototypes/quest_g1/scripts/quest_wifi.py
```

На macOS также можно запустить `prototypes/quest_g1/quest_wifi.command` из Finder
или терминала. Скрипт использует локальный `.tools/platform-tools/adb`, если он есть,
иначе ищет ADB в PATH. Дополнительные Python-пакеты не требуются.

Скрипт проверяет serial общего Quest `2G0YC5ZG7T05SR`, получает текущий Wi-Fi IP
через USB и сначала пробует уже работающий сетевой ADB. Если он недоступен,
выполняет `adb tcpip 5555` через USB, подключается по сети и повторно проверяет
serial. После сообщения об успешном подключении кабель можно отключить.
Последний подтверждённый адрес сохраняется локально в
`prototypes/quest_g1/outputs/quest-wifi/connection.json` и не публикуется в Git.

Если USB уже отключён, команда пробует имеющиеся сетевые подключения,
сохранённый IP и обнаружение legacy ADB через mDNS. При смене адреса можно
подключить USB повторно или явно указать новый IP:

```sh
python3 prototypes/quest_g1/scripts/quest_wifi.py --ip 192.168.10.216
```

Адрес в примере не закреплён за шлемом. Удобно сделать DHCP reservation на роутере.
Для другого Quest передайте `--serial SERIAL_ИЗ_ADB_DEVICES`.

## После полного перезапуска Quest

Для выбранного способа восстановления подключите USB и повторите ту же команду.
Она сама прочитает IP и включит сетевой ADB. Затем USB можно убрать.
Не рассчитывайте, что `adb tcpip 5555` переживёт полный перезапуск шлема.
Сохранённая авторизация компьютера и включённый сетевой ADB — разные состояния.

Если шлем запросит разрешение ещё раз, подтвердите диалог в Quest и повторите
команду. Без USB команда сможет подключиться только к уже работающему сетевому ADB.
Автоматическое включение после загрузки без USB этим скриптом не настраивается.

## Обновление APK той же командой

Перед установкой закончите запись и договоритесь, кто сейчас обновляет шлем.
Укажите конкретный готовый APK; скрипт не выбирает сборку автоматически.

macOS / Linux:

```sh
python3 prototypes/quest_g1/scripts/quest_wifi.py "/path/to/furry.apk"
```

Windows PowerShell:

```powershell
py -3 prototypes/quest_g1/scripts/quest_wifi.py "C:\Downloads\furry.apk"
```

Команда подключает Quest и выполняет `adb -s IP:5555 install -r APK`.
Опция `-r` сохраняет данные приложения. После `Success` откройте FURRY в шлеме.
Приложение автоматически не запускается; установка не доказывает живую
работоспособность новой сборки.

По окончании работы можно отключить подключение своего компьютера:

```sh
adb disconnect 192.168.10.216:5555
```

`disconnect` не запрещает другим авторизованным участникам подключаться.
Команда `adb -s IP:5555 usb` отключает сетевой режим на самом шлеме для всех;
используйте её, когда команда закончила работу.

## Общая подпись APK обязательна

Все обновления `org.g1lab.quest` должны иметь ту же подпись, что установленный APK.
Наш [build_apk.sh](../prototypes/quest_g1/android/build_apk.sh) создаёт
`.tools/g1-debug.keystore`, если файл отсутствует. Независимые сборки на разных
компьютерах поэтому получают разные ключи даже при одинаковом имени и пароле.

Рекомендуемый порядок: один сборщик или CI подписывает сборку совместимым ключом,
публикует APK и SHA256 в GitHub Releases, участники скачивают и устанавливают
этот готовый APK. Публикация этой инструкции не создаёт новый APK или GitHub Release.
Если несколько участников собирают самостоятельно, нужен один согласованный
ключ подписи, переданный отдельно. Ключи и пароли нельзя коммитить в Git.

При `INSTALL_FAILED_UPDATE_INCOMPATIBLE` получите APK с правильной подписью.
Не удаляйте приложение для обхода ошибки: обычное удаление стирает его данные.
Смена ключа требует отдельной согласованной миграции с проверенным backup.

## Если не подключается

| Симптом | Действие |
| --- | --- |
| `adb` не найден | Проверьте PATH или задайте `G1_ADB`. |
| `unauthorized` / failed to authenticate | Разбудите Quest, подтвердите диалог отладки и повторите команду. |
| После перезапуска нет Wi-Fi ADB | Подключите USB и повторите обычную команду. |
| Нет Wi-Fi IPv4 | Подключите Quest к Wi-Fi роутера. |
| Timeout / connection refused | Проверьте текущий IP, USB, firewall и изоляцию клиентов роутера. |
| `more than one device` в ручном ADB | Укажите `-s IP:5555`; скрипт делает это автоматически. |
| Найден другой serial | Проверьте IP; скрипт останавливается до установки. |

Legacy `tcpip 5555` не шифрует трафик. Используйте доверенную локальную сеть и
не пробрасывайте порт ADB в интернет. Android Wireless debugging с `adb pair`
использует TLS и другой порт; это отдельный режим, этим скриптом не настраиваемый.

## Источники и проверка

- [Meta: Use ADB with Meta Quest](https://developers.meta.com/vr/documentation/native/android/ts-adb/).
- [Android: ADB, авторизация и install -r](https://developer.android.com/tools/adb).
- [Android: одинаковый сертификат для обновлений](https://developer.android.com/studio/publish/app-signing).
- [AOSP: legacy TCP и TLS Wireless debugging](https://android.googlesource.com/platform/packages/modules/adb/+/refs/heads/main/docs/dev/adb_wifi.md).

Проверено 2026-10-10 на macOS и общем Quest 3: включение `tcpip 5555` через USB,
подключение по Wi-Fi, проверка serial и повторное подключение без использования
USB. Полный reboot, Windows/Linux и установка нового APK этой командой
в этой проверке не выполнялись.
