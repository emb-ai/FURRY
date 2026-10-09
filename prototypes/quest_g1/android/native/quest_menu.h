#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include "policy_catalog.h"

// The menu is a physical, upright plane. Coordinates are metres, +Y is up,
// +Z faces the user. The runtime owns state transitions and anchoring.
namespace questmenu {

enum class Page { Session, View, Debug, Policy };
enum class Mode { Simulation, Trajectories };
enum class Scene { Empty, Cup, PushT };
enum class Capture { Robot, Human };
enum class Plan { Train, Test };
enum class Action {
    None, Close, Start, Save, Reset, QuickReset, ToggleRecording, Calibrate, Place,
    ModeSimulation, ModeTrajectories, CaptureRobot, CaptureHuman, PlanTrain, PlanTest,
    SceneEmpty, SceneCup, ScenePushT, ViewObserver, ViewFirstPerson, Passthrough,
    DebugEnabled, DebugStats, DebugMeta, DebugCamera, DebugTargets, DebugContacts,
    PageSession, PageView, PageDebug, PagePolicy, PolicyPrevious, PolicyNext,
    PolicySelect0, PolicySelect1, PolicySelect2, PolicySelect3
};
enum class Kind { Text, MutedText, Divider, Tab, Button, Primary, Radio, Checkbox };

struct State {
    Page page = Page::Session;
    Mode mode = Mode::Simulation;
    Scene scene = Scene::Cup;
    Capture capture = Capture::Robot;
    Plan plan = Plan::Train;
    bool open = true, paused = true, recording = false;
    bool calibrated = false, trackingValid = false, faulted = false;
    bool firstPerson = false, passthrough = true;
    bool debugEnabled = false, debugStats = true, debugMeta = false;
    bool debugCamera = false, debugTargets = false, debugContacts = false;
    std::string notice, recordingInfo, cameraInfo;
    int exportStatus = 0;
    std::vector<questpolicy::Entry> policies;
    int policyIndex = 0, policyPage = 0;
};

inline constexpr int PoliciesPerPage = 4;
inline int PolicyRow(Action action) {
    const int row=int(action)-int(Action::PolicySelect0);
    return row>=0&&row<PoliciesPerPage?row:-1;
}

struct Rect {
    float x = 0, y = 0, w = 0, h = 0; // lower left and extent
    bool Contains(float px, float py) const {
        return std::isfinite(px) && std::isfinite(py) &&
               px >= x && px <= x + w && py >= y && py <= y + h;
    }
};
struct Widget {
    Rect rect;
    std::string text;
    Action action = Action::None;
    bool enabled = true, selected = false;
    Kind kind = Kind::Text;
};
struct Pointer {
    float x = 0, y = 0;
    bool valid = false, hover = false;
};

// A ray click requires a confirmed neutral trigger while the menu, focus and
// aim are all valid. Losing any of them cancels that permission, so reopening
// the menu or reacquiring a controller with a held trigger cannot select a row.
class TriggerLatch {
    bool armed = false;
public:
    bool Update(bool active, bool pressed) {
        if (!active) { armed = false; return false; }
        if (!pressed) { armed = true; return false; }
        if (!armed) return false;
        armed = false;
        return true;
    }
};

inline constexpr float PanelWidth = .76f;
inline constexpr float PanelHeight = .86f;
inline constexpr Rect PanelRect { -PanelWidth / 2, -PanelHeight / 2, PanelWidth, PanelHeight };

inline const char* SceneName(Scene scene) {
    switch (scene) {
    case Scene::Empty: return "stand";
    case Scene::Cup: return "cup";
    case Scene::PushT: return "push_t";
    }
    return "stand";
}
inline const char* SceneTitle(Scene scene) {
    switch (scene) {
    case Scene::Empty: return "Пустая";
    case Scene::Cup: return "Стол и чашка";
    case Scene::PushT: return "Push-T";
    }
    return "Пустая";
}
inline bool HumanCapture(const State& s) {
    return s.mode == Mode::Trajectories && s.capture == Capture::Human;
}
inline std::string StatusText(const State& s) {
    if (s.faulted) return "Ошибка симуляции";
    if (s.recording) return s.paused ? "Запись на паузе" : "Идёт запись";
    if (!s.trackingValid) return "Нет трекинга";
    if (!s.calibrated) return "Нужна калибровка";
    return s.paused ? "Пауза" : "Симуляция";
}

inline std::vector<Widget> BuildLayout(const State& s) {
    std::vector<Widget> out;
    if (!s.open) return out;
    auto add = [&](Rect rect, const std::string& text, Action action = Action::None,
                   bool enabled = true, bool selected = false, Kind kind = Kind::Text) {
        out.push_back({rect, text, action, enabled, selected, kind});
    };
    auto line = [&](float y) { add({-.352f, y, .704f, .001f}, "", Action::None, true, false, Kind::Divider); };
    auto label = [&](float y, const std::string& text) {
        add({-.352f, y, .142f, .054f}, text, Action::None, true, false, Kind::MutedText);
    };
    auto text = [&](float y, const std::string& value) { add({-.192f, y, .544f, .054f}, value); };
    auto segments = [&](float y, const char* left, const char* right,
                        Action a, Action b, bool selectedLeft, bool enabled = true) {
        add({-.192f, y, .233f, .054f}, left, a, enabled, selectedLeft, Kind::Button);
        add({.041f, y, .311f, .054f}, right, b, enabled, !selectedLeft, Kind::Button);
    };
    auto check = [&](float y, const char* value, Action a, bool selected, bool enabled = true) {
        add({-.352f, y, .704f, .058f}, value, a, enabled, selected, Kind::Checkbox);
        line(y);
    };

    add({-.352f, .369f, .145f, .04f}, "FURRY");
    add({-.183f, .369f, .366f, .04f}, StatusText(s), Action::None, true, false, Kind::MutedText);
    add({.213f, .362f, .139f, .052f}, "Закрыть", Action::Close, true, false, Kind::Button);
    add({-.352f, .291f, .144f, .055f}, "Сессия", Action::PageSession, true, s.page == Page::Session, Kind::Tab);
    add({-.183f, .291f, .102f, .055f}, "Вид", Action::PageView, true, s.page == Page::View, Kind::Tab);
    add({-.056f, .291f, .166f, .055f}, "Отладка", Action::PageDebug, true, s.page == Page::Debug, Kind::Tab);
    add({.135f, .291f, .217f, .055f}, "Политика", Action::PagePolicy, true, s.page == Page::Policy, Kind::Tab);
    line(.289f);

    const bool human = HumanCapture(s);
    if (s.page == Page::Session) {
        float y = .214f;
        label(y, "Режим");
        segments(y, "Симуляция", "Сбор траекторий", Action::ModeSimulation,
                 Action::ModeTrajectories, s.mode == Mode::Simulation, !s.recording);
        y -= .069f;
        if (s.mode == Mode::Trajectories) {
            label(y, "Запись");
            segments(y, "Человек + робот", "Движения человека", Action::CaptureRobot,
                     Action::CaptureHuman, s.capture == Capture::Robot, !s.recording);
            y -= .069f;
        }
        if (human) {
            label(y, "План");
            segments(y, "Обучение: 15 мин", "Тест: 5 минут", Action::PlanTrain,
                     Action::PlanTest, s.plan == Plan::Train, !s.recording);
            y -= .069f;
            label(y, "Сцена");
            text(y, "Реальная комната");
            y -= .069f;
        } else {
            label(y, "Сцена");
            const Scene scenes[] = {Scene::Empty, Scene::Cup, Scene::PushT};
            const Action actions[] = {Action::SceneEmpty, Action::SceneCup, Action::ScenePushT};
            for (int i = 0; i != 3; ++i) {
                add({-.192f, y, .544f, .055f}, SceneTitle(scenes[i]), actions[i],
                    !s.recording, s.scene == scenes[i], Kind::Radio);
                y -= .056f;
            }
            y -= .013f;
        }
        label(y, "Калибровка");
        add({-.192f, y, .24f, .054f}, s.calibrated ? "Готова" : "Нужна калибровка");
        add({.061f, y, .291f, .054f}, "Калибровать", Action::Calibrate, s.paused, false, Kind::Button);
        y -= .064f;
        label(y, "Трекинг");
        text(y, s.trackingValid ? "Meta: трекинг есть" : "Meta: нет свежего трекинга");
    } else if (s.page == Page::View) {
        label(.214f, "Камера");
        segments(.214f, "Наблюдатель", "От первого лица", Action::ViewObserver,
                 Action::ViewFirstPerson, !s.firstPerson, !human);
        check(.130f, "Реальная комната", Action::Passthrough, human || s.passthrough, !human);
        label(.047f, "Сцена");
        add({-.192f, .047f, .544f, .054f}, "Поставить перед собой", Action::Place,
            !human && !s.recording, false, Kind::Button);
        line(.032f);
        label(-.039f, "Meta");
        text(-.039f, s.trackingValid ? "Трекинг есть" : "Нет свежего трекинга");
        line(-.051f);
        label(-.121f, "RGB-D");
        text(-.121f, s.cameraInfo.empty() ? "Не подключена" : s.cameraInfo);
    } else if(s.page == Page::Policy) {
        const int count=int(s.policies.size());
        const int pages=std::max(1,(count+PoliciesPerPage-1)/PoliciesPerPage);
        const int page=std::clamp(s.policyPage,0,pages-1);
        add({-.352f,.214f,.704f,.04f}, "Выбор сбросит робота и калибровку", Action::None,true,false,Kind::MutedText);
        for(int row=0;row<PoliciesPerPage;row++){
            int index=page*PoliciesPerPage+row;if(index>=count)break;
            float y=.124f-row*.078f;const auto& policy=s.policies[index];
            add({-.352f,y,.704f,.052f},policy.title,Action(int(Action::PolicySelect0)+row),
                !s.recording&&!human,s.policyIndex==index,Kind::Radio);
            add({-.312f,y-.023f,.664f,.022f},policy.note,Action::None,true,false,Kind::MutedText);
        }
        if(pages>1){
            add({-.352f,-.198f,.18f,.052f},"Назад",Action::PolicyPrevious,page>0,false,Kind::Button);
            add({-.152f,-.198f,.304f,.052f},std::to_string(page+1)+" / "+std::to_string(pages));
            add({.172f,-.198f,.18f,.052f},"Далее",Action::PolicyNext,page+1<pages,false,Kind::Button);
        }
    } else {
        check(.213f, "Показывать отладку", Action::DebugEnabled, s.debugEnabled);
        check(.137f, "FPS и время вычислений", Action::DebugStats, s.debugStats, s.debugEnabled);
        check(.071f, "Скелет Meta", Action::DebugMeta, s.debugMeta, s.debugEnabled);
        check(.005f, "Скелет внешней камеры", Action::DebugCamera, s.debugCamera, s.debugEnabled);
        check(-.061f, "Цели ретаргетинга", Action::DebugTargets, s.debugTargets, s.debugEnabled);
        check(-.127f, "Коллизии и контакты", Action::DebugContacts, s.debugContacts, s.debugEnabled);
    }

    // Two compact status lines leave the action row fixed across all pages.
    add({-.352f, -.241f, .704f, .038f}, s.notice, Action::None, true, false, Kind::MutedText);
    std::string info = s.recordingInfo;
    if (s.exportStatus == 1) info = "Архив копируется в Download/G1Quest";
    else if (s.exportStatus == 2) info = "Архив сохранён в Download/G1Quest";
    else if (s.exportStatus == 3) info = "Ошибка экспорта. Исходники сохранены в очках";
    add({-.352f, -.278f, .704f, .036f}, info, Action::None, true, false, Kind::MutedText);
    line(-.292f);
    add({-.352f, -.362f, .26f, .054f}, s.recording ? "Сохранить запись" : "Сбросить",
        s.recording ? Action::Save : Action::Reset, true, false, Kind::Button);
    const char* start = s.recording ? "Продолжить запись" :
        s.mode == Mode::Trajectories ? "Начать запись" : "Запустить";
    add({.032f, -.362f, .32f, .054f}, start, Action::Start,
        s.calibrated && s.trackingValid && !s.faulted, false, Kind::Primary);
    add({-.352f, -.410f, .704f, .032f},
        "A: калибровка   B: пауза   X: сброс   Y: запись   L-стик: вид",
        Action::None, true, false, Kind::MutedText);
    return out;
}

inline Action HitTest(const std::vector<Widget>& layout, float x, float y) {
    // Last drawn widget wins, if a future page adds an overlapping popover.
    for (auto i = layout.rbegin(); i != layout.rend(); ++i)
        if (i->enabled && i->action != Action::None && i->rect.Contains(x, y)) return i->action;
    return Action::None;
}

inline bool RayHit(const float* menuWorld, const float origin[3], const float dir[3],
                   float& x, float& y, float& distance) {
    // A rigid column-major transform permits transpose inversion without an
    // OpenXR or MuJoCo dependency. The panel does not accept rays from its back.
    for (int i = 0; i != 16; ++i) if (!std::isfinite(menuWorld[i])) return false;
    for (int i = 0; i != 3; ++i)
        if (!std::isfinite(origin[i]) || !std::isfinite(dir[i])) return false;
    const float relative[3] = {origin[0] - menuWorld[12], origin[1] - menuWorld[13], origin[2] - menuWorld[14]};
    float o[3] {}, d[3] {};
    for (int c = 0; c != 3; ++c) for (int r = 0; r != 3; ++r) {
        o[c] += menuWorld[4 * c + r] * relative[r];
        d[c] += menuWorld[4 * c + r] * dir[r];
    }
    if (o[2] <= .005f || d[2] >= -1e-5f) return false;
    const float t = -o[2] / d[2];
    const float length = std::sqrt(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);
    const float metres = t * length;
    if (!std::isfinite(t) || !std::isfinite(metres) || metres < .02f || metres > 8.f) return false;
    const float hitX = o[0] + t * d[0], hitY = o[1] + t * d[1];
    if (!PanelRect.Contains(hitX, hitY)) return false;
    x = hitX; y = hitY; distance = metres;
    return true;
}

} // namespace questmenu
