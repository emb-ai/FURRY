#include "quest_menu.h"
#include <cstdio>
#include <limits>
#include <stdexcept>

using namespace questmenu;

static void Check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
static const Widget& Find(const std::vector<Widget>& layout, Action action) {
    for (const auto& widget : layout) if (widget.action == action) return widget;
    throw std::runtime_error("Expected menu action absent");
}
static Action CentreHit(const std::vector<Widget>& layout, Action action) {
    const auto& r = Find(layout, action).rect;
    return HitTest(layout, r.x + r.w / 2, r.y + r.h / 2);
}
static void CheckGeometry(const State& state) {
    const auto layout = BuildLayout(state);
    for (const auto& w : layout) {
        Check(w.rect.w > 0 && w.rect.h > 0, "Non-positive menu extent");
        Check(PanelRect.Contains(w.rect.x, w.rect.y) &&
              PanelRect.Contains(w.rect.x + w.rect.w, w.rect.y + w.rect.h), "Widget outside physical menu");
        if (w.action == Action::None) continue;
        Check(w.rect.h >= .05f, "Menu target too short for a controller ray");
        for (const auto& other : layout) {
            if (&other == &w || other.action == Action::None) continue;
            const float ix = std::min(w.rect.x + w.rect.w, other.rect.x + other.rect.w) - std::max(w.rect.x, other.rect.x);
            const float iy = std::min(w.rect.y + w.rect.h, other.rect.y + other.rect.h) - std::max(w.rect.y, other.rect.y);
            Check(ix < 1e-6f || iy < 1e-6f, "Overlapping menu targets");
        }
    }
}

int main() {
    TriggerLatch trigger;
    Check(!trigger.Update(true, true), "Held trigger selected a row at initial activation");
    Check(!trigger.Update(true, true), "Held trigger repeat before confirmed neutral");
    Check(!trigger.Update(true, false), "Neutral trigger generated a click");
    Check(trigger.Update(true, true), "Confirmed neutral did not arm the next press");
    Check(!trigger.Update(true, true), "Held trigger repeated a selection");
    Check(!trigger.Update(true, false) && trigger.Update(true, true), "Next deliberate trigger press lost");

    // Focus, menu visibility and aim validity all contribute to `active`.
    // Neither an inactive neutral nor a release hidden by that loss can rearm.
    Check(!trigger.Update(false, false), "Inactive neutral generated a click");
    Check(!trigger.Update(true, true), "Inactive neutral rearmed a trigger");
    Check(!trigger.Update(true, false), "Neutral trigger generated a click after aim recovery");
    Check(!trigger.Update(false, true), "Lost aim generated a click");
    Check(!trigger.Update(true, true), "Aim recovery with held trigger selected a row");
    Check(!trigger.Update(true, false), "Neutral trigger generated a click after focus recovery");
    Check(!trigger.Update(false, false), "Focus loss generated a click");
    Check(!trigger.Update(true, true), "Focus recovery retained a previously armed press");
    Check(!trigger.Update(true, false) && trigger.Update(true, true), "Fresh press after recovery cannot click");

    State state;
    Check(state.open && state.paused && !state.debugEnabled && !state.recording, "Initial menu state");
    Check(std::string(SceneName(Scene::Empty)) == "stand" &&
          std::string(SceneName(Scene::Cup)) == "cup" &&
          std::string(SceneName(Scene::PushT)) == "push_t", "Scene runtime names");
    auto layout = BuildLayout(state);
    Check(CentreHit(layout, Action::Start) == Action::None, "Uncalibrated session can start");
    state.calibrated = true;
    layout = BuildLayout(state);
    Check(CentreHit(layout, Action::Start) == Action::None, "Session without tracking can start");
    state.trackingValid = true;
    layout = BuildLayout(state);
    Check(CentreHit(layout, Action::Start) == Action::Start, "Ready session cannot start");
    state.faulted = true;
    layout = BuildLayout(state);
    Check(CentreHit(layout, Action::Start) == Action::None, "Faulted simulation can continue");
    state.faulted = false;

    // An active, paused recording permits calibration and saving, but cannot
    // silently change its mode, scene, capture format, or guided protocol.
    state.mode = Mode::Trajectories; state.recording = true;
    layout = BuildLayout(state);
    for (Action a : {Action::ModeSimulation, Action::ModeTrajectories, Action::CaptureRobot,
                     Action::CaptureHuman, Action::SceneEmpty, Action::SceneCup, Action::ScenePushT})
        Check(CentreHit(layout, a) == Action::None, "Recording configuration is editable");
    Check(CentreHit(layout, Action::Calibrate) == Action::Calibrate, "Recording calibration is locked");
    Check(CentreHit(layout, Action::Save) == Action::Save, "Paused recording cannot be saved");
    Check(CentreHit(layout, Action::Start) == Action::Start, "Paused recording cannot resume");
    state.capture = Capture::Human;
    layout = BuildLayout(state);
    Check(CentreHit(layout, Action::PlanTrain) == Action::None &&
          CentreHit(layout, Action::PlanTest) == Action::None, "Recording protocol is editable");
    Check(HumanCapture(state), "Human capture predicate");
    for (const auto& w : layout)
        Check(w.action != Action::SceneEmpty && w.action != Action::SceneCup && w.action != Action::ScenePushT,
              "Human capture advertises a simulated scene");
    state.page = Page::View;
    layout = BuildLayout(state);
    Check(CentreHit(layout, Action::Passthrough) == Action::None &&
          CentreHit(layout, Action::ViewFirstPerson) == Action::None &&
          CentreHit(layout, Action::Place) == Action::None, "Human capture scene controls are active");
    Check(Find(layout, Action::Passthrough).selected, "Human capture must show the real room");

    state.page = Page::Debug;
    layout = BuildLayout(state);
    Check(CentreHit(layout, Action::DebugEnabled) == Action::DebugEnabled, "Master debug control disabled");
    Check(CentreHit(layout, Action::DebugStats) == Action::None, "Debug sub-option editable while debug is off");
    state.debugEnabled = true;
    layout = BuildLayout(state);
    for (Action a : {Action::DebugStats, Action::DebugMeta, Action::DebugCamera, Action::DebugTargets, Action::DebugContacts})
        Check(CentreHit(layout, a) == a, "Enabled debug option cannot be selected");

    state.page=Page::Policy;state.mode=Mode::Simulation;state.recording=false;
    for(int i=0;i<11;i++)state.policies.push_back({"p"+std::to_string(i),"policy.onnx","","Candidate "+std::to_string(i),"Experimental"});
    state.policyIndex=5;state.policyPage=1;
    layout=BuildLayout(state);
    Check(Find(layout,Action::CatchUp).selected&&CentreHit(layout,Action::CatchUp)==Action::CatchUp,"Catch-up toggle unavailable/default wrong");
    Check(Find(layout,Action::PolicySelect1).selected,"Active policy highlight missing");
    Check(Find(layout,Action::PolicySelect0).text=="Candidate 4","Wrong policy page");
    Check(CentreHit(layout,Action::PolicySelect0)==Action::PolicySelect0,"Candidate cannot be selected");
    Check(CentreHit(layout,Action::PolicyNext)==Action::PolicyNext,"Next policy page missing");
    state.recording=true;layout=BuildLayout(state);
    Check(CentreHit(layout,Action::PolicySelect0)==Action::None,"Policy can change during recording");
    Check(CentreHit(layout,Action::CatchUp)==Action::None,"Catch-up can change during recording");
    state.recording=false;state.mode=Mode::Trajectories;state.capture=Capture::Human;
    Check(CentreHit(BuildLayout(state),Action::PolicySelect0)==Action::None,"Human-only capture edits policy");
    Check(CentreHit(BuildLayout(state),Action::CatchUp)==Action::None,"Human-only capture edits catch-up");
    state.policyPage=2;layout=BuildLayout(state);
    Check(CentreHit(layout,Action::PolicyNext)==Action::None,"Policy paging exceeds catalog");
    CheckGeometry(state);
    state.policyPage=0;layout=BuildLayout(state);
    Check(CentreHit(layout,Action::PolicyPrevious)==Action::None,"Policy paging underflows catalog");

    for (auto page : {Page::Session, Page::View, Page::Debug, Page::Policy})
        for (auto mode : {Mode::Simulation, Mode::Trajectories})
            for (auto capture : {Capture::Robot, Capture::Human}) {
                state.page = page; state.mode = mode; state.capture = capture;
                CheckGeometry(state);
            }
    const float nan = std::numeric_limits<float>::quiet_NaN();
    Check(HitTest(layout, nan, 0) == Action::None && HitTest(layout, 0, nan) == Action::None,
          "Non-finite pointer selected an action");
    state.open = false;
    Check(BuildLayout(state).empty(), "Closed menu has hit targets");

    const float identity[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float x = 91, y = 92, distance = 93;
    const float origin[3] = {.10f, -.12f, 1.1f}, forward[3] = {0,0,-1};
    Check(RayHit(identity, origin, forward, x, y, distance), "Front ray missed panel");
    Check(std::abs(x - .10f) < 1e-6 && std::abs(y + .12f) < 1e-6 && std::abs(distance - 1.1f) < 1e-6,
          "Identity menu ray coordinates");
    const float scaledForward[3] = {0,0,-2};
    Check(RayHit(identity, origin, scaledForward, x, y, distance) && std::abs(distance - 1.1f) < 1e-6,
          "Ray distance depends on direction normalization");
    const float backOrigin[3] = {0,0,-1}, backward[3] = {0,0,1}, parallel[3] = {1,0,0};
    Check(!RayHit(identity, backOrigin, backward, x, y, distance), "Panel accepts ray from the back");
    Check(!RayHit(identity, origin, backward, x, y, distance), "Panel accepts ray pointing away");
    Check(!RayHit(identity, origin, parallel, x, y, distance), "Parallel ray intersects menu");
    const float tooNear[3] = {0,0,.001f}, outside[3] = {.5f,0,1}, tooFar[3] = {0,0,9}, invalid[3] = {nan,0,1};
    Check(!RayHit(identity, tooNear, forward, x, y, distance), "Near plane ray accepted");
    Check(!RayHit(identity, outside, forward, x, y, distance), "Ray outside panel accepted");
    Check(!RayHit(identity, tooFar, forward, x, y, distance), "Unbounded ray reach");
    Check(!RayHit(identity, invalid, forward, x, y, distance), "Non-finite ray accepted");

    // Yaw and translation match the runtime's head-relative upright anchor.
    const float a = .63f, c = std::cos(a), s = std::sin(a);
    const float menu[16] = {c,0,-s,0, 0,1,0,0, s,0,c,0, 1.4f,1.65f,-2.1f,1};
    const float worldOrigin[3] = {1.4f + c*.1f + s*1.1f, 1.65f-.12f, -2.1f-s*.1f+c*1.1f};
    const float worldDir[3] = {-s,0,-c};
    Check(RayHit(menu, worldOrigin, worldDir, x, y, distance), "Rotated translated menu missed");
    Check(std::abs(x-.1f)<1e-6 && std::abs(y+.12f)<1e-6 && std::abs(distance-1.1f)<1e-6,
          "World ray did not map to panel metres");
    puts("Quest menu trigger recovery, layout, recording locks, debug controls and rigid world-ray checks passed");
}
