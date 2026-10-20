#include "recording.h"
#include "render_geom.h"
#include "tennis_task.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
static void Check(bool ok, const char *reason) {
  if (!ok)
    throw std::runtime_error(reason);
}
static void Near(double a, double b, double tol, const char *reason) {
  Check(std::abs(a - b) < tol, reason);
}
using T = TennisTask;
static void Step(const mjModel *m, mjData *d, T &t) {
  t.Begin(d);
  mj_step(m, d);
  t.Finish(m, d);
}
static void Ball(const mjModel *m, mjData *d, T &t, T::Vec pos, T::Vec vel,
                 bool hit) {
  t.position = pos;
  t.velocity = vel;
  t.Write(m, d);
  d->userdata[T::Active] = 1;
  d->userdata[T::Hit] = hit;
}
int main(int argc, char **argv) {
  try {
    Check(argc == 3, "Usage: check_tennis scene.xml temporary-output-root");
    char error[2048]{};
    mjModel *m = mj_loadXML(argv[1], nullptr, error, sizeof(error));
    if (!m)
      throw std::runtime_error(error);
    mjData *d = mj_makeData(m);
    T task;
    task.Initialize(m);
    task.Reset(m, d);
    Near(m->body_mass[task.ball], .0027, 1e-12, "Ball mass");
    Near(m->geom_size[3 * mj_name2id(m, mjOBJ_GEOM, "tennis_ball_geom")], .02,
         1e-12, "Ball radius");
    Check(m->geom_contype[mj_name2id(m, mjOBJ_GEOM, "tennis_ball_geom")] == 0,
          "Ball double contact");
    T::Vec p{}, v{5, 0, 0};
    for (int i = 0; i < 10000; i++) {
      auto mid = T::Add(v, T::Acceleration(v), .00005);
      mid[2] = 0;
      p = T::Add(p, mid, .0001);
      auto a = T::Acceleration(mid);
      a[2] = 0;
      v = T::Add(v, a, .0001);
    }
    double k = .5 * 1.204 * .5 * 3.141592653589793 * .02 * .02 / .0027;
    Near(v[0], 5 / (1 + 5 * k), 1e-7, "Quadratic drag analytical velocity");
    Near(p[0], std::log(1 + 5 * k) / k, 1e-7,
         "Quadratic drag analytical distance");
    p = {0, 0, 1};
    v = {1, 2, 3};
    for (int i = 0; i < 10000; i++)
      T::Fly(p, v, .0001, 0);
    Near(p[2], 1 + 3 - 9.81 / 2, 1e-9, "Ballistic gravity");
    Near(v[2], 3 - 9.81, 1e-9, "Ballistic velocity");
    auto reflected = T::Reflect({-5, 2, 1}, {2, 0, 0}, {1, 0, 0});
    Near(reflected[0], 9, 1e-12, "Moving surface e=1");
    Near(reflected[1], 2, 1e-12, "No tangential impulse");
    auto edge = T::Cylinder({.09, 0, .01}, .085, .006);
    Near(edge.distance, std::hypot(.005, .004), 1e-12, "Rounded paddle rim");
    d->time = 100;
    task.Begin(d);
    task.Finish(m, d);
    Check(!d->userdata[T::Serves], "Serve before touch");
    task.Touch(d);
    Near(d->userdata[T::NextServe], 107, 1e-12, "First touch countdown");
    d->time = 103;
    task.Touch(d);
    Near(d->userdata[T::NextServe], 107, 1e-12,
         "Repeated touch postpones feeder");
    task.Begin(d);
    task.Finish(m, d);
    Check(!d->userdata[T::Serves], "Serve during countdown");
    d->time = 107;
    task.Begin(d);
    task.Finish(m, d);
    Check(d->userdata[T::Serves] == 1 && d->userdata[T::Active], "First serve");
    Near(T::Norm(task.velocity), 5.5, 1e-10, "Muzzle speed");
    Near(d->userdata[T::NextServe], 111, 1e-12, "Serve interval");
    bool bounce = false;
    double crossHeight = 0;
    for (int i = 0; i < 1200 && d->userdata[T::Active]; i++) {
      double x = d->qpos[task.qa];
      Step(m, d, task);
      if (x > T::Mid && d->qpos[task.qa] <= T::Mid)
        crossHeight = d->qpos[task.qa + 2];
      if (d->userdata[T::OwnBounces]) {
        bounce = true;
        break;
      }
    }
    Check(bounce, "Serve must bounce on player half");
    Check(crossHeight > T::Top + .1525 + T::Radius, "Serve hits net");
    Check(d->qpos[task.qa] < T::Mid && d->qpos[task.qa] > T::Near,
          "Wrong bounce half");
    std::cout << "serve_net_height=" << crossHeight
              << " bounce_x=" << d->qpos[task.qa] << "\n";
    task.Reset(m, d);
    Check(!d->userdata[T::Armed] && !d->userdata[T::Serves], "Reset state");
    Ball(m, d, task, {2.2, 0, .7801}, {0, 0, -2}, false);
    Step(m, d, task);
    Check(d->userdata[T::Returns] == 0 && d->userdata[T::Misses] == 1,
          "Unhit ball counted");
    Ball(m, d, task, {2.2, 0, .7801}, {0, 0, -2}, true);
    Step(m, d, task);
    Check(d->userdata[T::Returns] == 1 && !d->userdata[T::Active],
          "Opponent landing not scored");
    Step(m, d, task);
    Check(d->userdata[T::Returns] == 1, "Duplicate score");
    Ball(m, d, task, {.95, 0, .7801}, {0, 0, -2}, true);
    Step(m, d, task);
    Check(d->userdata[T::Misses] == 2, "Own-half return counted");
    Ball(m, d, task, {T::Mid + .028, 0, .85}, {-20, 0, 0}, true);
    Step(m, d, task);
    Check(d->userdata[T::Misses] == 3, "Net miss");
    Ball(m, d, task, {2.2, 1, .021}, {0, 0, -4}, true);
    Step(m, d, task);
    Check(d->userdata[T::Misses] == 4, "Floor miss");
    // A ball colliding with G1's physical collider is a miss.
    int robotGeom = -1;
    for (int g = 0; g < m->ngeom; g++)
      if (m->geom_contype[g] == 2) {
        robotGeom = g;
        break;
      }
    Check(robotGeom >= 0, "No robot collider");
    T::Vec robotPos{};
    std::copy_n(d->geom_xpos + 3 * robotGeom, 3, robotPos.begin());
    double missed = d->userdata[T::Misses];
    Ball(m, d, task, robotPos, {0, 0, 0}, true);
    Step(m, d, task);
    Check(d->userdata[T::Misses] == missed + 1,
          "Robot-body hit did not end shot");
    // Isolated freely moving racket: check momentum and finite-mass energy.
    mj_resetData(m, d);
    int rj = mj_name2id(m, mjOBJ_JOINT, "tennis_racket_free");
    int rq = m->jnt_qposadr[rj], rv = m->jnt_dofadr[rj];
    d->qpos[rq] = 0;
    d->qpos[rq + 1] = -1;
    d->qpos[rq + 2] = 2;
    d->qpos[rq + 3] = 1;
    d->qvel[rv + 2] = 1;
    mj_forward(m, d);
    task.velocity = {0, 0, -5};
    double before = .5 * T::Mass * 25 + .5 * m->body_mass[task.racket];
    T::Vec point{};
    std::copy_n(d->xipos + 3 * task.racket, 3, point.begin());
    task.RacketImpulse(m, d, {0, 0, 1}, point);
    double after =
        .5 * T::Mass * T::Dot(task.velocity, task.velocity) +
        .5 * m->body_mass[task.racket] * d->qvel[rv + 2] * d->qvel[rv + 2];
    Near(before, after, 1e-10, "Elastic finite mass energy");
    Near(-5 * T::Mass + m->body_mass[task.racket],
         T::Mass * task.velocity[2] +
             m->body_mass[task.racket] * d->qvel[rv + 2],
         1e-10, "Impact momentum");
    // 60 m/s face sweep must not tunnel through a thin racket.
    std::fill_n(d->qvel, m->nv, 0);
    mj_forward(m, d);
    task.Reset(m, d);
    Ball(m, d, task, {0, -1, 2.05}, {0, 0, -60}, false);
    Step(m, d, task);
    Check(d->userdata[T::Hit] && d->qvel[task.va + 2] > 0,
          "High-speed racket tunnel");
    // Real MuJoCo palm/finger contact arms the feeder, table contact does not.
    mj_resetData(m, d);
    mj_forward(m, d);
    task.Reset(m, d);
    task.Begin(d);
    task.Finish(m, d);
    Check(!d->userdata[T::Armed], "Table armed feeder");
    int hg = -1;
    for (int i = 0; i < m->ngeom; i++)
      if (m->geom_contype[i] == 4) {
        hg = i;
        break;
      }
    Check(hg >= 0, "No hand collider");
    for (int k = 0; k < 3; k++)
      d->qpos[rq + k] = d->geom_xpos[3 * hg + k];
    mj_forward(m, d);
    task.Begin(d);
    task.Finish(m, d);
    Check(d->userdata[T::Armed], "Hand collision did not arm feeder");
    // Render scaling uses actual mjvGeom dimensions, not MJCF dimensions.
    mjvScene scene;
    mjv_defaultScene(&scene);
    mjv_makeScene(m, &scene, 2048);
    mjvOption option;
    mjv_defaultOption(&option);
    mjvCamera camera;
    mjv_defaultCamera(&camera);
    mjv_updateScene(m, d, &option, nullptr, &camera, mjCAT_ALL, &scene);
    bool faceFound = false;
    for (int i = 0; i < scene.ngeom; i++) {
      auto &g = scene.geoms[i];
      if (g.objtype != mjOBJ_GEOM)
        continue;
      if (g.objid == task.face) {
        auto size = RenderGeomSize(g);
        Near(size[0], .085, 1e-7, "Rendered racket width");
        Near(size[1], .085, 1e-7, "Rendered racket radius Y");
        Near(size[2], .006, 1e-7, "Rendered racket became thick cylinder");
        faceFound = true;
      }
    }
    Check(faceFound, "No racket in render scene");
    mjv_freeScene(&scene);
    // Assisted acquisition needs Grip near the handle, works with either hand,
    // survives motion and snapshots, and releases without cancelling the
    // feeder.
    for (int hand = 0; hand < 2; hand++) {
      mj_resetData(m, d);
      mj_forward(m, d);
      task.Reset(m, d);
      d->qpos[rq] = 4;
      d->qpos[rq + 1] = 0;
      d->qpos[rq + 2] = 2;
      mj_forward(m, d);
      task.UpdateGrip(m, d, 1, 1);
      Check(d->userdata[T::Holder] == 0, "Distant racket auto-grabbed");
      auto wanted = task.HoldingPose(m, d, hand);
      std::copy(wanted.p.begin(), wanted.p.end(), d->qpos + rq);
      std::copy(wanted.q.begin(), wanted.q.end(), d->qpos + rq + 3);
      mj_forward(m, d);
      task.UpdateGrip(m, d, 0, 0);
      Check(d->userdata[T::Holder] == 0, "Neutral Grip grabbed racket");
      task.UpdateGrip(m, d, hand == 0 ? .8 : 0, hand == 1 ? .8 : 0);
      Check(d->userdata[T::Holder] == hand + 1 && d->eq_active[task.weld[hand]],
            "Near-handle Grip failed");
      Check(!d->eq_active[task.weld[1 - hand]], "Two grip welds active");
      Check(d->userdata[T::Armed], "Grasp did not arm feeder");
      // A shakehand adaptation crosses the palm, rather than pointing like a
      // knife along the index finger. Check fingertips on opposite blade faces.
      std::string side = hand == 0 ? "left" : "right";
      for (int j = 0; j < m->njnt; j++) {
        const char *name = mj_id2name(m, mjOBJ_JOINT, j);
        if (name && std::string(name).rfind(side + "_hand_", 0) == 0)
          d->qpos[m->jnt_qposadr[j]] = task.HandControl(d, name, 0);
      }
      mj_forward(m, d);
      auto wristPose = task.WristPose(d, hand);
      auto held = task.RacketPose(d);
      auto axis = T::Rotate(wristPose, T::Rotate(held, {1, 0, 0}), true);
      Check(axis[2] > .85 && axis[0] > .3 && axis[0] < .5,
            "Handle points along fingers instead of crossing the palm");
      T::Vec tips[2];
      int tipIndex = 0;
      for (auto finger : {"index", "thumb"}) {
        int body = mj_name2id(
            m, mjOBJ_BODY, (side + "_hand_" + finger + "_finger_tip").c_str());
        Check(body >= 0, "Missing finger tip");
        T::Vec world;
        std::copy_n(d->xpos + 3 * body, 3, world.begin());
        tips[tipIndex] = T::Rotate(held, T::Add(world, held.p, -1), true);
        Check(T::Cylinder(tips[tipIndex], .085, .006).distance < .02,
              "Shakehand finger misses the blade");
        tipIndex++;
      }
      Check(tips[0][2] > .003 && tips[1][2] < -.003,
            "Index and thumb are not on opposite blade faces");
      Near(task.HandControl(d, side + "_hand_index_0_joint", 9),
           hand == 0 ? -.2 : .2, 1e-12, "Holding curled the index into a fist");
      Near(task.HandControl(d,
                            (hand == 0 ? "right" : "left") +
                                std::string("_hand_index_0_joint"),
                            .321),
           .321, 1e-12, "Holding changed the other hand");
      double serveAt = d->userdata[T::NextServe];
      for (int i = 0; i < 150; i++) {
        task.UpdateGrip(m, d, .5, .5);
        for (int aid = 0; aid < m->nu; aid++) {
          const char *name = mj_id2name(m, mjOBJ_ACTUATOR, aid);
          if (name && std::string(name).find("hand_") != std::string::npos)
            d->ctrl[aid] = task.HandControl(d, name, 0);
        }
        Step(m, d, task);
        auto actual = task.RacketPose(d), target = task.HoldingPose(m, d, hand);
        Check(T::Norm(T::Add(actual.p, target.p, -1)) < .004,
              "Racket slips from holding anchor");
        Check(std::abs(mju_dot(actual.q.data(), target.q.data(), 4)) > .999,
              "Racket orientation slips");
      }
      for (int j = 0; j < m->njnt; j++) {
        const char *name = mj_id2name(m, mjOBJ_JOINT, j);
        if (name && std::string(name).rfind(side + "_hand_", 0) == 0)
          Check(std::abs(d->qvel[m->jnt_dofadr[j]]) < 2,
                "Held fingers oscillate under the 1 ms integrator");
      }
      auto *heldCopy = mj_copyData(nullptr, m, d);
      Check(heldCopy->userdata[T::Holder] == hand + 1 &&
                heldCopy->eq_active[task.weld[hand]],
            "Snapshot lost grasp");
      mj_deleteData(heldCopy);
      // Hit impulse preserves the six grip velocity constraints.
      T::Vec contact{};
      std::copy_n(d->xpos + 3 * task.racket, 3, contact.begin());
      T::Pose heldPose = task.RacketPose(d);
      T::Vec n = T::Rotate(heldPose, {0, 0, 1});
      std::vector<double> jp(3 * m->nv), jr(3 * m->nv), hp(3 * m->nv),
          hr(3 * m->nv);
      mj_fwdPosition(m, d);
      mj_jac(m, d, jp.data(), jr.data(), contact.data(), task.racket);
      mj_jac(m, d, hp.data(), hr.data(), contact.data(), task.wrist[hand]);
      T::Vec surfaceVelocity{};
      for (int k = 0; k < 3; k++)
        for (int j = 0; j < m->nv; j++)
          surfaceVelocity[k] += jp[k * m->nv + j] * d->qvel[j];
      std::vector<double> beforeVelocity(d->qvel, d->qvel + m->nv);
      task.velocity = T::Add(surfaceVelocity, n, -5);
      task.RacketImpulse(m, d, n, contact);
      double afterNormal = 0;
      for (int j = 0; j < m->nv; j++) {
        for (int k = 0; k < 3; k++)
          afterNormal += n[k] * jp[k * m->nv + j] * d->qvel[j];
      }
      Near(T::Dot(task.velocity, n) - afterNormal, 5, 1e-7,
           "Held impact restitution differs from e=1");
      for (int k = 0; k < 3; k++) {
        double linear = 0, angular = 0;
        for (int j = 0; j < m->nv; j++) {
          double dv = d->qvel[j] - beforeVelocity[j];
          linear += (jp[k * m->nv + j] - hp[k * m->nv + j]) * dv;
          angular += (jr[k * m->nv + j] - hr[k * m->nv + j]) * dv;
        }
        Near(linear, 0, 1e-7, "Impact breaks linear grip constraint");
        Near(angular, 0, 1e-7, "Impact breaks angular grip constraint");
      }
      std::vector<double> releaseVelocity(d->qvel + task.racketVa,
                                          d->qvel + task.racketVa + 6);
      task.UpdateGrip(m, d, 0, 0);
      Check(!d->userdata[T::Holder] && !d->eq_active[task.weld[hand]],
            "Grip release failed");
      for (int k = 0; k < 6; k++)
        Near(d->qvel[task.racketVa + k], releaseVelocity[k], 1e-12,
             "Release destroyed throwing velocity");
      Near(d->userdata[T::NextServe], serveAt, 1e-12,
           "Grip release restarted feeder");
      task.Reset(m, d);
      Check(!d->userdata[T::Holder] && !d->eq_active[task.weld[0]] &&
                !d->eq_active[task.weld[1]],
            "Reset retained grip");
    }
    // Mid-session task state survives mj_copyData and is saved beside frames.
    auto *copy = mj_copyData(nullptr, m, d);
    for (int i = 0; i < T::StateSize; i++)
      Near(copy->userdata[i], d->userdata[i], 1e-12, "Snapshot task state");
    mj_deleteData(copy);
    std::filesystem::create_directories(argv[2]);
    std::ofstream meta(std::string(argv[2]) + "/recording_metadata.json");
    meta << "{\"scene\":\"tennis\"}";
    meta.close();
    EpisodeRecorder recorder;
    recorder.Start(argv[2], m);
    TrackingFrame f;
    std::array<float, 29> ref{};
    recorder.Frame(m, d, 1, f, false, false, false, true, 0, 0, 0, 0, false,
                   ref);
    auto ep = recorder.path();
    recorder.Stop();
    Check(std::filesystem::exists(ep + "/complete.json"),
          "Recording completion");
    std::ifstream states(ep + "/tennis.csv");
    std::string line;
    std::getline(states, line);
    Check(line.find("next_serve_s") != std::string::npos,
          "Task recording header");
    std::getline(states, line);
    Check(!line.empty(), "Missing task state row");
    std::cout
        << "tennis physics, timing, contact, scoring and recording passed\n";
    mj_deleteData(d);
    mj_deleteModel(m);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
