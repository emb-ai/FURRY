#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <mujoco/mujoco.h>
#include <stdexcept>
#include <string>
#include <vector>
#include "collection_task.h"

// Spin-free 40 mm/2.7 g ball. MuJoCo owns the articulated robot and the free
// racket (optionally held by a wrist weld); this task owns ball integration and
// hard impacts (no soft solver restitution). All persistent task state is in
// mjData::userdata for snapshots.
class TennisTask {
public:
  using Vec = std::array<double, 3>;
  enum State {
    Armed,
    NextServe,
    Active, // 0: hidden; 1: live rally; 2: missed, still physical; 3: scored, still physical
    Hit,
    OwnBounces,
    Serves,
    Returns,
    Misses,
    Holder, // 0: free; 1: left hand; 2: right hand
    LegacyStateSize,
    Seed=9, PreparedServe, LaunchX, LaunchY, LaunchZ, BounceX, BounceY,
    StateSize
  };
  static constexpr double Radius = .02, Mass = .0027;
  static constexpr double Near = .4, Far = 3.14, Mid = 1.77, HalfWidth = .7625,
                          Top = .76;
  struct Pose {
    Vec p{};
    std::array<double, 4> q{1, 0, 0, 0};
  };
  struct Surface {
    Vec normal{}, point{};
    double distance = 0;
  };
  int ball = -1, racket = -1, face = -1, handle = -1, ballGeom = -1, qa = -1,
      va = -1;
  std::vector<int> obstacles;
  int stateSize = 8, racketQa = -1, racketVa = -1;
  std::array<int, 2> wrist{-1, -1}, palm{-1, -1}, weld{-1, -1};
  bool assisted = false, shapedGrip = false, varied = false;
  int aimMocap=-1;
  std::array<double, 7> fingerPose{};
  double delay = 7, interval = 4, speed = 5.5, cd = .5, rho = 1.204,
         tableRestitution = .9;
  Vec position{}, velocity{}, launch{};
  Pose previous{};

  static Vec Add(Vec a, Vec b, double factor = 1) {
    for (int k = 0; k < 3; k++)
      a[k] += factor * b[k];
    return a;
  }
  static double Dot(Vec a, Vec b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }
  static double Norm(Vec a) { return std::sqrt(Dot(a, a)); }
  static Vec Unit(Vec a) {
    double n = Norm(a);
    if (n < 1e-12)
      return {0, 0, 1};
    for (auto &x : a)
      x /= n;
    return a;
  }
  static Vec Rotate(const Pose &pose, Vec v, bool inverse = false) {
    double mat[9];
    mju_quat2Mat(mat, pose.q.data());
    Vec out{};
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        out[i] += (inverse ? mat[3 * j + i] : mat[3 * i + j]) * v[j];
    return out;
  }
  static Pose At(const Pose &a, const Pose &b, double u) {
    Pose p;
    for (int k = 0; k < 3; k++)
      p.p[k] = a.p[k] + u * (b.p[k] - a.p[k]);
    double sign = mju_dot(a.q.data(), b.q.data(), 4) < 0 ? -1 : 1;
    for (int k = 0; k < 4; k++)
      p.q[k] = (1 - u) * a.q[k] + u * sign * b.q[k];
    mju_normalize4(p.q.data());
    return p;
  }
  // Closest surface of a cylinder, including its rim. Cylinder axis is local Z.
  static Surface Cylinder(Vec p, double radius, double half) {
    double r = std::hypot(p[0], p[1]);
    Vec q = p;
    if (r > radius) {
      q[0] *= radius / r;
      q[1] *= radius / r;
    }
    q[2] = std::clamp(p[2], -half, half);
    Vec delta = Add(p, q, -1);
    double dist = Norm(delta);
    if (dist < 1e-12) {
      if (half - std::abs(p[2]) <= radius - r) {
        q[2] = p[2] < 0 ? -half : half;
      } else {
        double scale = radius / std::max(r, 1e-12);
        q[0] = r < 1e-12 ? radius : p[0] * scale;
        q[1] = p[1] * scale;
      }
      delta = Add(q, p, -1);
      return {Unit(delta), q, -Norm(delta)};
    }
    return {Unit(delta), q, dist};
  }
  static Surface Box(Vec p, Vec half) {
    Vec q{};
    for (int k = 0; k < 3; k++)
      q[k] = std::clamp(p[k], -half[k], half[k]);
    Vec delta = Add(p, q, -1);
    double dist = Norm(delta);
    if (dist < 1e-12) {
      int axis = 0;
      for (int k = 1; k < 3; k++)
        if (half[k] - std::abs(p[k]) < half[axis] - std::abs(p[axis]))
          axis = k;
      q[axis] = p[axis] < 0 ? -half[axis] : half[axis];
      delta = Add(q, p, -1);
      return {Unit(delta), q, -Norm(delta)};
    }
    return {Unit(delta), q, dist};
  }
  static Vec Acceleration(Vec v, double dragCd = .5, double density = 1.204) {
    double k =
        .5 * density * dragCd * 3.141592653589793 * Radius * Radius / Mass;
    Vec a = Add(Vec{}, v, -k * Norm(v));
    a[2] -= 9.81;
    return a;
  }
  // Explicit midpoint at <=0.1 ms. No Magnus, spin, tangential drag or wind.
  static void Fly(Vec &p, Vec &v, double dt, double dragCd = .5,
                  double density = 1.204) {
    Vec mid = Add(v, Acceleration(v, dragCd, density), dt * .5);
    p = Add(p, mid, dt);
    v = Add(v, Acceleration(mid, dragCd, density), dt);
  }
  static Vec Reflect(Vec v, Vec surfaceVelocity, Vec n, double e = 1) {
    return Add(v, n, -(1 + e) * Dot(Add(v, surfaceVelocity, -1), n));
  }
  void Initialize(const mjModel *m) {
    ball = mj_name2id(m, mjOBJ_BODY, "tennis_ball");
    if (ball < 0)
      return;
    racket = mj_name2id(m, mjOBJ_BODY, "tennis_racket");
    face = mj_name2id(m, mjOBJ_GEOM, "tennis_racket_face");
    handle = mj_name2id(m, mjOBJ_GEOM, "tennis_racket_handle");
    int joint = mj_name2id(m, mjOBJ_JOINT, "tennis_ball_free"),
        config = mj_name2id(m, mjOBJ_NUMERIC, "tennis_config");
    if (racket < 0 || face < 0 || handle < 0 || joint < 0 || config < 0 ||
        (m->nuserdata != 8 && m->nuserdata != LegacyStateSize && m->nuserdata != StateSize) ||
        m->numeric_size[config] != 6)
      throw std::runtime_error("Invalid tennis scene");
    qa = m->jnt_qposadr[joint];
    va = m->jnt_dofadr[joint];
    const double *c = m->numeric_data + m->numeric_adr[config];
    delay = c[0];
    interval = c[1];
    speed = c[2];
    cd = c[3];
    rho = c[4];
    tableRestitution = c[5];
    for (int i = 0; i < 6; i++)
      if (!std::isfinite(c[i]))
        throw std::runtime_error("Invalid tennis configuration");
    if (delay < 5 || delay > 10 || interval < 2 || speed < 2 || speed > 20 ||
        cd <= 0 || rho <= 0 || tableRestitution <= 0 || tableRestitution > 1)
      throw std::runtime_error("Tennis configuration outside supported range");
    ballGeom = mj_name2id(m, mjOBJ_GEOM, "tennis_ball_geom");
    int top = mj_name2id(m, mjOBJ_GEOM, "tennis_top"),
        net = mj_name2id(m, mjOBJ_GEOM, "tennis_net");
    (void)net;
    obstacles.clear();
    for (int g = 0; g < m->ngeom; g++) {
      if ((m->geom_contype[g] || m->geom_conaffinity[g]) &&
          m->geom_bodyid[g] != racket && g != ballGeom && g != top)
        obstacles.push_back(g);
    }
    stateSize = m->nuserdata;
    assisted = stateSize >= LegacyStateSize;
    varied = stateSize == StateSize;
    if(varied){
      int b=mj_name2id(m,mjOBJ_BODY,"tennis_aim_head");
      if(b<0 || m->body_mocapid[b]<0)throw std::runtime_error("Missing feeder head");
      aimMocap=m->body_mocapid[b];
    }
    if (assisted) {
      int rj = mj_name2id(m, mjOBJ_JOINT, "tennis_racket_free");
      if (rj < 0)
        throw std::runtime_error("Missing tennis racket joint");
      racketQa = m->jnt_qposadr[rj];
      racketVa = m->jnt_dofadr[rj];
      int side = 0;
      for (auto name : {"left", "right"}) {
        wrist[side] = mj_name2id(
            m, mjOBJ_BODY, (std::string(name) + "_wrist_yaw_link").c_str());
        palm[side] = mj_name2id(
            m, mjOBJ_GEOM,
            (std::string(name) + "_hand_palm_link_collision").c_str());
        weld[side] = mj_name2id(m, mjOBJ_EQUALITY,
                                (std::string("tennis_grip_") + name).c_str());
        if (wrist[side] < 0 || palm[side] < 0 || weld[side] < 0)
          throw std::runtime_error("Missing tennis grip anchor");
        side++;
      }
    }
    int fingers = mj_name2id(m, mjOBJ_NUMERIC, "tennis_grip_fingers");
    if (fingers >= 0) {
      if (!assisted || m->numeric_size[fingers] != 7)
        throw std::runtime_error("Invalid tennis finger pose");
      std::copy_n(m->numeric_data + m->numeric_adr[fingers], 7,
                  fingerPose.begin());
      for (double value : fingerPose)
        if (!std::isfinite(value) || std::abs(value) > 3.141592653589793)
          throw std::runtime_error("Invalid tennis finger angle");
      shapedGrip = true;
    }
    launch = varied ? LaunchTo(1.15,0) : LaunchVelocity();
  }
  bool Enabled() const { return ball >= 0; }
  Pose RacketPose(const mjData *d) const {
    Pose p;
    std::copy_n(d->xpos + 3 * racket, 3, p.p.begin());
    std::copy_n(d->xquat + 4 * racket, 4, p.q.begin());
    return p;
  }
  void Write(const mjModel *m, mjData *d) {
    std::copy(position.begin(), position.end(), d->qpos + qa);
    d->qpos[qa + 3] = 1;
    std::fill_n(d->qpos + qa + 4, 3, 0);
    std::copy(velocity.begin(), velocity.end(), d->qvel + va);
    std::fill_n(d->qvel + va + 3, 3, 0);
    mj_kinematics(m, d);
    mj_comPos(m, d); // Current render/servo FK after custom ball step.
  }
  void Reset(const mjModel *m, mjData *d) {
    if (!Enabled())
      return;
    std::fill_n(d->userdata, stateSize, 0);
    if (assisted)
      for (int eq : weld)
        d->eq_active[eq] = false;
    position = {3.05, 0, -1};
    velocity = {};
    Write(m, d);
    if(varied)SetSeed(m,d,0);
  }
  Pose WristPose(const mjData *d, int hand) const {
    Pose p;
    std::copy_n(d->xpos + 3 * wrist[hand], 3, p.p.begin());
    std::copy_n(d->xquat + 4 * wrist[hand], 4, p.q.begin());
    return p;
  }
  Pose HoldingPose(const mjModel *m, const mjData *d, int hand) const {
    Pose w = WristPose(d, hand), p;
    const double *rel = m->eq_data + mjNEQDATA * weld[hand];
    p.p = Add(w.p, Rotate(w, {rel[3], rel[4], rel[5]}));
    mju_mulQuat(p.q.data(), w.q.data(), rel + 6);
    return p;
  }
  void Capture(const mjModel *m, mjData *d, int hand) {
    Pose pose = HoldingPose(m, d, hand);
    // Snap only during a deliberate near-handle squeeze; thereafter MuJoCo's
    // weld carries the free racket. No per-frame teleport or world-fixed pose.
    std::vector<double> jp(3 * m->nv), jr(3 * m->nv);
    Vec linear{}, angular{};
    mj_jac(m, d, jp.data(), jr.data(), pose.p.data(), wrist[hand]);
    for (int k = 0; k < 3; k++)
      for (int j = 0; j < m->nv; j++) {
        linear[k] += jp[k * m->nv + j] * d->qvel[j];
        angular[k] += jr[k * m->nv + j] * d->qvel[j];
      }
    angular = Rotate(pose, angular, true);
    std::copy(pose.p.begin(), pose.p.end(), d->qpos + racketQa);
    std::copy(pose.q.begin(), pose.q.end(), d->qpos + racketQa + 3);
    std::copy(linear.begin(), linear.end(), d->qvel + racketVa);
    std::copy(angular.begin(), angular.end(), d->qvel + racketVa + 3);
    for (int h = 0; h < 2; h++)
      d->eq_active[weld[h]] = h == hand;
    d->userdata[Holder] = hand + 1;
    Touch(d);
    mj_forward(m, d);
  }
  double HandControl(const mjData *d, const std::string &name,
                     double fallback) const {
    if (!shapedGrip)
      return fallback; // Legacy scene keeps its recorded fist controls.
    int hand = name.rfind("left", 0) == 0 ? 0 : 1;
    if (d->userdata[Holder] != hand + 1)
      return fallback;
    const char *parts[] = {"thumb_0", "thumb_1",  "thumb_2", "index_0",
                           "index_1", "middle_0", "middle_1"};
    for (int k = 0; k < 7; k++)
      if (name.find(parts[k]) != std::string::npos)
        return fingerPose[k] * (k == 0 || hand == 0 ? 1 : -1);
    return fallback;
  }
  void UpdateGrip(const mjModel *m, mjData *d, double left, double right) {
    if (!Enabled() || !assisted)
      return;
    const double grips[2] = {std::clamp(left, 0., 1.),
                             std::clamp(right, 0., 1.)};
    int held = int(d->userdata[Holder]) - 1;
    if (held >= 0 && held < 2) {
      if (grips[held] > .25)
        return;
      d->eq_active[weld[held]] = false;
      d->userdata[Holder] = 0;
      return; // Release preserves the current free-joint velocity; no stealing.
    }
    Pose pose = RacketPose(d);
    int nearest = -1;
    double distance = .12;
    Vec handleCentre{};
    std::copy_n(m->geom_pos + 3 * handle, 3, handleCentre.begin());
    Vec half{};
    std::copy_n(m->geom_size + 3 * handle, 3, half.begin());
    for (int h = 0; h < 2; h++) {
      // The grab point is the handle centre of the configured holding pose,
      // so visual pose and proximity acquisition use the same anchor.
      Pose holding = HoldingPose(m, d, h);
      Vec handPoint = Add(holding.p, Rotate(holding, handleCentre));
      Vec local =
          Add(Rotate(pose, Add(handPoint, pose.p, -1), true), handleCentre, -1);
      double gap = Box(local, half).distance;
      if (gap < .03 &&
          mj_geomDistance(m, d, handle, palm[h], .001, nullptr) <= 0)
        Touch(d);
      if (grips[h] >= .65 && gap <= distance) {
        distance = gap;
        nearest = h;
      }
    }
    if (nearest >= 0)
      Capture(m, d, nearest);
  }
  void Begin(const mjData *d) {
    if (!Enabled())
      return;
    std::copy_n(d->qpos + qa, 3, position.begin());
    std::copy_n(d->qvel + va, 3, velocity.begin());
    previous = RacketPose(d);
  }
  void Touch(mjData *d) {
    if (!d->userdata[Armed]) {
      d->userdata[Armed] = 1;
      d->userdata[NextServe] = d->time + delay;
    }
  }
  void Miss(mjData *d) {
    // Scoring must never teleport a physical ball out of the scene.
    if (d->userdata[Active] == 1) {
      d->userdata[Misses]++;
      d->userdata[Active] = 2;
    }
  }
  void Retire(mjData *d) {
    Miss(d);
    d->userdata[Active] = 0;
    position = {3.05, 0, -1};
    velocity = {};
  }
  Surface ObstacleSurface(const mjModel* m, mjData* d, int g) const {
    Pose pose;
    std::copy_n(d->geom_xpos+3*g,3,pose.p.begin());
    mju_mat2Quat(pose.q.data(),d->geom_xmat+9*g);
    Vec local=Rotate(pose,Add(position,pose.p,-1),true);
    const auto* size=m->geom_size+3*g;
    Surface surface;
    switch(m->geom_type[g]) {
      case mjGEOM_PLANE: surface={{0,0,1},{local[0],local[1],0},local[2]};break;
      case mjGEOM_BOX: surface=Box(local,{size[0],size[1],size[2]});break;
      case mjGEOM_CYLINDER: surface=Cylinder(local,size[0],size[1]);break;
      case mjGEOM_SPHERE: {
        Vec n=Unit(local);surface={n,Add({},n,size[0]),Norm(local)-size[0]};break;
      }
      case mjGEOM_CAPSULE: {
        Vec centre{0,0,std::clamp(local[2],-size[1],size[1])};
        Vec delta=Add(local,centre,-1),n=Unit(delta);
        surface={n,Add(centre,n,size[0]),Norm(delta)-size[0]};break;
      }
      default: {
        // MuJoCo signed distance includes the ball radius. Its witness vector
        // reverses under penetration, so divide by signed distance.
        std::copy(position.begin(),position.end(),d->geom_xpos+3*ballGeom);
        double segment[6]{};
        double gap=mj_geomDistance(m,d,ballGeom,g,.01,segment);
        if(gap>=.01 || std::abs(gap)<1e-10)return {{},{},Radius+1};
        Vec n{};for(int k=0;k<3;k++)n[k]=(segment[k]-segment[k+3])/gap;
        return {Unit(n),{segment[3],segment[4],segment[5]},gap+Radius};
      }
    }
    surface.normal=Rotate(pose,surface.normal);
    surface.point=Add(pose.p,Rotate(pose,surface.point));
    return surface;
  }
  // Solve vertical launch velocity so flight under drag first reaches z=.78
  // on our half at x=.95. The speed setting is total muzzle speed, not vx.
  Vec LaunchVelocity() const {
    auto height = [&](double angle) {
      Vec p{3.0, 0, 1.15},
          v{-speed * std::cos(angle), 0, speed * std::sin(angle)};
      for (int i = 0; i < 20000 && p[0] > .95; i++)
        Fly(p, v, .0001, cd, rho);
      return p[2];
    };
    double lo = -.4, hi = .6;
    if (height(lo) > .78 || height(hi) < .78)
      throw std::runtime_error("Tennis speed cannot reach bounce target");
    for (int i = 0; i < 36; i++) {
      double a = (lo + hi) * .5;
      if (height(a) > .78)
        hi = a;
      else
        lo = a;
    }
    double a = (lo + hi) * .5;
    return {-speed * std::cos(a), 0, speed * std::sin(a)};
  }
  Vec LaunchTo(double x,double y)const{
    // Fixed muzzle centre; solve pitch at the configured total speed under
    // the same drag integrator used in flight. Only choose the low branch.
    const double dx=x-3., distance=std::hypot(dx,y);
    auto height=[&](double angle){
      Vec p{3.,0,1.32},v{speed*std::cos(angle)*dx/distance,speed*std::cos(angle)*y/distance,speed*std::sin(angle)};
      for(int i=0;i<30000&&p[0]>x;i++)Fly(p,v,.0001,cd,rho);
      return p[2];
    };
    double lo=-.25,hi=.72;
    if(height(lo)>.78||height(hi)<.78)throw std::runtime_error("Unreachable tennis bounce");
    for(int i=0;i<30;i++){double a=(lo+hi)/2;if(height(a)>.78)hi=a;else lo=a;}
    double a=(lo+hi)/2;
    return {speed*std::cos(a)*dx/distance,speed*std::cos(a)*y/distance,speed*std::sin(a)};
  }
  void Aim(const mjModel* m,mjData* d){
    if(!varied)return;
    Vec dir=Unit({d->userdata[LaunchX],d->userdata[LaunchY],d->userdata[LaunchZ]});
    mju_quatZ2Vec(d->mocap_quat+4*aimMocap,dir.data());
    double p[3]={3.,0,1.32};std::copy_n(p,3,d->mocap_pos+3*aimMocap);
    mj_kinematics(m,d);mj_comPos(m,d);
  }
  void PrepareServe(const mjModel* m,mjData* d){
    if(!varied)return;
    auto* s=d->userdata;
    uint32_t rng=uint32_t(s[Seed])^(uint32_t(s[Serves]+1)*0x9e3779b9u);
    // Moderate spread, fully inside the table, announced >=1.5 s beforehand.
    double x=CollectionTask::Uniform(rng,1.02,1.30),y=CollectionTask::Uniform(rng,-.38,.38);
    Vec v=LaunchTo(x,y);std::copy(v.begin(),v.end(),s+LaunchX);
    s[BounceX]=x;s[BounceY]=y;s[PreparedServe]=s[Serves]+1;Aim(m,d);
  }
  void SetSeed(const mjModel* m,mjData* d,uint32_t seed){
    if(!varied)return;
    d->userdata[Seed]=seed;PrepareServe(m,d);
  }
  // e=1 with finite articulated racket mass, equal/opposite impulse, no
  // tangential impulse or ball rotation. An active assisted grip shares recoil
  // with G1 through its weld constraint.
  void RacketImpulse(const mjModel *m, mjData *d, Vec n, Vec point) {
    Impact(m,d,n,point,racket,1);
  }
  void Impact(const mjModel *m, mjData *d, Vec n, Vec point, int body, double restitution) {
    if(m->body_weldid[body]==0) {
      double relative=Dot(velocity,n);
      if(relative<0)velocity=Reflect(velocity,{},n,-relative<.2?0:restitution);
      return;
    }
    mj_fwdPosition(m, d);
    std::vector<double> jac(3 * m->nv), normalJac(m->nv), response(m->nv);
    mj_jac(m, d, jac.data(), nullptr, point.data(), body);
    double vn = 0;
    for (int j = 0; j < m->nv; j++) {
      for (int k = 0; k < 3; k++)
        normalJac[j] += n[k] * jac[k * m->nv + j];
      vn += normalJac[j] * d->qvel[j];
    }
    double relative = Dot(velocity, n) - vn;
    if (relative >= 0)
      return;
    mj_solveM(m, d, response.data(), normalJac.data(), 1);
    // During assisted holding, project inverse inertia through the active
    // six-dimensional weld, so recoil is shared with G1 rather than breaking
    // the grasp on every ball impact.
    if (body == racket && assisted && d->userdata[Holder] > 0) {
      int hand = int(d->userdata[Holder]) - 1;
      std::vector<double> hp(3 * m->nv), hr(3 * m->nv), rp(3 * m->nv),
          rr(3 * m->nv), constraint(6 * m->nv), inverse(6 * m->nv);
      mj_jac(m, d, hp.data(), hr.data(), point.data(), wrist[hand]);
      mj_jac(m, d, rp.data(), rr.data(), point.data(), racket);
      for (int k = 0; k < 3; k++)
        for (int j = 0; j < m->nv; j++) {
          constraint[k * m->nv + j] = rp[k * m->nv + j] - hp[k * m->nv + j];
          constraint[(k + 3) * m->nv + j] =
              rr[k * m->nv + j] - hr[k * m->nv + j];
        }
      mj_solveM(m, d, inverse.data(), constraint.data(), 6);
      double system[36]{}, rhs[6]{}, multipliers[6]{};
      for (int a = 0; a < 6; a++) {
        rhs[a] = mju_dot(constraint.data() + a * m->nv, response.data(), m->nv);
        for (int b = 0; b < 6; b++)
          system[6 * a + b] = mju_dot(constraint.data() + a * m->nv,
                                      inverse.data() + b * m->nv, m->nv);
      }
      if (mju_cholFactor(system, 6, 1e-12) != 6)
        throw std::runtime_error("Singular tennis grip inertia");
      mju_cholSolve(multipliers, system, rhs, 6);
      for (int a = 0; a < 6; a++)
        for (int j = 0; j < m->nv; j++)
          response[j] -= inverse[a * m->nv + j] * multipliers[a];
    }
    double inverseMass =
        std::max(0., mju_dot(normalJac.data(), response.data(), m->nv));
    double e = -relative < .2 ? 0 : restitution;
    double impulse = -(1+e) * relative / (1 / Mass + inverseMass);
    velocity = Add(velocity, n, impulse / Mass);
    for (int j = 0; j < m->nv; j++)
      d->qvel[j] -= impulse * response[j];
  }
  void Finish(const mjModel *m, mjData *d) {
    if (!Enabled())
      return;
    mj_kinematics(m, d);
    mj_comPos(m, d);
    Pose current = RacketPose(d);
    auto *s = d->userdata;
    for (int i = 0; i < d->ncon && !s[Armed]; i++) {
      auto &c = d->contact[i];
      int a = c.geom[0], b = c.geom[1];
      if (a < 0 || b < 0)
        continue;
      int other = m->geom_bodyid[a] == racket   ? b
                  : m->geom_bodyid[b] == racket ? a
                                                : -1;
      // Only physical palm/finger geometry can arm the feeder.
      if (other >= 0 && c.dist <= 0 && (m->geom_contype[other] & 4))
        Touch(d);
    }
    if (assisted && !s[Armed]) {
      for (int g = 0; g < m->ngeom && !s[Armed]; g++)
        if (m->geom_contype[g] == 4) {
          Vec centre{};
          std::copy_n(d->geom_xpos + 3 * g, 3, centre.begin());
          for (int tool : {face, handle}) {
            Vec toolCentre{};
            std::copy_n(d->geom_xpos + 3 * tool, 3, toolCentre.begin());
            if (Norm(Add(centre, toolCentre, -1)) <
                    m->geom_rbound[g] + m->geom_rbound[tool] &&
                mj_geomDistance(m, d, g, tool, .001, nullptr) <= 0) {
              Touch(d);
              break;
            }
          }
        }
    }
    if(varied && s[PreparedServe]<=s[Serves] && (!s[Active] || d->time>=s[NextServe]-1.5))PrepareServe(m,d);
    if (s[Armed] && d->time + 1e-9 >= s[NextServe]) {
      if (s[Active])
        Miss(d);
      position = varied ? Vec{3.,0,1.32} : Vec{3.0,0,1.15};
      velocity = varied ? Vec{s[LaunchX],s[LaunchY],s[LaunchZ]} : launch;
      s[Active] = 1;
      s[Hit] = 0;
      s[OwnBounces] = 0;
      s[Serves]++;
      s[NextServe] += interval;
      Write(m, d);
      return;
    }
    if (!s[Active]) {
      Write(m, d);
      return;
    }
    // Bound ball AND rotating racket displacement to 5 mm per subdivision.
    double angle =
        2 *
        std::acos(std::clamp(
            std::abs(mju_dot(previous.q.data(), current.q.data(), 4)), 0., 1.));
    double dt = m->opt.timestep;
    int count = std::max(
        10, int(std::ceil((Norm(velocity) * dt +
                           Norm(Add(current.p, previous.p, -1)) + .2 * angle) /
                          .005)));
    double h = dt / count;
    for (int i = 0; i < count && s[Active]; i++) {
      Fly(position, velocity, h, cd, rho);
      Pose pose = At(previous, current, double(i + 1) / count);
      Vec local = Rotate(pose, Add(position, pose.p, -1), true);
      Surface paddle =
          Cylinder(local, m->geom_size[3 * face], m->geom_size[3 * face + 1]);
      Vec handleCentre{}, handleHalf{};
      std::copy_n(m->geom_pos + 3 * handle, 3, handleCentre.begin());
      std::copy_n(m->geom_size + 3 * handle, 3, handleHalf.begin());
      Surface grip = Box(Add(local, handleCentre, -1), handleHalf);
      grip.point = Add(grip.point, handleCentre);
      if (grip.distance < paddle.distance)
        paddle = grip;
      if (paddle.distance < Radius) {
        Vec n = Rotate(pose, paddle.normal),
            point = Add(pose.p, Rotate(pose, paddle.point));
        Vec surfaceV = Add(Add(current.p, Rotate(current, paddle.point)),
                           Add(previous.p, Rotate(previous, paddle.point)), -1);
        for (auto &v : surfaceV)
          v /= dt;
        if (Dot(Add(velocity, surfaceV, -1), n) < 0) {
          RacketImpulse(m, d, n, point);
          s[Hit] = 1;
        }
        position = Add(point, n, Radius + 1e-7);
      }
      Surface top =
          Box(Add(position, {Mid, 0, Top - .015}, -1), {1.37, HalfWidth, .015});
      if (top.distance < Radius) {
        bool approaching=Dot(velocity,top.normal)<0;
        bool landing = top.normal[2] > .99 && position[0] >= Near &&
                       position[0] <= Far && std::abs(position[1]) <= HalfWidth;
        if (approaching && s[Active]==1) {
          if (landing && s[Hit] && position[0]>Mid) {
            s[Returns]++;
            s[Active]=3;
          } else if (!landing || s[Hit] || position[0]>Mid || ++s[OwnBounces]>1) {
            Miss(d);
          }
        }
        position=Add(Add(Vec{Mid,0,Top-.015},top.point),top.normal,Radius+1e-7);
        if(approaching)velocity=Reflect(velocity,{},top.normal,
                                      -Dot(velocity,top.normal)<.2?0:tableRestitution);
      }
      // Net, floor, G1, table legs and feeder all have collision response.
      // Contact geometry comes from the scene, including convex mesh hulls.
      for (int g : obstacles) {
        Vec centre{};std::copy_n(d->geom_xpos+3*g,3,centre.begin());
        if(m->geom_type[g]!=mjGEOM_PLANE &&
           Norm(Add(position,centre,-1))>m->geom_rbound[g]+Radius)continue;
        Surface surface=ObstacleSurface(m,d,g);
        if(surface.distance>=Radius)continue;
        const char* name=mj_id2name(m,mjOBJ_GEOM,g);
        double e=name && std::string(name)=="tennis_net" ? .15 : .7;
        Impact(m,d,surface.normal,surface.point,m->geom_bodyid[g],e);
        position=Add(surface.point,surface.normal,Radius+1e-7);
        Miss(d);
      }
      // Only retire far outside the playable room or on the next serve.
      if (position[2] < -1 || position[0] < -6 || position[0] > Far + 6 ||
          std::abs(position[1]) > 6 || position[2] > 8)
        Retire(d);
    }
    Write(m, d);
  }
  std::string Status(const mjData *d) const {
    if (!Enabled())
      return "";
    auto *s = d->userdata;
    char text[180];
    if (!s[Armed])
      return assisted ? "Поднесите кисть к рукояти и сожмите Grip"
                      : "Возьмите ракетку: касание запустит подачу";
    std::snprintf(
        text, sizeof(text), "%sПопадания %d/%d  Промахи %d  Подача %.1f с",
        assisted && s[Holder] > 0 ? "Grip: держим | " : "", int(s[Returns]),
        int(s[Serves]), int(s[Misses]), std::max(0., s[NextServe] - d->time));
    return text;
  }
};
