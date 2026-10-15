#pragma once
#include <array>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <mujoco/mujoco.h>

// Exact 157-value body-only observation/action contract used by evaluate_mujoco.py.
// Normalization is embedded in the ONNX. Reference is absolute XYZ + WXYZ + 29 joints.
namespace dexweave {
inline constexpr std::array<double,29> Home={-.312,0,0,.669,-.363,0,-.312,0,0,.669,-.363,0,0,0,0,.2,.2,0,.6,0,0,0,.2,-.2,0,.6,0,0,0};
inline constexpr std::array<double,29> Armature={.010177520,.025101925,.010177520,.025101925,.007219450,.007219450,.010177520,.025101925,.010177520,.025101925,.007219450,.007219450,.010177520,.007219450,.007219450,.003609725,.003609725,.003609725,.003609725,.003609725,.00425,.00425,.003609725,.003609725,.003609725,.003609725,.003609725,.00425,.00425};
inline constexpr std::array<double,29> Effort={88,139,88,139,50,50,88,139,88,139,50,50,88,50,50,25,25,25,25,25,5,5,25,25,25,25,25,5,5};
inline double Kp(int i){return Armature[i]*std::pow(20*3.141592653589793,2);}
inline double Kd(int i){return 4*Armature[i]*20*3.141592653589793;}
inline double Scale(int i){return .25*Effort[i]/Kp(i);}
inline std::array<float,157> Observe(const mjData* d,const mjData* ref,int torso,
    const std::array<int,29>& qa,const std::array<int,29>& va,
    const std::array<double,29>& dq,const std::array<float,29>& applied){
    std::array<float,157> out{};
    for(int i=0;i<29;i++){out[i]=ref->qpos[qa[i]];out[29+i]=dq[i];}
    double delta[3],local[3],relative[9];
    mju_sub3(delta,ref->xpos+3*torso,d->xpos+3*torso);
    const double* r=d->xmat+9*torso;
    mju_mulMatTVec(local,r,delta,3,3);
    for(int i=0;i<3;i++)out[58+i]=local[i];
    mju_mulMatTMat(relative,r,ref->xmat+9*torso,3,3,3);
    // Numpy [:,:2].reshape(6): row-major first two columns.
    for(int i=0;i<3;i++)for(int j=0;j<2;j++)out[61+2*i+j]=relative[3*i+j];
    // MuJoCo freejoint angular qvel is already in root-local coordinates.
    for(int i=0;i<3;i++)out[67+i]=d->qvel[3+i];
    for(int i=0;i<29;i++){
        out[70+i]=d->qpos[qa[i]]-Home[i];out[99+i]=d->qvel[va[i]];out[128+i]=applied[i];
    }
    for(float v:out)if(!std::isfinite(v))throw std::runtime_error("Nonfinite DexWeave observation");
    return out;
}
inline void Action(const float* action,std::array<float,29>& target,std::array<float,29>& applied){
    for(int i=0;i<29;i++){
        if(!std::isfinite(action[i]))throw std::runtime_error("Nonfinite DexWeave action");
        double next=Home[i]+Scale(i)*action[i];
        if(i<15){double limit=(i==2||i==8?.4:.32);next=std::clamp(next,double(target[i])-limit,double(target[i])+limit);}
        target[i]=next;applied[i]=(next-Home[i])/Scale(i);
    }
}
} // namespace dexweave
