#pragma once
#include "retarget.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

// Causal heuristic on raw Meta estimates, NOT a measured contact classifier.
// Coordinates are OpenXR STAGE (Y up). Raise the swing ankle toward 10 cm above
// the unmodified opposite/support ankle; this is not sole clearance.
class VirtualLegLift {
    int64_t last=0;BodyFrame cached;
    std::array<TrackedPose,2> previous{};
    std::array<double,2> speed{},vertical{};
    double age=0,weight=0;
    int selected=-1;
    static double Distance(const double* a,const double* b){double v[3];mju_sub3(v,a,b);return mju_norm3(v);}
    static void RotateBone(TrackedPose& pose,const double* oldDir,const double* newDir){
        double a[3],b[3],axis[3],dq[4],q[4];mju_copy3(a,oldDir);mju_copy3(b,newDir);mju_normalize3(a);mju_normalize3(b);
        mju_cross(axis,a,b);double dot=std::clamp(mju_dot3(a,b),-1.,1.);
        if(dot<-.999999){double fallback[3]={1,0,0};if(std::abs(a[0])>.9){fallback[0]=0;fallback[2]=1;}mju_cross(axis,a,fallback);mju_normalize3(axis);mju_axisAngle2Quat(dq,axis,3.141592653589793);}
        else{dq[0]=1+dot;for(int k=0;k<3;k++)dq[1+k]=axis[k];mju_normalize4(dq);}
        mju_mulQuat(q,dq,pose.quaternion.data());std::copy_n(q,4,pose.quaternion.begin());
    }
public:
    int swingSide=-1;
    std::array<double,2> lift{},rawClearance{};
    bool reachLimited=false;
    void Reset(){last=0;selected=swingSide=-1;age=weight=0;speed={};vertical={};lift={};rawClearance={};reachLimited=false;}
    static bool LegIK(BodyFrame& b,int side,double raise){
        auto& hip=b.joints[2+side];auto& knee=b.joints[4+side];auto& ankle=b.joints[6+side];
        auto h=hip.position,k=knee.position,a=ankle.position;
        double l1=Distance(h.data(),k.data()),l2=Distance(k.data(),a.data());
        if(l1<.15||l2<.15||l1>.8||l2>.8)throw std::runtime_error("Virtual leg IK: invalid segment lengths");
        auto target=a;target[1]+=raise;double minReach=std::abs(l1-l2)+1e-7,maxReach=l1+l2-1e-7;
        bool limited=false;double d=Distance(h.data(),target.data());
        if(d<minReach||d>maxReach){
            limited=true;double lo=0,hi=raise;
            for(int n=0;n<40;n++){double mid=(lo+hi)/2;target=a;target[1]+=mid;double r=Distance(h.data(),target.data());if(r>=minReach&&r<=maxReach)lo=mid;else hi=mid;}
            target=a;target[1]+=lo;d=Distance(h.data(),target.data());
        }
        if(d<1e-8)return true;
        double axis[3],bend[3];mju_sub3(axis,target.data(),h.data());mju_scl3(axis,axis,1/d);
        mju_sub3(bend,k.data(),h.data());double projection=mju_dot3(bend,axis);mju_addToScl3(bend,axis,-projection);
        if(mju_normalize3(bend)<1e-6){double fallback[3]={0,0,-1};if(std::abs(axis[2])>.9){fallback[0]=1;fallback[2]=0;}projection=mju_dot3(fallback,axis);mju_copy3(bend,fallback);mju_addToScl3(bend,axis,-projection);mju_normalize3(bend);}
        double x=(l1*l1-l2*l2+d*d)/(2*d),y=std::sqrt(std::max(0.,l1*l1-x*x));
        for(int j=0;j<3;j++)knee.position[j]=h[j]+x*axis[j]+y*bend[j];ankle.position=target;
        double oldDir[3],newDir[3];mju_sub3(oldDir,k.data(),h.data());mju_sub3(newDir,knee.position.data(),h.data());RotateBone(hip,oldDir,newDir);
        mju_sub3(oldDir,a.data(),k.data());mju_sub3(newDir,target.data(),knee.position.data());RotateBone(knee,oldDir,newDir);
        return limited;
    }
    BodyFrame Apply(const BodyFrame& raw){
        if(!raw.valid){Reset();return raw;}
        if(last&&raw.time_ns==last)return cached;
        double dt=last?(raw.time_ns-last)*1e-9:0;
        if(dt<=0||dt>.2){Reset();last=raw.time_ns;previous={raw.joints[6],raw.joints[7]};cached=raw;return cached;}
        lift={};reachLimited=false;double alpha=1-std::exp(-dt/.04);
        for(int side=0;side<2;side++){
            const auto& p=raw.joints[6+side].position;const auto& old=previous[side].position;
            speed[side]+=alpha*(std::hypot(p[0]-old[0],p[2]-old[2])/dt-speed[side]);
            vertical[side]+=alpha*((p[1]-old[1])/dt-vertical[side]);
            rawClearance[side]=p[1]-raw.joints[7-side].position[1];
        }
        int candidate=rawClearance[0]>rawClearance[1]?0:1;
        if(selected>=0){age+=dt;if((rawClearance[selected]<.004&&vertical[selected]<.02)||age>1.2)selected=-1;}
        if(selected<0&&rawClearance[candidate]>.012&&(speed[candidate]>.12||vertical[candidate]>.04)){selected=candidate;age=0;}
        // A smooth envelope prevents an instantaneous 10 cm jump at detection.
        double phase=selected<0?0:std::clamp((rawClearance[selected]-.004)/.036,0.,1.);
        phase=phase*phase*(3-2*phase);
        if(selected!=swingSide)weight=0;
        weight+=std::clamp(phase-weight,-dt/.06,dt/.06);swingSide=selected;
        cached=raw;
        if(selected>=0){
            double amount=weight*std::max(0.,.10-rawClearance[selected]);
            if(amount>0){reachLimited=LegIK(cached,selected,amount);lift[selected]=cached.joints[6+selected].position[1]-raw.joints[6+selected].position[1];}
        }
        previous={raw.joints[6],raw.joints[7]};last=raw.time_ns;return cached;
    }
};
