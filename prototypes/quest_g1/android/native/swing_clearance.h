#pragma once
#include "foot_floor.h"

// Source-driven clearance experiment, not a generated step schedule. Meta
// ankle separation is a heuristic, not a measured swing/contact phase.
class SwingClearance {
    FootFloor floor;
public:
    std::array<double,2> lastLift{};
    void Load(const mjModel* physical,const mjModel* reference){floor.Load(physical,reference);}
    void Correct(std::vector<TrackedPose>& targets,double leftMinusRightAnkle,double clearance){
        if(targets.size()!=14 || !std::isfinite(leftMinusRightAnkle) ||
           !std::isfinite(clearance) || clearance<0 || clearance>.08)
            throw std::runtime_error("Invalid swing-clearance input");
        lastLift={};if(clearance==0)return;
        auto heights=floor.Heights(targets[6],targets[7]);
        for(int side=0;side<2;side++){
            double separation=(side==0?1:-1)*leftMinusRightAnkle;
            double u=std::clamp((separation-.012)/(.040-.012),0.,1.);
            if(u==0)continue;
            // Blend the lift itself: a tiny positive separation must not
            // abruptly correct a pre-existing below-floor target.
            lastLift[side]=std::clamp(clearance-heights[side],0.,.06)*u*u*(3-2*u);
            targets[6+side].position[2]+=lastLift[side];
        }
    }
};
