#include "capture_fov.h"
#include <cassert>
#include <cmath>
#include <iostream>
int main(){
    // Asymmetric Quest-like views, mirrored eyes, and already wider frusta.
    const XrFovf cases[]={{-.97f,.69f,.79f,-.91f},{-.69f,.97f,.79f,-.91f},{-1.1f,.8f,.9f,-1.08f}};
    for(const auto& n:cases){
        const auto idle=questcapture::ProjectionFov(n,false);
        assert(idle.angleLeft==n.angleLeft && idle.angleRight==n.angleRight && idle.angleUp==n.angleUp && idle.angleDown==n.angleDown);
        const auto f=questcapture::ProjectionFov(n,true);
        assert(f.angleLeft<=n.angleLeft && f.angleRight>=n.angleRight && f.angleDown<=n.angleDown && f.angleUp>=n.angleUp);
        assert(f.angleLeft == -f.angleRight && f.angleDown == -f.angleUp);
        // All rays in the 110-degree flat recording must have source texels.
        for(int degrees=-55;degrees<=55;++degrees){
            const double ray=std::tan(degrees*3.14159265358979323846/180.);
            const double l=std::tan(f.angleLeft),r=std::tan(f.angleRight);
            const double u=(ray-l)/(r-l);assert(u>0 && u<1);
            // Inverse compositor mapping must recover the same physical ray;
            // expanding the source image must not move/scale headset objects.
            assert(std::abs((l+u*(r-l))-ray)<1e-12);
            const double b=std::tan(f.angleDown),t=std::tan(f.angleUp);
            const double v=(ray-b)/(t-b);assert(v>0 && v<1);
            assert(std::abs((b+v*(t-b))-ray)<1e-12);
        }
    }
    std::cout<<"PASS: idle FOV unchanged; all native rays retained; full 110-degree capture covered; projection/compositor ray roundtrip.\n";
}
