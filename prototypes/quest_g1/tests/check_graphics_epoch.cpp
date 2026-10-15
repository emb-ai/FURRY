#include "graphics_epoch.h"
#include <stdexcept>
#include <cstdio>
static void Check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
int main(){
    uint64_t scene=0,hud=0;unsigned program=71,texture=18,vao=32,vbo=45;
    g1graphics::NewContext();Check(g1graphics::ForgetStale(scene,program,vao,vbo),"New context did not invalidate scene handles");
    Check(program==0&&vao==0&&vbo==0,"Scene retained deleted context resources");
    program=8;vao=9;vbo=10;Check(!g1graphics::ForgetStale(scene,program,vao,vbo)&&program==8,"Same context invalidated live resources");
    Check(g1graphics::ForgetStale(hud,texture)&&texture==0,"HUD did not independently observe new context");texture=3;
    g1graphics::NewContext();Check(g1graphics::ForgetStale(scene,program,vao,vbo)&&program==0,"Recreated context reused old scene shader");
    Check(g1graphics::ForgetStale(hud,texture)&&texture==0,"Recreated context reused old atlas");
    puts("Graphics context recreation invalidates scene/HUD caches without deleting new-context handles");
}
