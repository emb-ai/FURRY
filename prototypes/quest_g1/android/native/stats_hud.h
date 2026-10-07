#pragma once
#include "runtime_stats.h"
#include "retarget.h"
#include <cmath>
#include <GLES3/gl32.h>
#include <fstream>
#include <algorithm>
#include <vector>
#include <string>
#include <cstdio>
#include <stdexcept>

// Lightweight native glyph atlas; drawn every eye even when physics is stalled.
inline int DrawStatsHud(const float* headViewProjection,const std::string& assets,const RuntimeStats& s,
                         double fps,double drawMs,uint64_t skipped,double now,int status,bool active,int exportStatus,
                         const TrackingFrame& raw,double rawAgeMs,bool firstPerson,float visualScale){
    static GLuint program=0,texture=0,vao=0,vbo=0;
    if(!program){
        auto shader=[](GLenum type,const char* source){GLuint h=glCreateShader(type);glShaderSource(h,1,&source,nullptr);glCompileShader(h);GLint ok;glGetShaderiv(h,GL_COMPILE_STATUS,&ok);if(!ok)throw std::runtime_error("Stats shader failed");return h;};
        GLuint v=shader(GL_VERTEX_SHADER,R"(#version 320 es
layout(location=0) in vec2 p;layout(location=1) in vec2 uv;uniform mat4 headViewProjection;out vec2 tex;
void main(){gl_Position=headViewProjection*vec4(p,-1.6,1);tex=uv;})");
        GLuint f=shader(GL_FRAGMENT_SHADER,R"(#version 320 es
precision mediump float;in vec2 tex;uniform sampler2D atlas;uniform vec4 tint;out vec4 color;
void main(){color=vec4(tint.rgb,tint.a*texture(atlas,tex).r);})");
        program=glCreateProgram();glAttachShader(program,v);glAttachShader(program,f);glLinkProgram(program);glDeleteShader(v);glDeleteShader(f);
        GLint ok;glGetProgramiv(program,GL_LINK_STATUS,&ok);if(!ok)throw std::runtime_error("Stats shader link failed");
        std::vector<unsigned char> pixels(512*240);std::ifstream input(assets+"/stats_font.bin",std::ios::binary);input.read(reinterpret_cast<char*>(pixels.data()),pixels.size());if(!input)throw std::runtime_error("Stats font missing");
        glGenTextures(1,&texture);glBindTexture(GL_TEXTURE_2D,texture);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D,0,GL_R8,512,240,0,GL_RED,GL_UNSIGNED_BYTE,pixels.data());
        glGenVertexArrays(1,&vao);glBindVertexArray(vao);glGenBuffers(1,&vbo);glBindBuffer(GL_ARRAY_BUFFER,vbo);
        glEnableVertexAttribArray(0);glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,4*sizeof(float),nullptr);
        glEnableVertexAttribArray(1);glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,4*sizeof(float),(void*)(2*sizeof(float)));
    }
    const char* states[]={"CALIBRATE A","TRACKING","PAUSED B","BODY INVALID","PHYSICS FAULT: X","TRACKING / POSE ERROR"};
    const char* exports[]={"Y: RECORD / SAVE ZIP","ZIP: COPYING TO DOWNLOADS","ZIP: Download/G1Quest","ZIP EXPORT FAILED (data kept)"};
    const char* reasons[]={"-","STAGE RECENTER: A","BODY RECALIBRATED: A","FALL/EXCEPTION: X"};
    char text[2048],cameraText[80];
    if(s.cameraPositionError>=0)std::snprintf(cameraText,sizeof(cameraText),"HEAD %.1f cm / %.1f deg",100*s.cameraPositionError,s.cameraOrientationError);
    else std::snprintf(cameraText,sizeof(cameraText),"HEAD target: OFF");
    std::snprintf(text,sizeof(text),
        "VIEW %s [L-stick]\nFPS %4.0f  CPU %.1f ms\nSIM %.2fx  MT %d  WARN %d\nPHYS %.1f GMR %.1f NN %.1f ms\nCONTACT %d  DEPTH %.1f mm\nSOLVER %d  CONSTRAINT %d\nINPUT %.0f  AGE %.0f ms\nROOT %.2f m  LEG %.0f deg\n%s\nSCALE %.2fx  TILT %.0f deg\nCMD %+.2f / %+.2f m/s\n%s\n%s\n%s",
        firstPerson?"EGO":"OBSERVER",fps,drawMs,active?s.realTimeFactor:0.,s.physicsWorkers,s.warnings,
        s.physicsMs,s.gmrMs,s.inferenceMs,s.contacts,s.depthMm,s.solverIterations,s.constraints,
        s.inputAgeMs,s.published>0?std::max(0.,now-s.published)*1000:0.,s.positionError,s.legErrorDegrees,
        cameraText,visualScale,s.tilt,s.commandXY[0],s.commandXY[1],
        active?states[std::clamp(status,0,5)]:"XR FOCUS PAUSED",reasons[std::clamp(s.reason,0,3)],exports[std::clamp(exportStatus,0,3)]);
    std::vector<float> vertices;vertices.reserve(24000);
    auto quad=[&](float x,float y,float w,float h,int code){int cell=code-32;float u=(cell%16)/16.f,v=(cell/16)/6.f;
        const float q[]={x,y,u,v, x+w,y-h,u+20/512.f,v+1/6.f, x+w,y,u+20/512.f,v,
                         x,y,u,v, x,y-h,u,v+1/6.f, x+w,y-h,u+20/512.f,v+1/6.f};vertices.insert(vertices.end(),q,q+24);};
    quad(-1.18f,.96f,.97f,.85f,127);
    float x=-1.16f,y=.945f;for(char c:std::string(text)){if(c=='\n'){x=-1.16f;y-=.055f;continue;}if(c>=32 && c<127)quad(x,y,.028f,.052f,c);x+=.026f;}
    const int statsEnd=vertices.size()/4;
    struct Batch{int first,count;std::array<float,4> color;};
    std::vector<Batch> batches;
    auto batch=[&](int first,std::array<float,4> color){batches.push_back({first,int(vertices.size()/4)-first,color});};
    int first=vertices.size()/4;quad(.35f,.96f,.83f,.85f,127);batch(first,{.025f,.04f,.06f,.82f});
    auto label=[&](float y,const std::string& value,std::array<float,4> color){
        int begin=vertices.size()/4;float x=.375f;
        for(char c:value){if(c>=32 && c<127)quad(x,y,.024f,.040f,c);x+=.022f;}
        batch(begin,color);
    };
    label(.945f,"RAW META / BEFORE GMR",{.8f,1.f,.95f,1.f});
    label(.911f,"ORANGE LEGS = ESTIMATED",{1.f,.65f,.25f,1.f});
    bool fresh=raw.body.valid && rawAgeMs<200 && std::abs(raw.xr_time_ns-raw.body.time_ns)<200000000LL;
    char sourceText[100];std::snprintf(sourceText,sizeof(sourceText),"%s  conf %.2f  age %.0f ms",fresh?"LIVE":"STALE/INVALID",raw.body.confidence,rawAgeMs);
    label(.157f,sourceText,fresh?std::array<float,4>{.8f,1.f,.95f,1.f}:std::array<float,4>{1.f,.4f,.4f,1.f});
    // Uniform miniature display scale only: input joints are the untouched
    // STAGE poses, before anatomical scaling, ankle offsets or GMR solving.
    // Keep STAGE floor at the bottom; do not auto-fit current pose (squats must
    // visibly get shorter). Bind height supplies a constant scale per skeleton.
    double stature=raw.body.rest[1].position[1]-std::min(raw.body.rest[6].position[1],raw.body.rest[7].position[1])+.35;
    double scale=.61/std::clamp(stature,.8,2.5),q[9];mju_quat2Mat(q,raw.head.quaternion.data());
    double fx=-q[2],fz=-q[8],length=std::hypot(fx,fz);if(length<.2){fx=0;fz=-1;length=1;}fx/=length;fz/=length;
    std::array<std::array<float,2>,15> points{};std::array<bool,15> valid{};
    for(int j=0;j<15;j++){
        const auto& p=j<14?raw.body.joints[j]:raw.head;
        double dx=p.position[0]-raw.body.joints[0].position[0],dz=p.position[2]-raw.body.joints[0].position[2];
        // Front three-quarter view, relative to headset yaw. Orange remains
        // inference, even when OpenXR reports a valid pose.
        double horizontal=.94*(-fz*dx+fx*dz)+.34*(fx*dx+fz*dz);
        points[j]={float(.765+scale*horizontal),float(.20+scale*p.position[1])};
        auto flags=j<14?raw.body.flags[j]:raw.location_flags[0];
        valid[j]=(flags&3)==3 && std::isfinite(points[j][0]) && std::isfinite(points[j][1]) &&
            points[j][0]>.37 && points[j][0]<1.16 && points[j][1]>.19 && points[j][1]<.875;
    }
    auto solidTriangle=[&](float ax,float ay,float bx,float by,float cx,float cy){
        const float uvx=15.5f/16,uvy=5.5f/6;
        const float t[]={ax,ay,uvx,uvy,bx,by,uvx,uvy,cx,cy,uvx,uvy};vertices.insert(vertices.end(),t,t+12);
    };
    auto line=[&](int a,int b){if(!valid[a]||!valid[b])return;
        float ax=points[a][0],ay=points[a][1],bx=points[b][0],by=points[b][1];
        float len=std::hypot(bx-ax,by-ay);if(len<1e-6)return;float nx=-(by-ay)/len*.003f,ny=(bx-ax)/len*.003f;
        solidTriangle(ax+nx,ay+ny,ax-nx,ay-ny,bx+nx,by+ny);solidTriangle(ax-nx,ay-ny,bx-nx,by-ny,bx+nx,by+ny);
    };
    const std::pair<int,int> edges[]={{0,1},{1,8},{8,10},{10,12},{1,9},{9,11},{11,13},{1,14},
                                     {0,2},{2,4},{4,6},{0,3},{3,5},{5,7}};
    for(int group=0;group<2;group++){
        first=vertices.size()/4;
        for(int i=group?8:0;i<(group?14:8);i++)line(edges[i].first,edges[i].second);
        for(int j=0;j<15;j++)if(valid[j] && ((j>=2 && j<=7)==bool(group))){
            float r=j==14?.014f:.008f;
            for(int k=0;k<8;k++){float a=k*6.2831853f/8,b=(k+1)*6.2831853f/8;
                solidTriangle(points[j][0],points[j][1],points[j][0]+r*std::cos(a),points[j][1]+r*std::sin(a),points[j][0]+r*std::cos(b),points[j][1]+r*std::sin(b));}
        }
        batch(first,fresh?(group?std::array<float,4>{1.f,.65f,.25f,1.f}:std::array<float,4>{.3f,.85f,1.f,1.f}):std::array<float,4>{.5f,.5f,.5f,.7f});
    }
    // Shared metre coordinates in the central head frame, not per-eye NDC.
    // The real eye poses and asymmetric FOVs provide the stereo disparity.
    for(size_t i=0;i<vertices.size();i+=4){
        bool left=i/4<size_t(statsEnd);float factor=left?.90f/.97f:.54f/.83f;
        vertices[i]=left?-.98f+(vertices[i]+1.18f)*factor:.30f+(vertices[i]-.35f)*factor;
        vertices[i+1]=.68f+(vertices[i+1]-.96f)*factor;
    }
    glDisable(GL_DEPTH_TEST);glEnable(GL_BLEND);glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(program);glUniformMatrix4fv(glGetUniformLocation(program,"headViewProjection"),1,GL_FALSE,headViewProjection);
    glActiveTexture(GL_TEXTURE0);glBindTexture(GL_TEXTURE_2D,texture);glUniform1i(glGetUniformLocation(program,"atlas"),0);
    glBindVertexArray(vao);glBindBuffer(GL_ARRAY_BUFFER,vbo);glBufferData(GL_ARRAY_BUFFER,vertices.size()*sizeof(float),vertices.data(),GL_STREAM_DRAW);
    glUniform4f(glGetUniformLocation(program,"tint"),.025f,.04f,.06f,.82f);glDrawArrays(GL_TRIANGLES,0,6);
    glUniform4f(glGetUniformLocation(program,"tint"),status==4?1.f:.80f,status==4?.4f:1.f,.95f,1);glDrawArrays(GL_TRIANGLES,6,statsEnd-6);
    for(const auto& b:batches){glUniform4fv(glGetUniformLocation(program,"tint"),1,b.color.data());glDrawArrays(GL_TRIANGLES,b.first,b.count);}
    glBindVertexArray(0);glUseProgram(0);glDisable(GL_BLEND);glEnable(GL_DEPTH_TEST);
    return int(vertices.size()/12);
}
