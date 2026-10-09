#pragma once
#include "world_skeleton.h"
#include <GLES3/gl32.h>
#include <stdexcept>

inline int DrawWorldSkeleton(const float* vp,const WorldSkeleton& skeleton){
    static GLuint program=0,vao=0,vbo=0;
    if(!program){
        auto shader=[](GLenum type,const char* src){GLuint s=glCreateShader(type);glShaderSource(s,1,&src,nullptr);glCompileShader(s);GLint ok;glGetShaderiv(s,GL_COMPILE_STATUS,&ok);if(!ok)throw std::runtime_error("World skeleton shader failed");return s;};
        GLuint v=shader(GL_VERTEX_SHADER,R"(#version 320 es
layout(location=0) in vec3 p;layout(location=1) in vec4 tint;uniform mat4 vp;out vec4 rgba;
void main(){gl_Position=vp*vec4(p,1);rgba=tint;})");
        GLuint f=shader(GL_FRAGMENT_SHADER,R"(#version 320 es
precision mediump float;in vec4 rgba;out vec4 color;void main(){color=rgba;})");
        program=glCreateProgram();glAttachShader(program,v);glAttachShader(program,f);glLinkProgram(program);glDeleteShader(v);glDeleteShader(f);
        GLint ok;glGetProgramiv(program,GL_LINK_STATUS,&ok);if(!ok)throw std::runtime_error("World skeleton shader link failed");
        glGenVertexArrays(1,&vao);glBindVertexArray(vao);glGenBuffers(1,&vbo);glBindBuffer(GL_ARRAY_BUFFER,vbo);
        glEnableVertexAttribArray(0);glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,7*sizeof(float),nullptr);
        glEnableVertexAttribArray(1);glVertexAttribPointer(1,4,GL_FLOAT,GL_FALSE,7*sizeof(float),(void*)(3*sizeof(float)));
    }
    using V=std::array<double,3>;std::vector<float> vertices;vertices.reserve(14000);
    auto tri=[&](V a,V b,V c,const std::array<float,4>& color){for(auto p:{a,b,c}){for(auto x:p)vertices.push_back(float(x));vertices.insert(vertices.end(),color.begin(),color.end());}};
    // Octahedral dots and square tubes keep the stereo overlay below 500 triangles.
    for(const auto& mark:skeleton.marks){
        V p[6];for(int i=0;i<6;i++){p[i]=mark.p;p[i][i/2]+=(i%2?-1:1)*mark.radius;}
        for(int x=0;x<2;x++)for(int y=2;y<4;y++)for(int z=4;z<6;z++)tri(p[x],p[y],p[z],mark.color);
    }
    for(const auto& bone:skeleton.bones){
        V d;for(int i=0;i<3;i++)d[i]=bone.b[i]-bone.a[i];double len=mju_norm3(d.data());if(len<1e-6)continue;for(auto& x:d)x/=len;
        V axis=std::abs(d[1])<.9?V{0,1,0}:V{1,0,0},u{},w{};mju_cross(u.data(),d.data(),axis.data());mju_normalize3(u.data());mju_cross(w.data(),d.data(),u.data());
        V a[4],b[4];for(int k=0;k<4;k++)for(int i=0;i<3;i++){double off=bone.radius*((k==0||k==3?1:-1)*u[i]+(k<2?1:-1)*w[i]);a[k][i]=bone.a[i]+off;b[k][i]=bone.b[i]+off;}
        for(int k=0;k<4;k++){int n=(k+1)%4;tri(a[k],b[k],a[n],bone.color);tri(a[n],b[k],b[n],bone.color);}
    }
    glDisable(GL_CULL_FACE);glDisable(GL_DEPTH_TEST);glDepthMask(GL_FALSE);glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA,GL_ONE,GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(program);glUniformMatrix4fv(glGetUniformLocation(program,"vp"),1,GL_FALSE,vp);
    glBindVertexArray(vao);glBindBuffer(GL_ARRAY_BUFFER,vbo);glBufferData(GL_ARRAY_BUFFER,vertices.size()*sizeof(float),vertices.data(),GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES,0,vertices.size()/7);
    glBindVertexArray(0);glUseProgram(0);glDisable(GL_BLEND);glDepthMask(GL_TRUE);glEnable(GL_DEPTH_TEST);
    return int(vertices.size()/21);
}
