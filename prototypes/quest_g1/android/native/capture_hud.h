#pragma once
#include "guided_capture.h"
#include "retarget.h"
#include <GLES3/gl32.h>
#include <vector>
#include <cstdio>
#include <stdexcept>

inline int DrawCaptureHud(const float* vp,const std::string& assets,const GuidedCapture& g,
                          bool focused,bool fresh,int recording,int exportStatus,
                          const TrackingFrame& raw,double rawAgeMs){
    static GLuint program=0,texture=0,vao=0,vbo=0;
    if(!program){
        auto shader=[](GLenum type,const char* src){GLuint s=glCreateShader(type);glShaderSource(s,1,&src,nullptr);glCompileShader(s);GLint ok;glGetShaderiv(s,GL_COMPILE_STATUS,&ok);if(!ok)throw std::runtime_error("Capture shader failed");return s;};
        GLuint v=shader(GL_VERTEX_SHADER,R"(#version 320 es
layout(location=0) in vec2 p;layout(location=1) in vec2 uv;uniform mat4 vp;out vec2 tex;
void main(){gl_Position=vp*vec4(p,-1.3,1);tex=uv;})");
        GLuint f=shader(GL_FRAGMENT_SHADER,R"(#version 320 es
precision mediump float;in vec2 tex;uniform sampler2D atlas;uniform vec4 tint;out vec4 color;
void main(){color=vec4(tint.rgb,tint.a*texture(atlas,tex).r);})");
        program=glCreateProgram();glAttachShader(program,v);glAttachShader(program,f);glLinkProgram(program);glDeleteShader(v);glDeleteShader(f);
        GLint ok;glGetProgramiv(program,GL_LINK_STATUS,&ok);if(!ok)throw std::runtime_error("Capture shader link failed");
        std::vector<unsigned char> bytes(512*440);std::ifstream in(assets+"/capture_font.bin",std::ios::binary);in.read(reinterpret_cast<char*>(bytes.data()),bytes.size());if(!in)throw std::runtime_error("Capture font missing");
        glGenTextures(1,&texture);glBindTexture(GL_TEXTURE_2D,texture);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D,0,GL_R8,512,440,0,GL_RED,GL_UNSIGNED_BYTE,bytes.data());
        glGenVertexArrays(1,&vao);glBindVertexArray(vao);glGenBuffers(1,&vbo);glBindBuffer(GL_ARRAY_BUFFER,vbo);
        glEnableVertexAttribArray(0);glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,4*sizeof(float),nullptr);
        glEnableVertexAttribArray(1);glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,4*sizeof(float),(void*)(2*sizeof(float)));
    }
    std::vector<float> vertices;
    auto quad=[&](float x,float y,float w,float h,int cell){
        float u=(cell%16)*32/512.f,v=(cell/16)*40/440.f;
        const float q[]={x,y,u,v,x+w,y-h,u+20/512.f,v+40/440.f,x+w,y,u+20/512.f,v,
                         x,y,u,v,x,y-h,u,v+40/440.f,x+w,y-h,u+20/512.f,v+40/440.f};
        vertices.insert(vertices.end(),q,q+24);
    };
    auto label=[&](float x,float y,const std::string& s,float advance=.016f,int maxChars=44){
        size_t count=0;for(unsigned char c:s)if((c&0xc0)!=0x80)count++;
        float fit=std::min(1.f,float(maxChars)/std::max(size_t(1),count));
        for(size_t i=0;i<s.size();){
            unsigned c=static_cast<unsigned char>(s[i++]);
            if((c&0xe0)==0xc0 && i<s.size())c=((c&31)<<6)|(static_cast<unsigned char>(s[i++])&63);
            int cell=c>=32&&c<=126?int(c)-32:c>=1040&&c<=1103?96+int(c)-1040:c==1025?160:c==1105?161:31;
            quad(x,y,advance*1.0625f*fit,.033f,cell);x+=advance*fit;
        }
    };
    auto line=[&](float y,const std::string& s){label(-.36f,y,s);};
    bool full=g.mode!=0;
    quad(-.385f,full?.58f:-.22f,.77f,full?.35f:.065f,95);
    if(!full)line(-.232f,"Menu: план, калибровка и старт записи");
    else{
        char text[160];int status=g.Status(focused,fresh);
        line(.568f,g.mode==1?"ОБУЧЕНИЕ - 15 МИНУТ":"ОТДЕЛЬНЫЙ ТЕСТ - 5 МИНУТ");
        std::snprintf(text,sizeof(text),"Этап %02d/%02d: %s",g.stage+1,g.Count(),g.Current().title);line(.526f,text);
        line(.484f,g.Current().hint);
        int left=int(std::ceil(g.Remaining())),total=int(g.Total()+1e-6);
        std::snprintf(text,sizeof(text),"Осталось %02d:%02d   Записано %02d:%02d",left/60,left%60,total/60,total%60);line(.442f,text);
        if(recording==2)line(.400f,"Ошибка записи: данные сохранены частично");
        else if(status==0)line(.400f,g.calibrated?"ГОТОВО   Y: начать   Menu: план":"Menu: калибровка, затем Y: начать");
        else if(status==1)line(.400f,"Встаньте прямо, смотрите вперёд. Menu/A");
        else if(status==2)line(.400f,"ПАУЗА   B: продолжить   A: калибровка");
        else if(status==3)line(.400f,"Нет скелета: таймер на паузе");
        else if(status==4){std::snprintf(text,sizeof(text),"Приготовьтесь: %d",int(std::ceil(g.countdown)));line(.400f,text);}
        else if(status==5)line(.400f,"ИДЁТ ЗАПИСЬ   B: пауза   Y: закончить");
        else line(.400f,"ГОТОВО   Y: новая запись");
        line(.358f,exportStatus==1?"ZIP: копируется в Download/G1Quest":exportStatus==3?"Ошибка ZIP: исходники остались в очках":exportStatus==2?"ZIP: сохранён в Download/G1Quest":"Файлы скелета сохраняются в очках");
        line(.316f,g.mode==2?"Тест храните отдельно от обучения":"Мало места: развернитесь или нажмите B");
        line(.274f,"Menu: план   Y: сохранить   B: пауза");
    }
    const int textEnd=vertices.size()/4;
    struct Batch{int first,count;std::array<float,4> tint;};
    std::vector<Batch> batches;
    auto batch=[&](int first,std::array<float,4> tint){batches.push_back({first,int(vertices.size()/4)-first,tint});};
    if(full){
        int first=vertices.size()/4;quad(.43f,.58f,.48f,.76f,95);batch(first,{.025f,.04f,.06f,.8f});
        first=vertices.size()/4;
        label(.45f,.566f,"ТВОЙ СКЕЛЕТ",.014f,30);
        label(.45f,.524f,"Оранжевый: оценка ног",.014f,30);
        bool bodyFresh=raw.body.valid && rawAgeMs>=0 && rawAgeMs<200 &&
            (raw.location_flags[0]&3)==3 && std::abs(raw.xr_time_ns-raw.body.time_ns)<200000000LL;
        label(.45f,-.137f,bodyFresh?"Руки: следуйте заданию":"Нет свежего скелета",.014f,30);
        batch(first,{.83f,1.f,.94f,1.f});
        // Untouched STAGE joints. Pelvis-relative horizontal centring, uniform
        // bind-height scale and a fixed floor preserve visible step/squat height.
        double stature=raw.body.rest[1].position[1]-std::min(raw.body.rest[6].position[1],raw.body.rest[7].position[1])+.35;
        // Reserve overhead space for the head and raised arms during standing tasks.
        double scale=.36/std::clamp(stature,.8,2.5),rotation[9];mju_quat2Mat(rotation,raw.head.quaternion.data());
        double fx=-rotation[2],fz=-rotation[8],length=std::hypot(fx,fz);
        if(!std::isfinite(length)||length<.2){fx=0;fz=-1;length=1;}fx/=length;fz/=length;
        std::array<std::array<float,2>,15> points{};std::array<bool,15> valid{};
        for(int j=0;j<15;j++){
            const auto& p=j<14?raw.body.joints[j]:raw.head;
            double dx=p.position[0]-raw.body.joints[0].position[0],dz=p.position[2]-raw.body.joints[0].position[2];
            double horizontal=.94*(-fz*dx+fx*dz)+.34*(fx*dx+fz*dz);
            points[j]={float(.67+scale*horizontal),float(-.08+scale*p.position[1])};
            auto flags=j<14?raw.body.flags[j]:raw.location_flags[0];
            valid[j]=(flags&3)==3 && std::isfinite(points[j][0]) && std::isfinite(points[j][1]) &&
                points[j][0]>.445 && points[j][0]<.895 && points[j][1]>-.1 && points[j][1]<.475;
        }
        auto triangle=[&](float ax,float ay,float bx,float by,float cx,float cy){
            constexpr float u=15.5f/16,v=5.5f/11;
            const float t[]={ax,ay,u,v,bx,by,u,v,cx,cy,u,v};vertices.insert(vertices.end(),t,t+12);
        };
        auto bone=[&](int a,int b){
            if(!valid[a]||!valid[b])return;
            float ax=points[a][0],ay=points[a][1],bx=points[b][0],by=points[b][1];
            float len=std::hypot(bx-ax,by-ay);if(len<1e-6)return;
            float nx=-(by-ay)/len*.002f,ny=(bx-ax)/len*.002f;
            triangle(ax+nx,ay+ny,ax-nx,ay-ny,bx+nx,by+ny);triangle(ax-nx,ay-ny,bx-nx,by-ny,bx+nx,by+ny);
        };
        constexpr std::pair<int,int> edges[]={{0,1},{1,8},{8,10},{10,12},{1,9},{9,11},{11,13},{1,14},
                                             {0,2},{2,4},{4,6},{0,3},{3,5},{5,7}};
        for(int group=0;group<2;group++){
            first=vertices.size()/4;
            for(int i=group?8:0;i<(group?14:8);i++)bone(edges[i].first,edges[i].second);
            for(int j=0;j<15;j++)if(valid[j] && ((j>=2&&j<=7)==bool(group))){
                float r=j==14?.012f:.006f;
                for(int k=0;k<8;k++){float a=k*6.2831853f/8,b=(k+1)*6.2831853f/8;
                    triangle(points[j][0],points[j][1],points[j][0]+r*std::cos(a),points[j][1]+r*std::sin(a),points[j][0]+r*std::cos(b),points[j][1]+r*std::sin(b));}
            }
            batch(first,bodyFresh?(group?std::array<float,4>{1.f,.65f,.25f,1.f}:std::array<float,4>{.3f,.85f,1.f,1.f}):std::array<float,4>{.5f,.5f,.5f,.65f});
        }
    }
    glDisable(GL_DEPTH_TEST);glEnable(GL_BLEND);glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(program);glUniformMatrix4fv(glGetUniformLocation(program,"vp"),1,GL_FALSE,vp);
    glActiveTexture(GL_TEXTURE0);glBindTexture(GL_TEXTURE_2D,texture);glUniform1i(glGetUniformLocation(program,"atlas"),0);
    glBindVertexArray(vao);glBindBuffer(GL_ARRAY_BUFFER,vbo);glBufferData(GL_ARRAY_BUFFER,vertices.size()*sizeof(float),vertices.data(),GL_STREAM_DRAW);
    glUniform4f(glGetUniformLocation(program,"tint"),.025f,.04f,.06f,.88f);glDrawArrays(GL_TRIANGLES,0,6);
    glUniform4f(glGetUniformLocation(program,"tint"),.83f,1.f,.94f,1);glDrawArrays(GL_TRIANGLES,6,textEnd-6);
    for(const auto& b:batches){glUniform4fv(glGetUniformLocation(program,"tint"),1,b.tint.data());glDrawArrays(GL_TRIANGLES,b.first,b.count);}
    glBindVertexArray(0);glUseProgram(0);glDisable(GL_BLEND);glEnable(GL_DEPTH_TEST);
    return vertices.size()/12;
}
