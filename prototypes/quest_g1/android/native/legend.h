#pragma once
#include <GLES3/gl32.h>
#include <fstream>
#include <vector>
#include <stdexcept>

// A world-space instruction card, two triangles for both flat text and background.
inline void DrawLegend(const float* vp, const std::string& assets,int status) {
    static GLuint program=0, vao=0, vbo=0, textures[18]{};
    if(!program){
        auto compile=[](GLenum type,const char* source){
            GLuint s=glCreateShader(type);glShaderSource(s,1,&source,nullptr);glCompileShader(s);
            GLint ok=0;glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
            if(!ok)throw std::runtime_error("Control card shader failed");return s;
        };
        const char* vertex=R"(#version 320 es
layout(location=0) in vec3 position;
layout(location=1) in vec2 uv;
uniform mat4 vp;out vec2 tex;
void main(){gl_Position=vp*vec4(position,1);tex=uv;})";
        const char* fragment=R"(#version 320 es
precision mediump float;in vec2 tex;uniform sampler2D card;out vec4 color;
void main(){color=texture(card,tex);})";
        GLuint v=compile(GL_VERTEX_SHADER,vertex),f=compile(GL_FRAGMENT_SHADER,fragment);
        program=glCreateProgram();glAttachShader(program,v);glAttachShader(program,f);glLinkProgram(program);
        GLint ok=0;glGetProgramiv(program,GL_LINK_STATUS,&ok);
        if(!ok)throw std::runtime_error("Control card link failed");
        glDeleteShader(v);glDeleteShader(f);
        const float vertices[]={-2.0f,1.9f,-2.5f,0,0, -0.7f,1.25f,-2.5f,1,1, -0.7f,1.9f,-2.5f,1,0,
                                -2.0f,1.9f,-2.5f,0,0, -2.0f,1.25f,-2.5f,0,1, -0.7f,1.25f,-2.5f,1,1};
        glGenVertexArrays(1,&vao);glBindVertexArray(vao);glGenBuffers(1,&vbo);glBindBuffer(GL_ARRAY_BUFFER,vbo);
        glBufferData(GL_ARRAY_BUFFER,sizeof(vertices),vertices,GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,5*sizeof(float),nullptr);
        glEnableVertexAttribArray(1);glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,5*sizeof(float),(void*)(3*sizeof(float)));
        glGenTextures(18,textures);
        for(int i=0;i<18;i++){
        std::vector<unsigned char> rgba(1024*512*4);
        std::ifstream input(assets+"/controls_"+std::to_string(i)+".rgba",std::ios::binary);input.read(reinterpret_cast<char*>(rgba.data()),rgba.size());
        if(!input)throw std::runtime_error("Missing control card asset");
        glBindTexture(GL_TEXTURE_2D,textures[i]);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,1024,512,0,GL_RGBA,GL_UNSIGNED_BYTE,rgba.data());
        }
    }
    glUseProgram(program);glUniformMatrix4fv(glGetUniformLocation(program,"vp"),1,GL_FALSE,vp);
    glActiveTexture(GL_TEXTURE0);glBindTexture(GL_TEXTURE_2D,textures[status]);glUniform1i(glGetUniformLocation(program,"card"),0);
    glBindVertexArray(vao);glDrawArrays(GL_TRIANGLES,0,6);glBindVertexArray(0);glUseProgram(0);
}
