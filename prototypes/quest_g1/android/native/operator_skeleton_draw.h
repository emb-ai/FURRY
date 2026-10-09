#pragma once
#include "operator_skeleton.h"
#include <GLES3/gl32.h>
#include <stdexcept>

namespace operatorskeleton_detail {
// Both first-call resource setup and later overlay passes run inside the
// simulator render pass; restore precisely the state they change.
struct StateGuard {
    GLint program = 0, vao = 0, buffer = 0, depthFunc = 0;
    GLint srcRGB = 0, dstRGB = 0, srcAlpha = 0, dstAlpha = 0, eqRGB = 0, eqAlpha = 0;
    GLboolean depth = false, blend = false, cull = false, depthMask = true;
    StateGuard() {
        glGetIntegerv(GL_CURRENT_PROGRAM, &program);
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &vao);
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &buffer);
        glGetIntegerv(GL_DEPTH_FUNC, &depthFunc);
        glGetIntegerv(GL_BLEND_SRC_RGB, &srcRGB);
        glGetIntegerv(GL_BLEND_DST_RGB, &dstRGB);
        glGetIntegerv(GL_BLEND_SRC_ALPHA, &srcAlpha);
        glGetIntegerv(GL_BLEND_DST_ALPHA, &dstAlpha);
        glGetIntegerv(GL_BLEND_EQUATION_RGB, &eqRGB);
        glGetIntegerv(GL_BLEND_EQUATION_ALPHA, &eqAlpha);
        depth = glIsEnabled(GL_DEPTH_TEST); blend = glIsEnabled(GL_BLEND); cull = glIsEnabled(GL_CULL_FACE);
        glGetBooleanv(GL_DEPTH_WRITEMASK, &depthMask);
    }
    ~StateGuard() {
        glUseProgram(GLuint(program));
        glBindVertexArray(GLuint(vao));
        glBindBuffer(GL_ARRAY_BUFFER, GLuint(buffer));
        glDepthFunc(GLenum(depthFunc)); glDepthMask(depthMask);
        glBlendFuncSeparate(GLenum(srcRGB), GLenum(dstRGB), GLenum(srcAlpha), GLenum(dstAlpha));
        glBlendEquationSeparate(GLenum(eqRGB), GLenum(eqAlpha));
        set(GL_DEPTH_TEST, depth); set(GL_BLEND, blend); set(GL_CULL_FACE, cull);
    }
    static void set(GLenum capability, bool enabled) {
        if (enabled) glEnable(capability); else glDisable(capability);
    }
};
} // namespace operatorskeleton_detail

// Two passes: solid where the bone is in front of the robot, translucent where
// the ego mesh covers the operator. Both use STAGE coordinates.
inline int DrawOperatorSkeleton(const float* vp, const TrackingFrame& raw, double rawAgeMs,
                                const CameraOverlay& optical, double now, double published,
                                bool showMeta = true, bool showCamera = true) {
    OperatorSkeleton mesh = BuildOperatorSkeleton(raw, rawAgeMs, optical, now, published, showMeta, showCamera);
    if (mesh.Triangles() == 0) return 0;
    if (mesh.Triangles() > kOperatorSkeletonMaxTriangles)
        throw std::runtime_error("Operator skeleton exceeds triangle budget");
    operatorskeleton_detail::StateGuard guard;
    static GLuint program = 0, vao = 0, vbo = 0;
    if (!program) {
        auto compile = [](GLenum type, const char* source) {
            GLuint shader = glCreateShader(type);
            glShaderSource(shader, 1, &source, nullptr);
            glCompileShader(shader);
            GLint ok = 0;
            glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
            if (!ok) throw std::runtime_error("Operator skeleton shader failed");
            return shader;
        };
        GLuint vertex = compile(GL_VERTEX_SHADER, R"(#version 320 es
layout(location=0) in vec3 position;
uniform mat4 vp;
void main(){gl_Position=vp*vec4(position,1.0);})");
        GLuint fragment = compile(GL_FRAGMENT_SHADER, R"(#version 320 es
precision mediump float;
uniform vec4 tint;
out vec4 color;
void main(){color=tint;})");
        program = glCreateProgram();
        glAttachShader(program, vertex);
        glAttachShader(program, fragment);
        glLinkProgram(program);
        glDeleteShader(vertex);
        glDeleteShader(fragment);
        GLint ok = 0;
        glGetProgramiv(program, GL_LINK_STATUS, &ok);
        if (!ok) throw std::runtime_error("Operator skeleton shader link failed");
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
    }
    std::vector<float> vertices;
    vertices.reserve(mesh.upper.size() + mesh.legs.size() + mesh.camera.size());
    vertices.insert(vertices.end(), mesh.upper.begin(), mesh.upper.end());
    vertices.insert(vertices.end(), mesh.legs.begin(), mesh.legs.end());
    vertices.insert(vertices.end(), mesh.camera.begin(), mesh.camera.end());
    const int upper = int(mesh.upper.size() / 3), legs = int(mesh.legs.size() / 3), camera = int(mesh.camera.size() / 3);
    glUseProgram(program);
    glUniformMatrix4fv(glGetUniformLocation(program, "vp"), 1, GL_FALSE, vp);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STREAM_DRAW);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    GLint tint = glGetUniformLocation(program, "tint");
    auto draw = [&](int first, int count, float r, float g, float b, float a, bool occluded) {
        if (count <= 0 || a <= 0) return;
        glUniform4f(tint, r, g, b, a);
        glDepthFunc(occluded ? GL_GREATER : GL_LEQUAL);
        glDepthMask(occluded || a < 1.f ? GL_FALSE : GL_TRUE);
        glDrawArrays(GL_TRIANGLES, first, count);
    };
    const float occludedAlpha = 0.5f;
    draw(0, upper, .30f, .85f, 1.f, occludedAlpha, true);
    draw(upper, legs, 1.f, .65f, .25f, occludedAlpha, true);
    draw(upper + legs, camera, 1.f, .25f, .85f, mesh.cameraAlpha * occludedAlpha, true);
    draw(0, upper, .30f, .85f, 1.f, 1.f, false);
    draw(upper, legs, 1.f, .65f, .25f, 1.f, false);
    draw(upper + legs, camera, 1.f, .25f, .85f, mesh.cameraAlpha, false);
    return mesh.Triangles() * 2;
}
