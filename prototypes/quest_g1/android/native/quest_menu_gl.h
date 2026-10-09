#pragma once

#include "quest_menu.h"
#include <GLES3/gl32.h>
#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace questmenu {
namespace glmenu {

using Color = std::array<float, 4>;
inline constexpr Color Background {.13f, .14f, .12f, 1.f};
inline constexpr Color Control {.18f, .19f, .16f, 1.f};
inline constexpr Color Selected {.29f, .31f, .26f, 1.f};
inline constexpr Color Text {.92f, .93f, .89f, 1.f};
inline constexpr Color Muted {.68f, .71f, .64f, 1.f};
inline constexpr Color Border {.31f, .33f, .28f, 1.f};
inline constexpr Color Warning {.96f, .63f, .60f, 1.f};

// Menu drawing is embedded in the simulator's render pass. Preserve the state
// changed by both resource initialization and rendering, including unit zero.
struct StateGuard {
    GLint program = 0, vao = 0, buffer = 0, activeTexture = 0;
    GLint activeBinding = 0, zeroBinding = 0, unpack = 0, unpackBuffer = 0;
    GLint unpackRowLength = 0, unpackSkipRows = 0, unpackSkipPixels = 0;
    GLint srcRGB = 0, dstRGB = 0, srcAlpha = 0, dstAlpha = 0, eqRGB = 0, eqAlpha = 0;
    GLboolean depth = false, blend = false, cull = false, scissor = false, depthMask = true;
    StateGuard() {
        glGetIntegerv(GL_CURRENT_PROGRAM, &program);
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &vao);
        glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &buffer);
        glGetIntegerv(GL_ACTIVE_TEXTURE, &activeTexture);
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &activeBinding);
        glActiveTexture(GL_TEXTURE0);
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &zeroBinding);
        glGetIntegerv(GL_UNPACK_ALIGNMENT, &unpack);
        glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &unpackBuffer);
        glGetIntegerv(GL_UNPACK_ROW_LENGTH, &unpackRowLength);
        glGetIntegerv(GL_UNPACK_SKIP_ROWS, &unpackSkipRows);
        glGetIntegerv(GL_UNPACK_SKIP_PIXELS, &unpackSkipPixels);
        glGetIntegerv(GL_BLEND_SRC_RGB, &srcRGB);
        glGetIntegerv(GL_BLEND_DST_RGB, &dstRGB);
        glGetIntegerv(GL_BLEND_SRC_ALPHA, &srcAlpha);
        glGetIntegerv(GL_BLEND_DST_ALPHA, &dstAlpha);
        glGetIntegerv(GL_BLEND_EQUATION_RGB, &eqRGB);
        glGetIntegerv(GL_BLEND_EQUATION_ALPHA, &eqAlpha);
        depth = glIsEnabled(GL_DEPTH_TEST); blend = glIsEnabled(GL_BLEND);
        cull = glIsEnabled(GL_CULL_FACE); scissor = glIsEnabled(GL_SCISSOR_TEST);
        glGetBooleanv(GL_DEPTH_WRITEMASK, &depthMask);
    }
    ~StateGuard() {
        glUseProgram(GLuint(program));
        glBindVertexArray(GLuint(vao));
        glBindBuffer(GL_ARRAY_BUFFER, GLuint(buffer));
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, GLuint(zeroBinding));
        glActiveTexture(GLenum(activeTexture)); glBindTexture(GL_TEXTURE_2D, GLuint(activeBinding));
        glPixelStorei(GL_UNPACK_ALIGNMENT, unpack);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, GLuint(unpackBuffer));
        glPixelStorei(GL_UNPACK_ROW_LENGTH, unpackRowLength);
        glPixelStorei(GL_UNPACK_SKIP_ROWS, unpackSkipRows);
        glPixelStorei(GL_UNPACK_SKIP_PIXELS, unpackSkipPixels);
        glBlendFuncSeparate(GLenum(srcRGB), GLenum(dstRGB), GLenum(srcAlpha), GLenum(dstAlpha));
        glBlendEquationSeparate(GLenum(eqRGB), GLenum(eqAlpha));
        set(GL_DEPTH_TEST, depth); set(GL_BLEND, blend); set(GL_CULL_FACE, cull); set(GL_SCISSOR_TEST, scissor);
        glDepthMask(depthMask);
    }
    static void set(GLenum capability, bool enabled) {
        if (enabled) glEnable(capability); else glDisable(capability);
    }
};

struct Vertex { float x, y, u, v, r, g, b, a; };

inline unsigned NextCodepoint(const std::string& s, size_t& i) {
    unsigned c = static_cast<unsigned char>(s[i++]);
    if (c < 128) return c;
    unsigned count = (c & 0xe0) == 0xc0 ? 1 : (c & 0xf0) == 0xe0 ? 2 : (c & 0xf8) == 0xf0 ? 3 : 0;
    if (!count || i + count > s.size()) return '?';
    c &= (1u << (6 - count)) - 1;
    for (unsigned k = 0; k < count; ++k) {
        unsigned next = static_cast<unsigned char>(s[i]);
        if ((next & 0xc0) != 0x80) return '?';
        ++i; c = (c << 6) | (next & 63);
    }
    return c;
}
inline int Cell(unsigned c) {
    if (c >= 32 && c <= 126) return int(c) - 32;
    if (c >= 1040 && c <= 1103) return 96 + int(c) - 1040;
    if (c == 1025) return 160;
    if (c == 1105) return 161;
    // Common punctuation in status strings has a legible ASCII fallback.
    if (c == 0x2013 || c == 0x2014) return '-' - 32;
    if (c == 0x00b7) return ':' - 32;
    return '?' - 32;
}

class Renderer {
    GLuint program = 0, texture = 0, vao = 0, vbo = 0;
    GLint matrixUniform = -1;
    std::array<uint8_t, 162> widths {};
public:
    std::vector<Vertex> vertices;
    void Initialize(const std::string& assets) {
        if (program) return;
        // Verify the assets before creating GL objects: a missing atlas should
        // leave initialization retryable and the enclosing render state intact.
        std::vector<uint8_t> pixels(512 * 440);
        std::ifstream atlas(assets + "/menu_font.bin", std::ios::binary);
        atlas.read(reinterpret_cast<char*>(pixels.data()), pixels.size());
        if (!atlas) throw std::runtime_error("Menu font missing");
        std::ifstream advances(assets + "/menu_font_widths.bin", std::ios::binary);
        advances.read(reinterpret_cast<char*>(widths.data()), widths.size());
        if (!advances) throw std::runtime_error("Menu font widths missing");
        auto shader = [](GLenum type, const char* source) {
            GLuint s = glCreateShader(type);
            glShaderSource(s, 1, &source, nullptr); glCompileShader(s);
            GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
            if (!ok) { glDeleteShader(s); throw std::runtime_error("Menu shader failed"); }
            return s;
        };
        GLuint vertex = shader(GL_VERTEX_SHADER, R"(#version 320 es
layout(location=0) in vec2 p;layout(location=1) in vec2 uv;layout(location=2) in vec4 tint;
uniform mat4 matrix;out vec2 tex;out vec4 rgba;
void main(){gl_Position=matrix*vec4(p,0,1);tex=uv;rgba=tint;})");
        GLuint fragment = shader(GL_FRAGMENT_SHADER, R"(#version 320 es
precision mediump float;in vec2 tex;in vec4 rgba;uniform sampler2D atlas;out vec4 color;
void main(){color=vec4(rgba.rgb,rgba.a*texture(atlas,tex).r);})");
        GLuint linked = glCreateProgram();
        glAttachShader(linked, vertex); glAttachShader(linked, fragment); glLinkProgram(linked);
        glDeleteShader(vertex); glDeleteShader(fragment);
        GLint ok = 0; glGetProgramiv(linked, GL_LINK_STATUS, &ok);
        if (!ok) { glDeleteProgram(linked); throw std::runtime_error("Menu shader link failed"); }
        program = linked;
        matrixUniform = glGetUniformLocation(program, "matrix");
        glGenTextures(1, &texture); glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);
        glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, 512, 440, 0, GL_RED, GL_UNSIGNED_BYTE, pixels.data());
        glGenVertexArrays(1, &vao); glBindVertexArray(vao);
        glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), nullptr);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(2 * sizeof(float)));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(4 * sizeof(float)));
    }
    void Triangle(float ax, float ay, float bx, float by, float cx, float cy, const Color& c) {
        // Cell 95 (DEL) is an opaque solid in the shared 16-column atlas.
        constexpr float u = 15.5f / 16, v = 5.5f / 11;
        vertices.push_back({ax, ay, u, v, c[0], c[1], c[2], c[3]});
        vertices.push_back({bx, by, u, v, c[0], c[1], c[2], c[3]});
        vertices.push_back({cx, cy, u, v, c[0], c[1], c[2], c[3]});
    }
    void Quad(Rect r, const Color& c) {
        Triangle(r.x, r.y, r.x + r.w, r.y, r.x + r.w, r.y + r.h, c);
        Triangle(r.x, r.y, r.x + r.w, r.y + r.h, r.x, r.y + r.h, c);
    }
    void RoundRect(Rect r, float radius, const Color& c) {
        radius = std::min(radius, std::min(r.w, r.h) / 2);
        Quad({r.x + radius, r.y, r.w - 2 * radius, r.h}, c);
        Quad({r.x, r.y + radius, radius, r.h - 2 * radius}, c);
        Quad({r.x + r.w - radius, r.y + radius, radius, r.h - 2 * radius}, c);
        const float centres[4][2] = {{r.x + r.w - radius, r.y + r.h - radius},
            {r.x + radius, r.y + r.h - radius}, {r.x + radius, r.y + radius},
            {r.x + r.w - radius, r.y + radius}};
        for (int corner = 0; corner < 4; ++corner) for (int j = 0; j < 4; ++j) {
            float a = (corner + j / 4.f) * 1.570796327f;
            float b = (corner + (j + 1) / 4.f) * 1.570796327f;
            float x = centres[corner][0], y = centres[corner][1];
            Triangle(x, y, x + radius * std::cos(a), y + radius * std::sin(a),
                     x + radius * std::cos(b), y + radius * std::sin(b), c);
        }
    }
    void Circle(float x, float y, float radius, const Color& c) {
        for (int j = 0; j < 20; ++j) {
            float a = j * 6.283185307f / 20, b = (j + 1) * 6.283185307f / 20;
            Triangle(x, y, x + radius * std::cos(a), y + radius * std::sin(a),
                     x + radius * std::cos(b), y + radius * std::sin(b), c);
        }
    }
    void Glyph(float x, float top, float height, int cell, const Color& c) {
        const float u = (cell % 16) * 32 / 512.f, v = (cell / 16) * 40 / 440.f;
        float w = height * 32 / 40;
        const Vertex q[] = {
            {x, top, u, v, c[0], c[1], c[2], c[3]},
            {x + w, top - height, u + 32 / 512.f, v + 40 / 440.f, c[0], c[1], c[2], c[3]},
            {x + w, top, u + 32 / 512.f, v, c[0], c[1], c[2], c[3]},
            {x, top, u, v, c[0], c[1], c[2], c[3]},
            {x, top - height, u, v + 40 / 440.f, c[0], c[1], c[2], c[3]},
            {x + w, top - height, u + 32 / 512.f, v + 40 / 440.f, c[0], c[1], c[2], c[3]}};
        vertices.insert(vertices.end(), std::begin(q), std::end(q));
    }
    void Label(Rect r, const std::string& value, Color c = Text, float height = .034f,
               bool centre = false, float padding = 0) {
        if (value.empty()) return;
        std::vector<int> cells;
        float pixelWidth = 0;
        for (size_t i = 0; i < value.size();) {
            int cell = Cell(NextCodepoint(value, i));
            cells.push_back(cell); pixelWidth += widths[cell];
        }
        if (pixelWidth <= 0) return;
        const float fit = std::min(1.f, std::max(.001f, r.w - 2 * padding) / (pixelWidth * height / 40));
        // Keep label height consistent; fit only the horizontal advance and
        // glyph extent for long camera/export notices in their bounded row.
        float advanceScale = height / 40 * fit;
        float x = r.x + padding + (centre ? (r.w - 2 * padding - pixelWidth * advanceScale) / 2 : 0);
        const float top = r.y + (r.h + height) / 2 + .002f;
        for (int cell : cells) {
            // Atlas cells have two pixels of left padding. Crop the quad's
            // horizontal scale with the same fit as the proportional advance.
            const size_t start = vertices.size();
            Glyph(x, top, height, cell, c);
            for (size_t v = start; v < vertices.size(); ++v) vertices[v].x = x + (vertices[v].x - x) * fit;
            x += widths[cell] * advanceScale;
        }
    }
    int Draw(const float* matrix) {
        if (vertices.empty()) return 0;
        glDisable(GL_DEPTH_TEST); glDisable(GL_CULL_FACE); glDisable(GL_SCISSOR_TEST);
        glDepthMask(GL_FALSE); glEnable(GL_BLEND);
        glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        glUseProgram(program); glUniformMatrix4fv(matrixUniform, 1, GL_FALSE, matrix);
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, texture);
        glUniform1i(glGetUniformLocation(program, "atlas"), 0);
        glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_STREAM_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, GLint(vertices.size()));
        return int(vertices.size() / 3);
    }
};
inline Renderer& SharedRenderer() { static Renderer renderer; return renderer; }
inline void Multiply(const float* a, const float* b, float* out) {
    for (int c = 0; c < 4; ++c) for (int r = 0; r < 4; ++r) {
        out[4 * c + r] = 0;
        for (int k = 0; k < 4; ++k) out[4 * c + r] += a[4 * k + r] * b[4 * c + k];
    }
}

} // namespace glmenu

inline int DrawQuestMenu(const float* vp, const float* menuWorld, const std::string& assetsPath,
                         const State& state, const std::vector<Pointer>& pointers) {
    if (!state.open) return 0;
    glmenu::StateGuard guard;
    auto& draw = glmenu::SharedRenderer(); draw.Initialize(assetsPath); draw.vertices.clear();
    draw.RoundRect(PanelRect, .005f, glmenu::Border);
    draw.RoundRect({PanelRect.x + .001f, PanelRect.y + .001f, PanelRect.w - .002f, PanelRect.h - .002f},
                   .004f, glmenu::Background);
    auto layout = BuildLayout(state);
    for (const auto& w : layout) {
        bool hovered = false;
        for (const auto& p : pointers)
            if (p.valid && w.enabled && w.action != Action::None && w.rect.Contains(p.x, p.y)) hovered = true;
        glmenu::Color text = w.kind == Kind::MutedText ? glmenu::Muted : glmenu::Text;
        if (!w.enabled) { text[0] *= .52f; text[1] *= .52f; text[2] *= .52f; }
        if (w.kind == Kind::Divider) { draw.Quad(w.rect, glmenu::Border); continue; }
        if (w.kind == Kind::Tab) {
            if (hovered) draw.Quad(w.rect, glmenu::Control);
            if (w.selected) draw.Quad({w.rect.x, w.rect.y, w.rect.w, .003f}, glmenu::Text);
            draw.Label(w.rect, w.text, w.selected ? glmenu::Text : glmenu::Muted);
        } else if (w.kind == Kind::Button || w.kind == Kind::Primary) {
            draw.RoundRect(w.rect, .003f, glmenu::Border);
            auto fill = w.selected || hovered ? glmenu::Selected : glmenu::Control;
            if (w.kind == Kind::Primary && w.enabled) { fill = glmenu::Text; text = glmenu::Background; }
            draw.RoundRect({w.rect.x + .001f, w.rect.y + .001f, w.rect.w - .002f, w.rect.h - .002f}, .002f, fill);
            draw.Label(w.rect, w.text, text, .032f, true, .008f);
        } else if (w.kind == Kind::Radio) {
            if (w.selected || hovered) draw.Quad(w.rect, w.selected ? glmenu::Selected : glmenu::Control);
            float cx = w.rect.x + .024f, cy = w.rect.y + w.rect.h / 2;
            draw.Circle(cx, cy, .009f, w.enabled ? glmenu::Muted : glmenu::Border);
            draw.Circle(cx, cy, .007f, w.selected || hovered ? glmenu::Selected : glmenu::Background);
            if (w.selected) draw.Circle(cx, cy, .004f, text);
            draw.Label({w.rect.x + .045f, w.rect.y, w.rect.w - .055f, w.rect.h}, w.text, text);
        } else if (w.kind == Kind::Checkbox) {
            if (hovered) draw.Quad(w.rect, glmenu::Control);
            const Rect box {w.rect.x + w.rect.w - .028f, w.rect.y + (w.rect.h - .022f) / 2, .022f, .022f};
            draw.Quad(box, w.enabled ? glmenu::Muted : glmenu::Border);
            draw.Quad({box.x + .0015f, box.y + .0015f, box.w - .003f, box.h - .003f}, glmenu::Background);
            if (w.selected) draw.Quad({box.x + .005f, box.y + .005f, box.w - .01f, box.h - .01f}, text);
            draw.Label({w.rect.x, w.rect.y, w.rect.w - .045f, w.rect.h}, w.text, text);
        } else {
            float height = w.rect.h < .04f ? .028f : .033f;
            draw.Label(w.rect, w.text, text, height);
        }
    }
    for (const auto& p : pointers) if (p.valid && PanelRect.Contains(p.x, p.y)) {
        const bool actionable = HitTest(layout, p.x, p.y) != Action::None;
        draw.Circle(p.x, p.y, .006f, glmenu::Background);
        draw.Circle(p.x, p.y, .004f, actionable ? glmenu::Text : glmenu::Muted);
    }
    float matrix[16]; glmenu::Multiply(vp, menuWorld, matrix);
    return draw.Draw(matrix);
}

inline int DrawMinimalStatus(const float* headVP, const std::string& assetsPath, const State& state,
                             int trackingStatus, int recordingStatus) {
    // Operational status stays visible when every optional debug overlay is off.
    glmenu::StateGuard guard;
    auto& draw = glmenu::SharedRenderer(); draw.Initialize(assetsPath); draw.vertices.clear();
    std::string status = StatusText(state);
    if (recordingStatus == 2) status = "Ошибка записи. Исходники сохранены";
    else if (state.faulted || trackingStatus == 4) status = "Ошибка симуляции. Откройте меню";
    // The runtime fills this boolean from the current XR frame. The worker's
    // trackingStatus can remain stale while the menu has paused physics.
    else if (!state.trackingValid) status = "Нет свежего трекинга";
    const bool warning = state.faulted || recordingStatus == 2 || trackingStatus == 4 || !state.trackingValid;
    const float bottom = -.47f;
    draw.RoundRect({-.37f, bottom, .74f, .125f}, .004f, glmenu::Background);
    std::string title = state.mode == Mode::Trajectories ? "Сбор траекторий" : "Симуляция";
    if (state.recording) title = state.paused ? "REC - пауза" : "REC";
    draw.Label({-.35f, bottom + .070f, .22f, .036f}, title, state.recording ? glmenu::Warning : glmenu::Text, .029f);
    draw.Label({-.115f, bottom + .070f, .465f, .036f}, status, warning ? glmenu::Warning : glmenu::Muted, .029f);
    std::string detail = state.recordingInfo;
    if (detail.empty()) detail = HumanCapture(state) ?
        (state.plan == Plan::Train ? "Движения человека - обучение" : "Движения человека - тест") : SceneTitle(state.scene);
    if(!HumanCapture(state)&&state.policyIndex>=0&&state.policyIndex<int(state.policies.size()))detail += " / " + state.policies[state.policyIndex].title;
    draw.Label({-.35f, bottom + .037f, .70f, .034f}, detail, glmenu::Muted, .027f);
    const char* exportText = state.exportStatus == 1 ? "Архив копируется" :
        state.exportStatus == 2 ? "Архив сохранён в Download/G1Quest" :
        state.exportStatus == 3 ? "Ошибка экспорта. Исходники в очках" : "Menu: меню     B: пауза";
    draw.Label({-.35f, bottom + .005f, .70f, .034f}, exportText,
               state.exportStatus == 3 ? glmenu::Warning : glmenu::Muted, .027f);
    const float local[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,-1.3f,1};
    float matrix[16]; glmenu::Multiply(headVP, local, matrix);
    return draw.Draw(matrix);
}

} // namespace questmenu
