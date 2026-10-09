#pragma once
#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <filesystem>
#include "policy_catalog.h"

namespace episodereplay {
namespace detail {
// Read only the few top-level replay fields. Nested hashes/body metadata cannot
// impersonate a scene field; no additional JSON library is needed by the tools.
class JsonFields {
    const std::string &text;
    size_t pos = 0;
    int depth = 0;
    [[noreturn]] void Fail() const {
        throw std::runtime_error("Invalid episode manifest JSON");
    }
    void Space() {
        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])))
            pos++;
    }
    bool Take(char c) {
        Space();
        if (pos < text.size() && text[pos] == c) {
            pos++;
            return true;
        }
        return false;
    }
    void Expect(char c) {
        if (!Take(c))
            Fail();
    }
    unsigned Hex() {
        unsigned out = 0;
        for (int i = 0; i < 4; i++) {
            if (pos >= text.size())
                Fail();
            char c = text[pos++];
            out <<= 4;
            if (c >= '0' && c <= '9')
                out += c - '0';
            else if (c >= 'a' && c <= 'f')
                out += c - 'a' + 10;
            else if (c >= 'A' && c <= 'F')
                out += c - 'A' + 10;
            else
                Fail();
        }
        return out;
    }
    void Value() {
        Space();
        if (pos >= text.size() || depth >= 64)
            Fail();
        char c = text[pos];
        if (c == '"') {
            String();
            return;
        }
        if (c == '{') {
            Object(false);
            return;
        }
        if (c == '[') {
            pos++;
            depth++;
            if (!Take(']')) {
                do {
                    Value();
                } while (Take(','));
                Expect(']');
            }
            depth--;
            return;
        }
        for (const auto *word : {"true", "false", "null"}) {
            std::string value(word);
            if (text.compare(pos, value.size(), value) == 0) {
                pos += value.size();
                return;
            }
        }
        if (c == '-')
            pos++;
        if (pos >= text.size() || !std::isdigit(static_cast<unsigned char>(text[pos])))
            Fail();
        if (text[pos] == '0')
            pos++;
        else
            while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos])))
                pos++;
        auto digits = [&] {
            size_t start = pos;
            while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos])))
                pos++;
            if (pos == start)
                Fail();
        };
        if (pos < text.size() && text[pos] == '.') {
            pos++;
            digits();
        }
        if (pos < text.size() && (text[pos] == 'e' || text[pos] == 'E')) {
            pos++;
            if (pos < text.size() && (text[pos] == '+' || text[pos] == '-'))
                pos++;
            digits();
        }
    }
    std::map<std::string, std::string> Object(bool collect) {
        Expect('{');
        depth++;
        std::map<std::string, std::string> out;
        if (!Take('}')) {
            do {
                std::string key = String();
                Expect(':');
                Space();
                size_t start = pos;
                Value();
                if (collect && (key == "scene" || key == "nq" || key == "nv" || key == "nu" || key == "policy_file" || key == "policy_sha256")) {
                    if (!out.emplace(key, text.substr(start, pos - start)).second)
                        throw std::runtime_error("Duplicate episode manifest field: " + key);
                }
            } while (Take(','));
            Expect('}');
        }
        depth--;
        return out;
    }

  public:
    explicit JsonFields(const std::string &value) : text(value) {
    }
    std::string String() {
        Expect('"');
        std::string out;
        while (pos < text.size()) {
            unsigned char c = text[pos++];
            if (c == '"')
                return out;
            if (c < 32)
                Fail();
            if (c != '\\') {
                out.push_back(char(c));
                continue;
            }
            if (pos >= text.size())
                Fail();
            char escaped = text[pos++];
            if (escaped == '"' || escaped == '\\' || escaped == '/')
                out.push_back(escaped);
            else if (escaped == 'b')
                out.push_back('\b');
            else if (escaped == 'f')
                out.push_back('\f');
            else if (escaped == 'n')
                out.push_back('\n');
            else if (escaped == 'r')
                out.push_back('\r');
            else if (escaped == 't')
                out.push_back('\t');
            else if (escaped == 'u') {
                unsigned code = Hex();
                if (code >= 0xd800 && code <= 0xdbff) {
                    if (pos + 2 > text.size() || text[pos++] != '\\' || text[pos++] != 'u')
                        Fail();
                    unsigned low = Hex();
                    if (low < 0xdc00 || low > 0xdfff)
                        Fail();
                    code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00);
                } else if (code >= 0xdc00 && code <= 0xdfff)
                    Fail();
                if (code < 0x80)
                    out.push_back(char(code));
                else if (code < 0x800) {
                    out.push_back(char(0xc0 | (code >> 6)));
                    out.push_back(char(0x80 | (code & 63)));
                } else if (code < 0x10000) {
                    out.push_back(char(0xe0 | (code >> 12)));
                    out.push_back(char(0x80 | ((code >> 6) & 63)));
                    out.push_back(char(0x80 | (code & 63)));
                } else {
                    out.push_back(char(0xf0 | (code >> 18)));
                    out.push_back(char(0x80 | ((code >> 12) & 63)));
                    out.push_back(char(0x80 | ((code >> 6) & 63)));
                    out.push_back(char(0x80 | (code & 63)));
                }
            } else
                Fail();
        }
        Fail();
    }
    void End() {
        Space();
        if (pos != text.size())
            Fail();
    }
    std::map<std::string, std::string> Read() {
        auto out = Object(true);
        End();
        return out;
    }
};
inline int Dimension(const std::string &value) {
    if (value.empty())
        throw std::runtime_error("Invalid episode model dimensions");
    int out = 0;
    for (char c : value) {
        if (c < '0' || c > '9' || out > (std::numeric_limits<int>::max() - (c - '0')) / 10)
            throw std::runtime_error("Invalid episode model dimensions");
        out = out * 10 + c - '0';
    }
    return out;
}
} // namespace detail

struct Manifest {
    std::string scene = "lab", raw, policyFile, policySha256;
    std::string PolicyPath(const std::string& assets,const std::string& folder) const {
        if(policyFile.empty())return assets+"/policy.onnx"; // legacy baseline episodes
        std::string path=std::filesystem::exists(folder+"/policy.onnx")?folder+"/policy.onnx":assets+"/"+policyFile;
        questpolicy::VerifySha256(path,policySha256);return path;
    }
    int nq = -1, nv = -1, nu = -1;
    std::string ModelPath(const std::string &assets) const {
        return assets + (scene == "lab" ? "/scene.xml" : "/scene-" + scene + ".xml");
    }
    void ValidateDimensions(int q, int v, int u) const {
        if ((nq >= 0 && nq != q) || (nv >= 0 && nv != v) || (nu >= 0 && nu != u))
            throw std::runtime_error("Episode/model dimensions differ for scene " + scene);
    }
};
inline Manifest LoadManifest(const std::string &folder) {
    std::ifstream input(folder + "/manifest.json");
    if (!input)
        throw std::runtime_error("Cannot read episode manifest.json");
    Manifest out;
    out.raw.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    if (out.raw.size() > 4 * 1024 * 1024)
        throw std::runtime_error("Episode manifest is too large");
    auto fields = detail::JsonFields(out.raw).Read();
    auto scene = fields.find("scene");
    if (scene != fields.end()) {
        detail::JsonFields reader(scene->second);
        out.scene = reader.String();
        reader.End();
    }
    if (out.scene != "lab" && out.scene != "stand" && out.scene != "cup" && out.scene != "push_t")
        throw std::runtime_error("Unknown episode scene: " + out.scene);
    auto readString=[&](const std::string& key){detail::JsonFields reader(fields.at(key));auto value=reader.String();reader.End();return value;};
    if(fields.count("policy_file")!=fields.count("policy_sha256"))throw std::runtime_error("Incomplete episode policy identity");
    if(fields.count("policy_file")){
        out.policyFile=readString("policy_file");out.policySha256=readString("policy_sha256");
        if(!questpolicy::ModelFile(out.policyFile)||out.policySha256.size()!=64)throw std::runtime_error("Invalid episode policy identity");
        for(char c:out.policySha256)if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')))throw std::runtime_error("Invalid episode policy SHA256");
    }
    if (fields.count("nq"))
        out.nq = detail::Dimension(fields.at("nq"));
    if (fields.count("nv"))
        out.nv = detail::Dimension(fields.at("nv"));
    if (fields.count("nu"))
        out.nu = detail::Dimension(fields.at("nu"));
    return out;
}
inline bool IsPauseEvent(const std::string &event) {
    return event == "focus_pause" || event == "menu_open" || event == "user_pause";
}
inline std::vector<int64_t> PauseTimes(const std::string &folder) {
    std::vector<int64_t> times;
    std::ifstream input(folder + "/events.csv");
    std::string line;
    std::getline(input, line);
    while (std::getline(input, line)) {
        size_t first = line.find(','),
               second = first == std::string::npos ? first : line.find(',', first + 1);
        if (second == std::string::npos)
            throw std::runtime_error("Invalid episode event row");
        std::string event = line.substr(second + 1);
        if (!event.empty() && event.back() == '\r')
            event.pop_back();
        if (IsPauseEvent(event)) {
            size_t end = 0;
            int64_t time = std::stoll(line.substr(0, first), &end);
            if (end != first || time < 0 || (!times.empty() && time < times.back()))
                throw std::runtime_error("Invalid episode pause time");
            times.push_back(time);
        }
    }
    return times;
}
} // namespace episodereplay
