#pragma once
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "file_sha256.h"

namespace questpolicy {
struct Entry { std::string id, file, sha256, title, note; };
inline bool Token(const std::string& value) {
    if(value.empty() || value.size()>80)return false;
    for(unsigned char c:value)if(!((c>='a'&&c<='z')||(c>='A'&&c<='Z')||
        (c>='0'&&c<='9')||c=='-'||c=='_'))return false;
    return true;
}
inline bool ModelFile(const std::string& value) {
    return value.size()>5 && value.substr(value.size()-5)==".onnx" && Token(value.substr(0,value.size()-5));
}
inline std::string Metadata(const std::string& scene,const Entry& entry) {
    return "recording_metadata-"+scene+"-policy-"+entry.id+".json";
}
inline std::string VerifiedPath(const std::string& assets,const Entry& entry) {
    std::string path=assets+"/"+entry.file;VerifySha256(path,entry.sha256);return path;
}
inline std::vector<Entry> Load(const std::string& assets) {
    std::ifstream input(assets+"/policy_catalog.txt");
    if(!input)throw std::runtime_error("Missing policy catalog");
    std::vector<Entry> out;std::set<std::string> ids,files;std::string line;
    while(std::getline(input,line)){
        Entry e;std::istringstream row(line);
        if(!(row>>std::quoted(e.id)>>std::quoted(e.file)>>std::quoted(e.sha256)>>std::quoted(e.title)>>std::quoted(e.note)))
            throw std::runtime_error("Invalid policy catalog row");
        row>>std::ws;
        if(!row.eof()||!Token(e.id)||!ModelFile(e.file)||e.sha256.size()!=64||e.title.empty()||
           e.title.size()>100||e.note.size()>140||!ids.insert(e.id).second||!files.insert(e.file).second||out.size()>=128)
            throw std::runtime_error("Invalid policy catalog entry");
        for(char c:e.sha256)if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')))throw std::runtime_error("Invalid policy digest");
        out.push_back(e);
    }
    if(out.empty()||out[0].id!="twist2-20k"||out[0].file!="policy.onnx")throw std::runtime_error("Missing baseline policy");
    return out;
}
} // namespace questpolicy
