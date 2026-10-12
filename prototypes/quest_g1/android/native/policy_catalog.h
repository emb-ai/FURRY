#pragma once
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "file_sha256.h"
#include "policy_backend.h"

namespace questpolicy {
struct Entry {
    std::string id, file, sha256, title, note;
    std::string backend=Twist2, secondaryFile{}, secondarySha256{}, plannerFile{}, plannerSha256{}, inputPatch{};
};
inline bool Digest(const std::string& value){
    if(value.size()!=64)return false;
    for(char c:value)if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')))return false;
    return true;
}
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
inline RuntimePolicy VerifiedPolicy(const std::string& assets,const Entry& entry) {
    RuntimePolicy out{entry.backend,VerifiedPath(assets,entry),""};
    if(!entry.secondaryFile.empty()){
        out.secondaryPath=assets+"/"+entry.secondaryFile;VerifySha256(out.secondaryPath,entry.secondarySha256);
    }
    if(!entry.plannerFile.empty()){
        out.plannerPath=assets+"/"+entry.plannerFile;VerifySha256(out.plannerPath,entry.plannerSha256);
#ifdef __ANDROID__
        out.asyncPlanner=true;
#endif
    }
    out.Validate();return out;
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
        if(!row.eof()){
            if(!(row>>std::quoted(e.backend)>>std::quoted(e.secondaryFile)>>std::quoted(e.secondarySha256)))
                throw std::runtime_error("Invalid policy backend row");
            row>>std::ws;
        }
        if(e.backend==Sonic){
            if(!(row>>std::quoted(e.plannerFile)>>std::quoted(e.plannerSha256))||!ModelFile(e.plannerFile)||!Digest(e.plannerSha256)||!files.insert(e.plannerFile).second)
                throw std::runtime_error("Invalid SONIC planner identity");
            row>>std::ws;
        }
        if(!row.eof()){
            if(!(row>>std::quoted(e.inputPatch))||e.inputPatch!="virtual_leg_lift10_v1"||e.backend!=Twist2)throw std::runtime_error("Invalid source patch mode");
            row>>std::ws;
        }
        if(!KnownBackend(e.backend) ||
           (e.backend!=Twist2 ? (!ModelFile(e.secondaryFile)||!Digest(e.secondarySha256)||!files.insert(e.secondaryFile).second)
                                     : (!e.secondaryFile.empty()||!e.secondarySha256.empty())))
            throw std::runtime_error("Invalid policy backend identity");
        if(!row.eof()||!Token(e.id)||!ModelFile(e.file)||!Digest(e.sha256)||e.title.empty()||
           e.title.size()>100||e.note.size()>140||!ids.insert(e.id).second||!files.insert(e.file).second||out.size()>=128)
            throw std::runtime_error("Invalid policy catalog entry");
        out.push_back(e);
    }
    if(out.empty()||out[0].id!="twist2-20k"||out[0].file!="policy.onnx"||out[0].backend!=Twist2)throw std::runtime_error("Missing baseline policy");
    return out;
}
} // namespace questpolicy
