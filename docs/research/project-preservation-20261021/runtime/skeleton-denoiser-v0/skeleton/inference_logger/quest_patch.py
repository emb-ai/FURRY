"""Generate a reviewed, version-guarded Quest patch without editing its checkout."""
import difflib
from pathlib import Path


def patched_source(source):
    if '// inference_logger protocol v1' in source:
        raise ValueError('Source is already patched')
    clock_anchor = 'namespace {\n'
    probe_anchor = '   double now=CameraEpochMs();\n'
    submit_anchor = ' Json::Value v;v["t"]=CameraEpochMs();v["hmd"]=Pose(f.head);\n'
    for anchor in (clock_anchor, probe_anchor, submit_anchor):
        if source.count(anchor) != 1:
            raise ValueError('Unsupported camera_stream.cpp revision; review the protocol manually')
    source = source.replace(clock_anchor, clock_anchor + '''// inference_logger protocol v1
int64_t LoggerMonotonicNs(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
''')
    source = source.replace(probe_anchor, '''   const int64_t loggerReceived=LoggerMonotonicNs();
   try{
    Json::CharReaderBuilder builder;Json::Value probe;std::string errors;
    std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    if(m->str.size()<2048 && reader->parse(m->str.data(),m->str.data()+m->str.size(),&probe,&errors)
       && probe.get("type", "").asString()=="sync_probe"){
     Json::Value reply;reply["type"]="sync_reply";
     reply["request_id"]=probe["request_id"];reply["t1_ns"]=probe["t1_ns"];
     reply["t2_ns"]=std::to_string(loggerReceived);reply["t3_ns"]=std::to_string(LoggerMonotonicNs());
     Json::StreamWriterBuilder writer;writer["indentation"]="";
     socket->send(Json::writeString(writer,reply));return;
    }
   }catch(...){return;}
''' + probe_anchor)
    source = source.replace(submit_anchor, submit_anchor + ''' v["schema_version"]=1;
 v["input_sequence"]=std::to_string(f.sequence);
 v["quest_monotonic_ns"]=std::to_string(LoggerMonotonicNs());
 v["xr_time_ns"]=std::to_string(f.xr_time_ns);
 v["body_time_ns"]=std::to_string(f.body.time_ns);
 v["time_semantics"]="tracking_submit; xr_time_ns is predicted display, body_time_ns is runtime sample";
 v["valid"]=f.valid;v["body"]["valid"]=f.body.valid;
 v["body"]["confidence"]=f.body.confidence;
 v["body"]["skeleton_version"]=f.body.skeleton_version;
 v["body"]["lower_body_source"]="runtime_estimated";
 for(int j=0;j<14;j++){
  Json::Value joint=Pose(f.body.joints[j]);joint["flags"]=std::to_string(f.body.flags[j]);
  v["body"]["joints"].append(joint);
 }
 for(auto flag:f.location_flags)v["location_flags"].append(std::to_string(flag));
''')
    return source


def generate(source, output):
    source, output = Path(source), Path(output)
    before = source.read_text(encoding='utf-8')
    after = patched_source(before)
    diff = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
        fromfile='a/android/native/camera_stream.cpp', tofile='b/android/native/camera_stream.cpp'))
    with output.open('x', encoding='utf-8', newline='\n') as f:
        f.write(diff)
    return output
