"""Convert authorized cup HDF5 to two disjoint LeRobot v2 datasets."""
import argparse
import json
import subprocess
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

TASK = "Pick up the cup and place it at the target."
VIDEO = "observation.images.ego"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2))


def convert(path, destination):
    destination.mkdir(parents=True, exist_ok=False)
    with h5py.File(path, "r") as h:
        splits = {s:[x.decode() for x in h[f"mask/{s}"][:]] for s in ["train", "valid"]}
        assert not set(splits["train"]) & set(splits["valid"])
        for split, names in splits.items():
            root = destination / split
            meta = root / "meta"; meta.mkdir(parents=True)
            data = root / "data/chunk-000"; data.mkdir(parents=True)
            videos = root / f"videos/chunk-000/{VIDEO}"; videos.mkdir(parents=True)
            episodes = []; offset = 0
            for i,name in enumerate(names):
                d = h[f"data/{name}"]; n = len(d["actions"])
                actions, state = d["actions"][:], d["obs/previous_command"][:]
                assert actions.shape == state.shape == (n,39)
                assert np.isfinite(actions).all() and np.isfinite(state).all()
                columns = {"action":list(actions),"observation.state":list(state),"timestamp":np.arange(n,dtype=np.float32)/30,"frame_index":np.arange(n,dtype=np.int64),"episode_index":np.full(n,i,dtype=np.int64),"index":np.arange(offset,offset+n,dtype=np.int64),"task_index":np.zeros(n,dtype=np.int64),"annotation.human.task_description":np.zeros(n,dtype=np.int64),"next.reward":np.zeros(n,dtype=np.float32),"next.done":np.arange(n)==n-1}
                parquet = data/f"episode_{i:06d}.parquet"
                pd.DataFrame(columns).to_parquet(parquet,index=False)
                movie = videos/f"episode_{i:06d}.mp4"
                images = d["obs/agentview_image"]
                height,width = images.shape[1:3]
                process = subprocess.Popen(["ffmpeg","-v","error","-nostdin","-f","rawvideo","-pix_fmt","rgb24","-s",f"{width}x{height}","-r","30","-i","pipe:0","-an","-c:v","libx264","-crf","18","-threads","2","-pix_fmt","yuv420p",str(movie)],stdin=subprocess.PIPE)
                for start in range(0,n,64): process.stdin.write(images[start:start+64].tobytes())
                process.stdin.close()
                if process.wait() != 0: raise RuntimeError(f"ffmpeg failed: {name}")
                count = subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries","stream=nb_read_frames","-of","csv=p=0",str(movie)],text=True).strip()
                assert int(count)==n
                read = pd.read_parquet(parquet)
                assert np.array_equal(np.stack(read["action"]),actions)
                assert np.array_equal(np.stack(read["observation.state"]),state)
                episodes.append({"episode_index":i,"tasks":[TASK],"length":n,"source_episode":name})
                offset += n
                print(json.dumps({"split":split,"episode":i,"frames":n,"source":name}),flush=True)
            features = {k:{"dtype":"int64","shape":[1],"names":None} for k in ["frame_index","episode_index","index","task_index","annotation.human.task_description"]}
            features.update({"timestamp":{"dtype":"float32","shape":[1],"names":None},"action":{"dtype":"float32","shape":[39],"names":[f"command_{i}" for i in range(39)]},"observation.state":{"dtype":"float32","shape":[39],"names":[f"previous_command_{i}" for i in range(39)]},VIDEO:{"dtype":"video","shape":[height,width,3],"names":["height","width","channels"],"info":{"video.height":height,"video.width":width,"video.codec":"h264","video.pix_fmt":"yuv420p","video.is_depth_map":False,"video.fps":30,"video.channels":3,"has_audio":False}}})
            write_json(meta/"info.json",{"codebase_version":"v2.1","robot_type":"furry_g1_homie_commands","total_episodes":len(names),"total_frames":offset,"total_tasks":1,"total_chunks":1,"total_videos":len(names),"chunks_size":1000,"fps":30,"splits":{"train":f"0:{len(names)}"},"data_path":"data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet","video_path":"videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4","features":features})
            write_json(meta/"modality.json",{"state":{"commands":{"start":0,"end":39}},"action":{"commands":{"start":0,"end":39}},"video":{"ego":{"original_key":VIDEO}},"annotation":{"human.task_description":{"original_key":"task_index"}}})
            (meta/"tasks.jsonl").write_text(json.dumps({"task_index":0,"task":TASK})+"\n")
            (meta/"episodes.jsonl").write_text("".join(json.dumps(x)+"\n" for x in episodes))
            write_json(meta/"furry_provenance.json",{"source":str(path),"original_split":split,"state_semantics":"previous HOMIE command, not measured robot state","video_reencoded":True,"action_and_state_roundtrip_exact":True,"video_frame_counts_verified":True})


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--data",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(); convert(args.data,args.out)
