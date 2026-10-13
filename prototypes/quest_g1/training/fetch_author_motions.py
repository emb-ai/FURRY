"""Fetch selected public ZIP entries by HTTP range; verify filename, size and CRC."""
import argparse,concurrent.futures,hashlib,json,struct,time,urllib.request,zlib
from pathlib import Path,PurePosixPath

def fetch_one(row,url,out,total):
 name=row['author_path'];parts=PurePosixPath(name)
 if parts.is_absolute() or '..' in parts.parts or row['author_split']!='train':raise ValueError('Unsafe source path/split')
 entry=row['archive'];dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():
  data=dest.read_bytes()
  if len(data)==entry['size'] and zlib.crc32(data)==entry['crc32']:return {'author_path':name,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
 start=entry['offset'];end=min(total-1,start+entry['compressed_size']+65536)
 for attempt in range(4):
  try:
   with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'}),timeout=60) as r:
    if r.status!=206:raise RuntimeError('Range not honored')
    blob=r.read()
   header=struct.unpack('<IHHHHHIIIHH',blob[:30]);sig,version,flags,method,_,_,crc,cs,us,fn,extra=header
   if sig!=0x04034b50 or flags&1:raise ValueError('Invalid/encrypted ZIP entry')
   filename=blob[30:30+fn].decode('utf-8' if flags&2048 else 'cp437')
   if filename!=entry['name']:raise ValueError('ZIP filename mismatch')
   payload=blob[30+fn+extra:30+fn+extra+entry['compressed_size']]
   if len(payload)!=entry['compressed_size']:raise ValueError('Truncated ZIP range')
   data=zlib.decompress(payload,-15) if method==8 else payload if method==0 else None
   if data is None or len(data)!=entry['size'] or zlib.crc32(data)!=entry['crc32']:raise ValueError('ZIP integrity failure')
   temp=dest.with_suffix('.tmp');temp.write_bytes(data);temp.replace(dest)
   return {'author_path':name,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
  except Exception:
   if attempt==3:raise
   time.sleep(2**attempt)

def main():
 p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True);p.add_argument('--url-file',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();s=json.loads(a.selection.read_text());a.output.mkdir(parents=True,exist_ok=True);url=a.url_file.read_text().strip()
 print(json.dumps({'selected':len(s['clips']),'compressed_bytes':sum(x['archive']['compressed_size'] for x in s['clips'])}),flush=True)
 records=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  futures={pool.submit(fetch_one,row,url,a.output,s['source_archive_bytes']):row for row in s['clips']}
  for f in concurrent.futures.as_completed(futures):
   records.append(f.result())
   if len(records)%32==0:print(json.dumps({'downloaded':len(records),'total':len(futures)}),flush=True)
 (a.output/'download-manifest.json').write_text(json.dumps({'selection_sha256':hashlib.sha256(a.selection.read_bytes()).hexdigest(),'files':sorted(records,key=lambda r:r['author_path'])},indent=2))
 print(json.dumps({'phase':'download_complete','files':len(records),'uncompressed_bytes':sum(x['bytes'] for x in records)}),flush=True)

if __name__=='__main__':main()
