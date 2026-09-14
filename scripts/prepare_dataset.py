#!/usr/bin/env python3
import argparse, json, random
from pathlib import Path
import soundfile as sf
from tqdm import tqdm
AUDIO_EXTENSIONS={'.wav','.flac','.ogg'}
def info(path):
    x=sf.info(str(path)); return x.frames/float(x.samplerate),x.samplerate,x.frames,x.channels
def norm_text(s,uppercase=True):
    s=' '.join(s.strip().split()); return s.upper() if uppercase else s
def write_jsonl(items,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for x in items: f.write(json.dumps(x,ensure_ascii=False)+'\n')
def parse_trans(path,uppercase=True):
    out={}
    for line in path.read_text(encoding='utf-8').splitlines():
        line=line.strip()
        if not line: continue
        p=line.split(maxsplit=1); out[p[0]]=norm_text(p[1] if len(p)==2 else '',uppercase)
    return out
def prepare_librispeech(args):
    records=[]
    for tr in tqdm(sorted(args.input_dir.rglob('*.trans.txt'))):
        for uid,text in parse_trans(tr,not args.keep_case).items():
            audio=tr.parent/f'{uid}.flac'
            if not audio.exists(): continue
            dur,sr,ns,ch=info(audio)
            if dur<args.min_duration or (args.max_duration>0 and dur>args.max_duration): continue
            p=uid.split('-')
            records.append({'id':uid,'audio':str(audio.resolve()),'text':text,'duration':dur,'sample_rate':sr,'num_samples':ns,'channels':ch,'speaker':p[0],'chapter':p[1] if len(p)>1 else ''})
    write_jsonl(records,args.output); return records
def prepare_generic(args):
    records=[]
    audio_files=[p for p in args.input_dir.rglob('*') if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS]
    for audio in tqdm(sorted(audio_files)):
        txt=audio.with_suffix('.txt')
        if not txt.exists(): continue
        text=norm_text(txt.read_text(encoding='utf-8'),not args.keep_case)
        if not text: continue
        dur,sr,ns,ch=info(audio)
        if dur<args.min_duration or (args.max_duration>0 and dur>args.max_duration): continue
        records.append({'id':audio.stem,'audio':str(audio.resolve()),'text':text,'duration':dur,'sample_rate':sr,'num_samples':ns,'channels':ch})
    write_jsonl(records,args.output); return records
def stats(records):
    if not records: print('No records.'); return
    d=[x['duration'] for x in records]; srs={}
    for x in records: srs[x['sample_rate']]=srs.get(x['sample_rate'],0)+1
    print(f'utterances={len(records)}'); print(f'hours={sum(d)/3600:.2f}'); print(f'duration min/avg/max={min(d):.2f}/{sum(d)/len(d):.2f}/{max(d):.2f}s'); print(f'sample_rates={srs}')
def split_manifest(args):
    items=[json.loads(x) for x in args.manifest.read_text(encoding='utf-8').splitlines() if x.strip()]
    random.Random(args.seed).shuffle(items); n=len(items); nt=int(n*args.test_ratio); nv=int(n*args.valid_ratio)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    write_jsonl(items[nt+nv:],args.output_dir/'train.jsonl'); write_jsonl(items[nt:nt+nv],args.output_dir/'valid.jsonl'); write_jsonl(items[:nt],args.output_dir/'test.jsonl')
def common(p):
    p.add_argument('--input-dir',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--min-duration',type=float,default=0.2); p.add_argument('--max-duration',type=float,default=30.0); p.add_argument('--keep-case',action='store_true')
def main():
    ap=argparse.ArgumentParser(); sp=ap.add_subparsers(dest='cmd',required=True)
    p=sp.add_parser('librispeech'); common(p); p=sp.add_parser('generic'); common(p)
    p=sp.add_parser('split'); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--valid-ratio',type=float,default=0.05); p.add_argument('--test-ratio',type=float,default=0.05); p.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    if args.cmd=='librispeech': stats(prepare_librispeech(args))
    elif args.cmd=='generic': stats(prepare_generic(args))
    else: split_manifest(args)
if __name__=='__main__': main()
