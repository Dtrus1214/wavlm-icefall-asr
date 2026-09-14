#!/usr/bin/env python3
import argparse
from pathlib import Path
from lhotse import Recording,RecordingSet,SupervisionSegment,SupervisionSet,CutSet
from lhotse.qa import fix_manifests,validate_recordings_and_supervisions
def norm_text(s,keep_case=False):
    s=' '.join(s.strip().split()); return s if keep_case else s.upper()
def save(prefix,subset,out,recordings,supervisions):
    out.mkdir(parents=True,exist_ok=True)
    recordings,supervisions=fix_manifests(recordings,supervisions)
    validate_recordings_and_supervisions(recordings,supervisions)
    cuts=CutSet.from_manifests(recordings=recordings,supervisions=supervisions)
    rp=out/f'{prefix}_recordings_{subset}.jsonl.gz'; sp=out/f'{prefix}_supervisions_{subset}.jsonl.gz'; cp=out/f'{prefix}_cuts_{subset}.jsonl.gz'
    recordings.to_file(rp); supervisions.to_file(sp); cuts.to_file(cp)
    print(f'recordings={len(recordings)}'); print(f'supervisions={len(supervisions)}'); print(f'hours={sum(r.duration for r in recordings)/3600:.2f}'); print(rp); print(sp); print(cp)
def prepare_librispeech(args):
    recs=[]; sups=[]
    for tr in sorted(args.input_dir.rglob('*.trans.txt')):
        for line in tr.read_text(encoding='utf-8').splitlines():
            line=line.strip()
            if not line: continue
            p=line.split(maxsplit=1); uid=p[0]; text=norm_text(p[1] if len(p)==2 else '',args.keep_case); audio=tr.parent/f'{uid}.flac'
            if not audio.exists(): continue
            r=Recording.from_file(audio,recording_id=uid)
            if r.duration<args.min_duration or (args.max_duration>0 and r.duration>args.max_duration): continue
            recs.append(r); ids=uid.split('-')
            sups.append(SupervisionSegment(id=uid,recording_id=uid,start=0.0,duration=r.duration,channel=0,text=text,speaker=ids[0],language=args.language,custom={'chapter':ids[1] if len(ids)>1 else ''}))
    save('librispeech',args.subset,args.output_dir,RecordingSet.from_recordings(recs),SupervisionSet.from_segments(sups))
def prepare_generic(args):
    recs=[]; sups=[]; exts={'.wav','.flac','.ogg'}
    for audio in sorted(p for p in args.input_dir.rglob('*') if p.is_file() and p.suffix.lower() in exts):
        txt=audio.with_suffix('.txt')
        if not txt.exists(): continue
        text=norm_text(txt.read_text(encoding='utf-8'),args.keep_case)
        if not text: continue
        uid=str(audio.relative_to(args.input_dir).with_suffix('')).replace('/','_').replace('\\','_')
        r=Recording.from_file(audio,recording_id=uid)
        if r.duration<args.min_duration or (args.max_duration>0 and r.duration>args.max_duration): continue
        if r.num_channels!=1: print(f'skip non-mono: {audio}'); continue
        recs.append(r); speaker=audio.parent.name if args.speaker_from_parent else None
        sups.append(SupervisionSegment(id=uid,recording_id=uid,start=0.0,duration=r.duration,channel=0,text=text,speaker=speaker,language=args.language))
    save(args.prefix,args.subset,args.output_dir,RecordingSet.from_recordings(recs),SupervisionSet.from_segments(sups))
def add_common(p):
    p.add_argument('--input-dir',type=Path,required=True); p.add_argument('--subset',required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--language',default='en'); p.add_argument('--min-duration',type=float,default=0.2); p.add_argument('--max-duration',type=float,default=30.0); p.add_argument('--keep-case',action='store_true')
def main():
    ap=argparse.ArgumentParser(); sp=ap.add_subparsers(dest='cmd',required=True)
    p=sp.add_parser('librispeech'); add_common(p)
    p=sp.add_parser('generic'); add_common(p); p.add_argument('--prefix',default='mydata'); p.add_argument('--speaker-from-parent',action='store_true')
    args=ap.parse_args(); prepare_librispeech(args) if args.cmd=='librispeech' else prepare_generic(args)
if __name__=='__main__': main()
