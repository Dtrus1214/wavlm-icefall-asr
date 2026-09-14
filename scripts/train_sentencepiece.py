#!/usr/bin/env python3
import argparse
from pathlib import Path
import sentencepiece as spm
from lhotse import load_manifest
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--vocab-size',type=int,default=500); ap.add_argument('--model-type',choices=['bpe','unigram'],default='bpe'); args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True); sups=load_manifest(args.manifest); text=args.output_dir/'transcript.txt'
    with text.open('w',encoding='utf-8') as f:
        for s in sups:
            if s.text: f.write(s.text.strip()+'\n')
    prefix=args.output_dir/'bpe'
    spm.SentencePieceTrainer.train(input=str(text),model_prefix=str(prefix),vocab_size=args.vocab_size,model_type=args.model_type,character_coverage=1.0,bos_id=-1,eos_id=-1,unk_id=0,pad_id=-1)
    print(prefix.with_suffix('.model'))
if __name__=='__main__': main()
