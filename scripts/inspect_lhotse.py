#!/usr/bin/env python3
import argparse
from lhotse import load_manifest_lazy
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cuts',required=True); ap.add_argument('--n',type=int,default=5); args=ap.parse_args(); cuts=load_manifest_lazy(args.cuts); total=0.0; count=0
    for cut in cuts:
        if count<args.n:
            text=cut.supervisions[0].text if cut.supervisions else ''; print(cut.id,f'{cut.duration:.2f}s',text)
        total+=cut.duration; count+=1
    print(f'cuts={count}, hours={total/3600:.2f}')
if __name__=='__main__': main()
