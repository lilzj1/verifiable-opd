"""Fetch immutable dependencies without touching an existing checkout or GPUs."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'verl-v0.8.0': ('https://github.com/verl-project/verl.git', '7aed6b230776f963fa09509c10d9c3a767d1102c'),
    'open-mopd-reference': ('https://github.com/BytedTsinghua-SIA/Open-MOPD.git', '4809a96cf85a869106ff0ff3f37d0a51e12010ae'),
    'ifbench-reference': ('https://github.com/allenai/IFBench.git', '1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d'),
}

def main():
    for name, (url, revision) in SOURCES.items():
        dest = ROOT/'upstream'/name
        if dest.exists():
            head = subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'], text=True).strip()
            dirty = subprocess.check_output(['git','-C',str(dest),'status','--porcelain'], text=True)
            if head != revision or dirty:
                raise RuntimeError(f'{name}: existing checkout differs; refusing to overwrite')
        else:
            dest.mkdir(parents=True)
            for args in (['init'], ['remote','add','origin',url], ['fetch','--depth','1','origin',revision], ['checkout','--detach','FETCH_HEAD']):
                subprocess.run(['git','-C',str(dest),*args], check=True)
            if subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True).strip() != revision:
                raise RuntimeError('Commit verification failed')
        print(name, revision)

if __name__ == '__main__':
    main()
