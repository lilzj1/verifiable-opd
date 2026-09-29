"""Install public model-config fixtures only; never synthesize GPU admission evidence."""
from pathlib import Path
import shutil

root=Path(__file__).resolve().parents[1]
pairs=[(root/'results/compat_reference4576.json',root/'artifacts/config_compat_v1/reference4576/report.json')]
for source in (root/'results/model_config_fixtures').glob('*.config.json'):
    pairs.append((source,root/'artifacts/two_gpu_preflight_v1'/source.name))
if len(pairs)!=4:raise RuntimeError('Missing published CPU model-config fixtures')
for source,target in pairs:
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and target.read_bytes()!=source.read_bytes():
        raise RuntimeError('Refusing to overwrite a different local reference')
    if not target.exists():shutil.copy2(source,target)
print('CPU fixtures ready; this is not pretrained model/GPU validation.')
