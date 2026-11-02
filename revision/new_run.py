"""Copy a frozen experiment's declared inputs/code into an empty work directory."""
from pathlib import Path
import argparse,json,shutil
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--experiment',choices=['lomo36','lomo10','lomo3','unitedmet49'],required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();src=Path(__file__).resolve().parent/'experiments'/a.experiment
manifest=json.loads((src/'manifest.json').read_text())
a.output.mkdir(parents=True,exist_ok=False)
for name in [*manifest['files'],'manifest.json']:
 dest=a.output/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src/name,dest)
print('Prepared',a.output)
