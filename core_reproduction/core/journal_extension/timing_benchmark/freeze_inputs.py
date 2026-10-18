"""Freeze already audited server inputs; never refit preprocessing in timing."""
import argparse,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
SHA='27751ea6abf0a0ab055b9b964e23eb14fc7cf4a84c56d91e68ce6d6cdd2682f1'
def main():
    p=argparse.ArgumentParser();p.add_argument('archive');a=p.parse_args()
    archive=Path(a.archive);assert hashlib.sha256(archive.read_bytes()).hexdigest()==SHA
    folder=ROOT/'frozen_inputs';folder.mkdir(exist_ok=True);entries=[]
    with zipfile.ZipFile(archive) as z:
        for name in sorted(z.namelist()):
            if 'transfer_full_' not in name or not name.endswith('.npz'):continue
            record=json.loads(z.read(name[:-4]+'.json'))
            with np.load(io.BytesIO(z.read(name)),allow_pickle=False) as arrays:
                assert np.array_equal(arrays['encoded'],record['encoded_input'])
                assert np.array_equal(arrays['target'],record['target_standardized'])
            item={'X':record['encoded_input'],'y':record['target_standardized'],
                  'raw_y':record['raw_target'],'metadata':record['metadata'],
                  'source_archive_sha256':SHA,'source_member':name}
            dest=folder/(Path(name).stem+'.json')
            if dest.exists():raise FileExistsError(dest)
            dest.write_text(json.dumps(item,sort_keys=True,indent=2)+'\n')
            entries.append({'path':str(dest.relative_to(ROOT)),
                            'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
    assert len(entries)==15
    (ROOT/'frozen_manifest.json').write_text(json.dumps({'archive_sha256':SHA,'files':entries},indent=2)+'\n')
    print('Frozen 15 audited tasks, each 64 train + 64 test; no validation/model selection.')
if __name__=='__main__':main()
