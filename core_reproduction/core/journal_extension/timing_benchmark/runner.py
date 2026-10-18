import sys,json
from experiment import run,ROOT,digest
from audit import audit_run
if __name__=='__main__':
    release=ROOT/'release_manifest.json'
    if release.exists():
        for entry in json.loads(release.read_text())['files']:
            if digest(ROOT/entry['path'])!=entry['sha256']:
                raise RuntimeError('Release hash mismatch: '+entry['path'])
        print('RELEASE_HASHES_OK',flush=True)
    else:print('DEVELOPMENT_RUN_UNSEALED_RELEASE',flush=True)
    folder=run(sys.argv[1]);audit_run(folder)
    print('COMPLETE_AND_AUDITED',folder.with_suffix('.zip'),flush=True)
