"""Recompute the original strong-baseline accounting for thirty frozen inputs.

Inactivity comes from the supplied/fresh Gram certificates. This check audits
resource accounting, not those identities. The optimizer is unchanged.
"""
import argparse
import json
from pathlib import Path
import sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--module', type=Path, required=True)
    p.add_argument('--gram', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.module))
    from experiment import resource_audit
    rows = []
    for slot in json.loads((a.module/'frozen_inputs/scope.json').read_text()):
        name = slot['id']
        info = json.loads((a.module/'frozen_inputs'/(name+'.json')).read_text())
        result = json.loads((a.gram/(name+'.json')).read_text())
        inactive = [v['symbol'] for v in result['per_symbol'] if v['verdict'] == 'INACTIVE']
        audit = resource_audit(info, inactive)
        if any(audit['incremental'].values()):
            raise AssertionError('Nonzero resource increment: '+name)
        rows.append({'id': name, 'inactive': inactive, 'metrics': audit['metrics'],
                     'incremental': audit['incremental']})
    assert len(rows) == 30
    output = {'passed': True, 'settings': 30, 'resource_positive_settings': 0,
              'scope': 'Recomputed original strong optimizer; certificate validity checked separately.', 'rows': rows}
    a.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k: v for k, v in output.items() if k != 'rows'}, indent=2))

if __name__ == '__main__':
    main()
