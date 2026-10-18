"""Compare an emitted 54-row census to the two archived TeX rank tables.

This is presentation consistency, not a substitute for certificate replay.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
FAMILIES = {'ChebyshevRx': 'chebyshev_rx', 'ParamZ': 'param_z_feature_map',
            'YZ-CX': 'yz_cx_encoding_circuit', 'Hubregtsen': 'hubregtsen_encoding_circuit',
            'Kyriienko': 'kyriienko_nonlinear_encoding_circuit'}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('census', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    rows = {}
    with a.census.open(newline='') as f:
        for row in csv.DictReader(f):
            if row['setting'] in rows: raise ValueError('Duplicate census row')
            rows[row['setting']] = row
    seen = set(); sources = {}
    for name in ('dimension_results.tex', 'extension_results.tex'):
        p = ROOT/'manuscript_reference'/name
        sources[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        family = None
        for line in p.read_text().splitlines():
            fields = [s.strip() for s in line.split('\\\\')[0].split('&')]
            if len(fields) != 7 or not fields[1].isdigit(): continue
            if fields[0]: family = FAMILIES[fields[0]]
            n, layer, count = map(int, fields[1:4])
            for readout, cell in zip(('local_xyz', 'local_z', 'pauli_weight_le_2'), fields[4:]):
                rank = int(re.sub(r'\D', '', cell))
                name = '%s_n%d_l%d_%s' % (family, n, layer, readout)
                r = rows[name]
                if (int(r['source_symbols']), int(r['rank_lower']), int(r['rank_upper'])) != (count, rank, rank):
                    raise ValueError('Manuscript/census mismatch: '+name)
                if name in seen: raise ValueError('Duplicate manuscript row')
                seen.add(name)
    if len(seen) != 54 or set(rows) != seen: raise ValueError('Expected exact 54-setting coverage')
    result = {'passed': True, 'settings': 54, 'manuscript_sha256': sources,
              'census_sha256': hashlib.sha256(a.census.read_bytes()).hexdigest(),
              'scope': 'Table consistency only; certificate calculations are separate.'}
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
