#!/usr/bin/env python3
"""Deterministic aggregation: no inference selection or rerun."""
import json
from pathlib import Path
from scipy.stats import beta
R=Path(__file__).resolve().parent
m=json.loads((R/'CONTINUATION_MANIFEST.json').read_text())
assert (R/'PROVENANCE_PRECONDITIONS_VERIFIED.txt').is_file()
results=[]
for j in range(8):
    if j<2:
        a=m['original_AB_observed'][j];k=int(a['covered']);n=int(a['n']);name=a['scenario']
    else:
        a=json.loads((R/f'scenario_{j}.json').read_text());assert a['scenario_index']==j and a['spawn_key']==[j];k=int(a['covered']);n=int(a['n']);name=a['scenario']
    assert n==50000
    bound=float(beta.ppf(.00625,k,n-k+1))
    results.append({'index':j,'scenario':name,'covered':k,'n':n,'exact_lower':bound,'pass':bound>=.95})
assert len(results)==8 and len(set(x['scenario'] for x in results))==8
status='PASS' if all(x['pass'] for x in results) else 'FAIL'
out={'status':status,'results':results,'note':'Deterministic segmented execution; A/B observed before continuation. No uninterrupted-run claim.'}
with (R/'FINAL_VALIDATION_VERDICT.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
