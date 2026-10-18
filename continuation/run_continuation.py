#!/usr/bin/env python3
"""Operational wrapper only; frozen scenario child stream, 50,000 draws and 4.20 threshold."""
import argparse, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parent
PRE = ROOT / "frozen_preexecution"
assert (ROOT / "PROVENANCE_PRECONDITIONS_VERIFIED.txt").is_file(), "HOLD: provenance not archived"
manifest = json.loads((PRE / "PREEXECUTION_MANIFEST.json").read_text())
source = (PRE / "run_untouched_validation.py").read_bytes()
assert hashlib.sha256(source).hexdigest() == manifest["code_sha256"]
assert hashlib.sha256((PRE / "PREEXECUTION_MANIFEST.json").read_bytes()).hexdigest() == "4693eb9c128aa75130aad62f93c47d40474e66047595a7f3a5bd46eb47766e7c"
assert manifest["parameters"]["master_seed"] == 202610180359
assert manifest["parameters"]["reps_per_scenario"] == 50000
assert manifest["parameters"]["threshold"] == 4.20
assert manifest["parameters"]["bonferroni_alpha_per_scenario"] == .00625
p=argparse.ArgumentParser();p.add_argument("--scenario-index",type=int,required=True,choices=range(2,8));args=p.parse_args()
idx=args.scenario_index
assert not (ROOT / f"scenario_{idx}.json").exists(), "No re-run permitted"
# Execute only frozen definitions (not the frozen simulation loop). The original
# local receipt was not recovered; the nonempty receipt for this NEW execution
# explicitly says POST-INTERRUPTION and never claims to be the original.
prelude=source.decode().split("rows=[]; root=np.random.SeedSequence",1)[0]
scope={"__file__":str(PRE/"run_untouched_validation.py"),"__name__":"frozen_definitions"}
exec(compile(prelude,str(PRE/"run_untouched_validation.py"),"exec"),scope)
root=np.random.SeedSequence(202610180359)
children=root.spawn(8)
assert tuple(children[idx].spawn_key) == (idx,)
scenarios=scope["scenarios"]
assert len(scenarios)==8
cfg=scenarios[idx];rng=np.random.default_rng(children[idx]);draw=scope["draw"]
start=time.time();v=np.empty(50000)
for i in range(50000):v[i]=draw(cfg,rng)
covered=int(np.count_nonzero(v<=4.20));prob=covered/50000
lower=float(beta.ppf(.00625,covered,50000-covered+1)) if covered else 0.
result={"scenario":cfg[0],"scenario_index":idx,"spawn_key":list(children[idx].spawn_key),"covered":covered,"n":50000,"coverage":prob,"lower_bound":lower,"passed":lower>=.95,"q95":float(np.quantile(v,.95)),"q975":float(np.quantile(v,.975)),"critical":4.20,"elapsed_seconds":time.time()-start,"python":sys.version,"numpy":np.__version__,"scipy":__import__("scipy").__version__,"pandas":__import__("pandas").__version__,"platform":platform.platform()}
output=ROOT/f"scenario_{idx}.json"
with output.open("x") as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2),flush=True)
