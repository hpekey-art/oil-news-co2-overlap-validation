#!/usr/bin/env python3
"""One-shot, prospectively timestamped fixed-threshold validation. NEVER run before external timestamp deposit."""
import argparse, hashlib, json, math, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta

DIR=Path(__file__).resolve().parent
assert (DIR/'EXTERNAL_TIMESTAMP_RECEIPT.txt').exists(), 'STOP: External timestamp receipt absent. Do not execute validation.'
manifest=json.loads((DIR/'PREEXECUTION_MANIFEST.json').read_text())
assert manifest['parameters']['threshold']==4.20
assert manifest['parameters']['reps_per_scenario']==50000
assert manifest['parameters']['master_seed']==202610180359
assert manifest['parameters']['bonferroni_alpha_per_scenario']==0.00625
assert hashlib.sha256((DIR/'run_untouched_validation.py').read_bytes()).hexdigest()==manifest['code_sha256'], 'Code changed since lock.'
assert (DIR/'EXTERNAL_TIMESTAMP_RECEIPT.txt').read_text().strip(), 'Empty receipt'
T=31; J=6; N=36
q=np.array([0.683166,-0.840445,0.776775,-0.686588,-0.402135,0.023196,0.632417,-2.103639,-1.659146,1.208154,-0.108200,-0.402119,0.952214,-0.399745,0.409944,0.671073,-1.060105,0.560746,-0.168689,2.965236,-0.493889,0.138733,0.441713,0.584718,-1.629526,-0.974378,0.294171,-1.208686,-0.107121,0.523759,-0.899533]);q1=np.r_[0,q[:-1]];q2=np.r_[0,0,q[:-2]];tr=np.linspace(-1,1,T)
X=np.column_stack([q,np.ones(T),tr,q1,q2]);
for k in (2,3,4):X[:,k]=(X[:,k]-X[:,k].mean())/X[:,k].std(ddof=0)
A=np.linalg.inv(X.T@X)@X.T;a=A[0];den=1-np.diag(X@A)
assert abs(np.linalg.cond(X.T@X)-1.7774713036514531)<1e-5
assert abs(np.max(1-den)-0.3834497634525881)<1e-5
assert hashlib.sha256(X.tobytes()).hexdigest()==manifest['design_matrix_sha256']
R=np.full((J,J),.45);np.fill_diagonal(R,1);L=np.linalg.cholesky(R)
scenarios=[('A_overlap_iid',0.,False,0.,False,False),('B_overlap_ar03',.30,False,0.,False,False),('C_overlap_ar06',.60,False,0.,False,False),('D_overlap_t5_ar03',.30,True,0.,False,False),('E_overlap_common_ar04',.40,False,.45,False,False),('F_overlap_ar075_common',.75,False,.20,False,False),('G_overlap_hetero_ar04',.40,False,.20,True,False),('H_overlap_MA2_ar03',.30,False,.20,False,True)]

def draw(cfg,rr):
    _,rho,heavy,common,hetero,ma2=cfg
    Z=rr.standard_normal((N,J))@L.T
    if heavy: Z*=np.sqrt(5/rr.chisquare(5,N))[:,None]/np.sqrt(5/3)
    if common: Z=np.sqrt(1-common**2)*Z+common*rr.standard_normal(N)[:,None]
    if ma2:
        raw=Z.copy();Z=np.zeros_like(raw);w=np.array([1.,.65,.35]);norm=np.sqrt(np.sum(w*w))
        for t in range(N):
            for l,ww in enumerate(w):
                if t-l>=0:Z[t]+=ww*raw[t-l]
            Z[t]/=norm
    E=np.zeros((N,J));E[0]=Z[0]
    for t in range(1,N):E[t]=rho*E[t-1]+np.sqrt(max(1-rho*rho,1e-12))*Z[t]
    if hetero:E*=np.linspace(.60,1.80,N)[:,None]
    Y=np.empty((T,J))
    for t in range(T):Y[t,0]=E[t:t+6,0].sum();Y[t,1:]=E[t+3:t+6,1:].mean(axis=0)
    B=A@Y;res=Y-X@B;se=np.sqrt(np.sum((a[:,None]**2)*(res**2)/(den[:,None]**2),axis=0))
    return np.max(np.abs(B[0]/se))

rows=[]; root=np.random.SeedSequence(manifest['parameters']['master_seed']); children=root.spawn(8)
for idx,cfg in enumerate(scenarios):
    rr=np.random.default_rng(children[idx]); values=np.empty(50000)
    for i in range(50000):values[i]=draw(cfg,rr)
    count=int(np.count_nonzero(values<=4.20));p=count/50000
    lower=float(beta.ppf(0.00625,count,50000-count+1)) if count>0 else 0.0
    rows.append(dict(scenario=cfg[0],covered=count,reps=50000,empirical_coverage=p,mc_se=math.sqrt(p*(1-p)/50000),bonferroni_one_sided_exact_lower_bound=lower,q95=float(np.quantile(values,.95)),q975=float(np.quantile(values,.975)),pass_rule=lower>=.95))
    print(cfg[0], count, lower,flush=True)
output=DIR/'UNTOUCHED_VALIDATION_RESULTS.csv';pd.DataFrame(rows).to_csv(output,index=False)
summary=dict(all_pass=all(x['pass_rule'] for x in rows),threshold=4.20,bonferroni_alpha=.00625,results_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),numpy=np.__version__,python=sys.version,platform=platform.platform(),scenario_results=rows)
(DIR/'UNTOUCHED_VALIDATION_VERDICT.json').write_text(json.dumps(summary,indent=2))
print('FINAL VALIDATION', 'PASS' if summary['all_pass'] else 'FAIL')
