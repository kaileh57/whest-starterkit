#!/usr/bin/env python3
"""Recursive ablation of ARC's four dropped augmented-K3 diagram classes.

Runs the unfactorized source implementation with exactly controlled term sets:
kept-only, each omitted class added separately, all kappa3-edge classes, and full
AUGMENT.  Every arm sees identical generated MLPs and is scored against an
independent high-sample Monte Carlo reference.  This is intentionally small-width
scientific identification, not a challenge score estimate.
"""
from __future__ import annotations
import argparse,json,math,sys,time
from pathlib import Path
from typing import Any
import numpy as np
import torch

TRI=((1,1,1),((0,1,1),(1,0,1),(1,1,0)))
E111=((1,1,1),((0,1,1),(1,1,1)))
E211A=((2,1,1),((0,1,1),(1,1,1)))
E211B=((2,1,1),((1,0,1),(1,1,1)))


def key(ip,vp):return (tuple(ip),tuple(tuple(x) for x in vp))

def make_filter(kh, extras:set[tuple]):
    real=kh.get_all_terms_iso
    def filtered(k_max,d_max=None,use_mean_var=False,augment=False):
        ret=real(k_max,d_max=d_max,use_mean_var=use_mean_var,augment=augment)
        return {ip:{vp:c for vp,c in vps.items() if kh.factored_keeps_term(k_max,ip,vp) or key(ip,vp) in extras} for ip,vps in ret.items()}
    return filtered

def mc_reference(weights,samples,seed,batch=32768):
    rng=np.random.default_rng(seed);depth=len(weights);n=weights[0].shape[0];sums=np.zeros((depth,n));N=0
    for st in range(0,samples,batch):
        nb=min(batch,samples-st);x=rng.standard_normal((nb,n))
        for ell,W in enumerate(weights):
            x=x@W.T;np.maximum(x,0,out=x);sums[ell]+=x.sum(0)
        N+=nb
    return sums/N

def run_arm(kh,weights,extras,full=False):
    from mlp_kprop.kprop_harmonic import Kind,coerce_input,linear_kprop,nonlin_kprop
    from mlp_kprop.wick import relu_wick_coef
    n=weights[0].shape[0];K=coerce_input({1:torch.zeros(n),2:torch.eye(n)},k_max=3,kind=Kind.AUGMENT);rows=[];real=kh.get_all_terms_iso
    if not full:kh.get_all_terms_iso=make_filter(kh,extras)
    try:
        for W in weights:
            WK=linear_kprop(K,W,k_max=3,set_metric=None)
            K=nonlin_kprop(WK,nonlin_wick_coef=relu_wick_coef,k_max=3,kind=Kind.AUGMENT)
            rows.append(K[1].to_tensor().detach().cpu().numpy())
    finally:kh.get_all_terms_iso=real
    return np.stack(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arc-src',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--width',type=int,default=12);ap.add_argument('--depth',type=int,default=32);ap.add_argument('--networks',type=int,default=6);ap.add_argument('--mc-samples',type=int,default=300000);ap.add_argument('--seed',type=int,default=260811);a=ap.parse_args();sys.path.insert(0,str(a.arc_src));torch.set_default_dtype(torch.float64);torch.set_grad_enabled(False)
    import mlp_kprop.kprop_harmonic as kh
    arms={'kept':set(),'triangle':{TRI},'edge111':{E111},'edge211a':{E211A},'edge211b':{E211B},'all_edges':{E111,E211A,E211B}}
    result={'width':a.width,'depth':a.depth,'networks':a.networks,'mc_samples':a.mc_samples,'seed':a.seed,'records':[]};rng=np.random.default_rng(a.seed);t0=time.time()
    for net in range(a.networks):
        ws=[torch.tensor(rng.standard_normal((a.width,a.width))*math.sqrt(2/a.width),dtype=torch.float64) for _ in range(a.depth)];wn=[w.numpy() for w in ws];ref=mc_reference(wn,a.mc_samples,a.seed+10000+net);pred={}
        for name,extra in arms.items():
            st=time.time();pred[name]=run_arm(kh,ws,extra);print(f'net={net} arm={name} sec={time.time()-st:.2f}',flush=True)
        st=time.time();pred['full']=run_arm(kh,ws,set(),full=True);print(f'net={net} arm=full sec={time.time()-st:.2f}',flush=True)
        rec={'network':net,'arms':{}}
        for name,p in pred.items():
            rec['arms'][name]={'final_mse':float(np.mean((p[-1]-ref[-1])**2)),'all_mse':float(np.mean((p-ref)**2)),'finite':bool(np.all(np.isfinite(p))),'final_mean_rms':float(np.sqrt(np.mean(p[-1]**2)))}
        result['records'].append(rec);print(json.dumps(rec),flush=True)
    for name in list(arms)+['full']:
        vals=[r['arms'][name]['final_mse'] for r in result['records']];result.setdefault('aggregate',{})[name]={'mean_final_mse':float(np.mean(vals)),'median_final_mse':float(np.median(vals)),'geomean_final_mse':float(np.exp(np.mean(np.log(vals)))),'wins_vs_kept':int(sum(r['arms'][name]['final_mse']<r['arms']['kept']['final_mse'] for r in result['records']))}
    result['elapsed']=time.time()-t0;a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,sort_keys=True));print(json.dumps(result['aggregate'],indent=2),flush=True)
if __name__=='__main__':main()
