#!/usr/bin/env python3
"""Identify the exact all-distinct kappa3-times-covariance edge term.

A Gaussian input is passed through one dense ReLU layer, then another linear
map, producing a generic non-Gaussian preactivation state.  At the second ReLU
we compare ARC's kept augmented term set against kept + exactly one dropped
class:

    int_part=(1,1,1), vec_part=((0,1,1),(1,1,1)).

The candidate closed form on all-distinct indices is

  K3_ijk [q1_i q2_j q2_k C_jk
        + q2_i q1_j q2_k C_ik
        + q2_i q2_j q1_k C_ij].

The script recovers any scalar coefficient and reports the residual, so a wrong
multiplicity or Wick convention fails visibly.
"""
from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
from typing import Any
import torch

EDGE_KEY=((1,1,1),((0,1,1),(1,1,1)))

def key(ip,vp):return (tuple(ip),tuple(tuple(x) for x in vp))

def mask3(n,device):
    x=torch.arange(n,device=device);i=x[:,None,None];j=x[None,:,None];k=x[None,None,:]
    return (i!=j)&(i!=k)&(j!=k)

def metrics(t,x,m):
    t=t[m];x=x[m];a=torch.dot(t,x)/torch.dot(x,x);r=t-a*x
    return {'coefficient':float(a),'relative_squared_residual':float(torch.dot(r,r)/torch.dot(t,t)),'max_abs_residual':float(torch.max(torch.abs(r))),'correlation':float(torch.corrcoef(torch.stack((t,x)))[0,1]),'target_rms':float(torch.sqrt(torch.mean(t*t))),'candidate_rms':float(torch.sqrt(torch.mean(x*x)))}

def make_filter(kh,include_edge):
    real=kh.get_all_terms_iso
    def filt(k_max,d_max=None,use_mean_var=False,augment=False):
        ret=real(k_max,d_max=d_max,use_mean_var=use_mean_var,augment=augment)
        return {ip:{vp:c for vp,c in vps.items() if kh.factored_keeps_term(k_max,ip,vp) or (include_edge and key(ip,vp)==EDGE_KEY)} for ip,vps in ret.items()}
    return filt

def nonlinear_with_filter(kh,WK,include_edge):
    from mlp_kprop.kprop_harmonic import Kind,nonlin_kprop
    from mlp_kprop.wick import relu_wick_coef
    real=kh.get_all_terms_iso;kh.get_all_terms_iso=make_filter(kh,include_edge)
    try:return nonlin_kprop(WK,nonlin_wick_coef=relu_wick_coef,k_max=3,kind=Kind.AUGMENT)
    finally:kh.get_all_terms_iso=real

def one(seed,n,avg_metric):
    from mlp_kprop.kprop_harmonic import Kind,coerce_input,linear_kprop,nonlin_kprop
    from mlp_kprop.wick import relu_wick_coef
    torch.manual_seed(seed);dtype=torch.float64
    K0=coerce_input({1:torch.zeros(n,dtype=dtype),2:torch.eye(n,dtype=dtype)},k_max=3,kind=Kind.AUGMENT)
    W1=torch.randn(n,n,dtype=dtype)*math.sqrt(2/n);metric=2*torch.ones(n,dtype=dtype) if avg_metric else None
    P1=linear_kprop(K0,W1,k_max=3,set_metric=metric)
    # Use full first nonlinearity so the probe state contains every generated K3/K4 component.
    A1=nonlin_kprop(P1,nonlin_wick_coef=relu_wick_coef,k_max=3,kind=Kind.AUGMENT)
    W2=torch.randn(n,n,dtype=dtype)*math.sqrt(2/n);P2=linear_kprop(A1,W2,k_max=3,set_metric=metric)
    kept=nonlinear_with_filter(sys.modules['mlp_kprop.kprop_harmonic'],P2,False)
    edge=nonlinear_with_filter(sys.modules['mlp_kprop.kprop_harmonic'],P2,True)
    delta=edge[3].to_tensor()-kept[3].to_tensor()
    C=P2[2].core;K3=P2[3].to_tensor();mean=P2[1].core;var=torch.diagonal(C)
    q1=relu_wick_coef(mean=mean,var=var,k=1,p=1);q2=relu_wick_coef(mean=mean,var=var,k=2,p=1)
    cand=K3*(q1[:,None,None]*q2[None,:,None]*q2[None,None,:]*C[None,:,:]+q2[:,None,None]*q1[None,:,None]*q2[None,None,:]*C[:,None,:]+q2[:,None,None]*q2[None,:,None]*q1[None,None,:]*C[:,:,None])
    return {'seed':seed,'n':n,'avg_metric':avg_metric,'metrics':metrics(delta,cand,mask3(n,delta.device)),'delta_full_rms':float(torch.sqrt(torch.mean(delta*delta))),'delta_diag_rms':float(torch.sqrt(torch.mean(torch.diagonal(delta,dim1=0,dim2=1)**2)))}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arc-src',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--sizes',default='6,8,12');ap.add_argument('--seeds',default='101,202,303');a=ap.parse_args();sys.path.insert(0,str(a.arc_src));torch.set_default_dtype(torch.float64);torch.set_grad_enabled(False)
    rows=[]
    for n in map(int,a.sizes.split(',')):
        for seed in map(int,a.seeds.split(',')):
            for avg in (False,True):
                r=one(seed,n,avg);rows.append(r);m=r['metrics'];print(f"n={n} seed={seed} avg={avg} coef={m['coefficient']:.12g} residual={m['relative_squared_residual']:.3e} corr={m['correlation']:.12f}",flush=True)
    out={'edge_key':EDGE_KEY,'cases':rows};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2,sort_keys=True));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
