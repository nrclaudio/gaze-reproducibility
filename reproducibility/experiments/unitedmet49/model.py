"""Pinned UnitedMet likelihood, with explicit target masking and saved posteriors."""
import sys
import time
from pathlib import Path
import numpy as np
import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam


def load_upstream(root):
    sys.path.insert(0,str(Path(root)/'upstream'))
    from Performance_Benchmarking.scripts.pyro_model import generate_pyro_data, PlackettLuce_2D
    return generate_pyro_data, PlackettLuce_2D


def training_matrix(met,rna,column):
    labels=np.array(met,dtype=np.float64,copy=True)
    if not 0 <= column < labels.shape[1]: raise ValueError('Invalid target column.')
    labels[:,column]=np.nan
    # The upstream routine substitutes 0 for missing observations. Source
    # workbook values can be negative, so shift each observed column above 0.
    for j in range(labels.shape[1]):
        ok=np.isfinite(labels[:,j])
        if ok.any(): labels[ok,j]=labels[ok,j]-labels[ok,j].min()+1.
    matrix=np.column_stack([labels,np.asarray(rna,dtype=np.float64)])
    if np.isinf(matrix).any(): raise ValueError('Infinite model input.')
    assert np.isnan(matrix[:,column]).all()
    return matrix


def fit(root,matrix,column,dimensions=30,steps=4000,lr=.001,seed=42):
    generate,distribution=load_upstream(root)
    N,J,K,n_obs,orders,_,_=generate(matrix,dimensions,1,np.zeros(len(matrix),dtype=int))
    assert int(n_obs[0,column]) == 0
    orders=torch.tensor(orders); counts=torch.tensor(n_obs[0])
    pyro.clear_param_store(); pyro.set_rng_seed(seed)
    def model():
        w=pyro.sample('W',dist.Normal(torch.zeros(N,K),torch.ones(N,K)).to_event(2))
        h=pyro.sample('H',dist.Normal(torch.zeros(K,J),torch.ones(K,J)).to_event(2))
        pyro.sample('R_0',distribution(w@h,counts),obs=orders)
    guide=AutoNormal(poutine.block(model,expose=['W','H']))
    svi=SVI(model,guide,Adam({'lr':lr,'betas':[.95,.999]}),loss=Trace_ELBO())
    started=time.monotonic(); losses=[]
    for step in range(steps):
        value=float(svi.step())
        if not np.isfinite(value): raise ValueError('Nonfinite training loss.')
        losses.append(value)
        if step==0 or (step+1)%500==0:
            print(f'Step {step+1}/{steps}; loss {value:.3f}; elapsed {time.monotonic()-started:.1f}s',flush=True)
    result={key:pyro.param('AutoNormal.'+kind+'.'+name).detach().cpu().numpy().copy()
            for key,kind,name in [('w','locs','W'),('ws','scales','W'),('h','locs','H'),('hs','scales','H')]}
    if not all(np.isfinite(v).all() for v in result.values()): raise ValueError('Nonfinite posterior.')
    if (result['ws']<=0).any() or (result['hs']<=0).any(): raise ValueError('Invalid posterior scales.')
    # Verify the target cannot influence the likelihood, using a nonzero random
    # target direction and a positive observed-feature control.
    rng=np.random.default_rng(seed)
    w=torch.tensor(result['w']); h=torch.tensor(result['h'],requires_grad=True)
    value=distribution(w@h,counts).log_prob(orders)
    grad=torch.autograd.grad(value,h)[0]
    changed=h.detach().clone(); changed[:,column]+=torch.tensor(rng.normal(size=K),dtype=h.dtype)*10
    delta=float((distribution(w@changed,counts).log_prob(orders)-value.detach()).abs())
    target_grad=float(grad[:,column].abs().max())
    other_grad=float(grad[:,counts>0].abs().max())
    if target_grad!=0 or delta!=0 or other_grad<=0: raise ValueError('Masking diagnostic failed.')
    result['losses']=np.asarray(losses)
    return result,dict(target_observations=0,target_likelihood_gradient_max=target_grad,
                       likelihood_change_on_target_perturbation=delta,observed_feature_gradient_max=other_grad)


def predict(posterior,column,draws=500,seed=42):
    """Posterior Gumbel rank draws; score increases with predicted abundance."""
    rng=np.random.RandomState(seed)
    n=len(posterior['w']); summed=np.zeros(n)
    for _ in range(draws):
        w=rng.normal(posterior['w'],posterior['ws'])
        h=rng.normal(posterior['h'][:,column],posterior['hs'][:,column])
        logits=w@h
        order=np.argsort(-(logits+rng.gumbel(size=n)),kind='stable')
        summed+=order.argsort(kind='stable')
    return 1-(summed/draws+.5)/n
