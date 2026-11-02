"""Fixed-coverage, paired descriptive statistics, conditional on these studies."""
import numpy as np
from scipy.stats import rankdata, binomtest


def correlations(truth, prediction, minimum=10):
    truth=np.asarray(truth,dtype=float); prediction=np.asarray(prediction,dtype=float)
    observed=np.isfinite(truth)
    if not np.isfinite(prediction[observed]).all(): raise ValueError('Missing prediction on a measured outcome.')
    x=rankdata(truth,axis=0,nan_policy='omit'); y=rankdata(np.where(observed,prediction,np.nan),axis=0,nan_policy='omit')
    n=observed.sum(0)
    x=x-np.nansum(x,axis=0)/np.maximum(n,1); y=y-np.nansum(y,axis=0)/np.maximum(n,1)
    xx=np.nansum(x*x,axis=0); yy=np.nansum(y*y,axis=0)
    denominator=np.sqrt(xx*yy)
    rho=np.divide(np.nansum(x*y,axis=0),denominator,out=np.zeros(truth.shape[1]),where=denominator>0)
    eligible=(n>=minimum)&(xx>0)
    rho[~eligible]=np.nan
    return rho,eligible,n,(yy==0)&eligible


def score(truth,prediction,groups,kind,panels=None,bootstrap=False):
    scores=[]; eligibility=[]
    for j,group in enumerate(np.unique(groups)):
        rows=groups==group
        rho,mask,_,_=correlations(truth[rows],prediction[rows],minimum=1 if bootstrap else 10)
        if panels is not None:
            mask=panels[j]
            # A resampled stratum can lose outcome variation; retain its fixed cells at zero.
            rho=np.nan_to_num(rho,nan=0.)
        scores.append(np.where(mask,rho,np.nan)); eligibility.append(mask)
    scores=np.asarray(scores)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        # Equal weight per target for LOMO; equal weight per stratum for sample holdout.
        summary=np.nanmedian(np.nanmedian(scores,axis=0)) if kind=='lomo' else np.nanmean(np.nanmedian(scores,axis=1))
    return float(summary),np.asarray(eligibility)


def metrics_rows(layout,predictions):
    truth=layout['targets']; groups=layout['groups']; result=[]
    for group in [*np.unique(groups),'__pooled_secondary__']:
        rows=np.ones(len(groups),bool) if group=='__pooled_secondary__' else groups==group
        y,p=truth[rows],predictions[rows]
        rho,eligible,n,constant=correlations(y,p)
        for col,feature in enumerate(layout['features']):
            good=np.isfinite(y[:,col]); yt=y[good,col]; yp=p[good,col]
            mse=float(np.mean((yt-yp)**2)) if len(yt) else None
            denom=float(np.sum((yt-yt.mean())**2)) if len(yt) else 0.
            result.append(dict(group=str(group),feature=str(feature),node=int(layout['nodes'][col]),n=int(n[col]),
                scorable=bool(eligible[col]),constant_prediction=bool(constant[col]),
                spearman=float(rho[col]) if eligible[col] else None,mse=mse,
                r2=float(1-np.sum((yt-yp)**2)/denom) if denom>0 else None))
    return result


def paired_comparison(truth,reference,candidate,groups,kind,repetitions=1000,seed=27102026):
    """Same row/seed draws for both methods; targets and study panel remain fixed."""
    if reference.shape!=candidate.shape or reference.shape[1:]!=truth.shape: raise ValueError('Unpaired predictions.')
    _,panels=score(truth,reference[0],groups,kind)
    effects=np.array([score(truth,c,groups,kind,panels)[0]-score(truth,r,groups,kind,panels)[0]
                      for r,c in zip(reference,candidate)])
    if not np.isfinite(effects).all(): raise ValueError('No scorable fixed panel.')
    rng=np.random.default_rng(seed); draws=[]
    strata=[np.flatnonzero(groups==g) for g in np.unique(groups)]
    for _ in range(repetitions):
        rows=np.concatenate([rng.choice(ix,len(ix),replace=True) for ix in strata])
        chosen=rng.integers(0,len(reference),size=len(reference))
        differences=[score(truth[rows],candidate[s,rows],groups[rows],kind,panels,True)[0]-
                     score(truth[rows],reference[s,rows],groups[rows],kind,panels,True)[0] for s in chosen]
        draws.append(float(np.mean(differences)))
    nonzero=effects[effects!=0]
    p=float(binomtest(int((nonzero>0).sum()),len(nonzero)).pvalue) if len(nonzero) else 1.
    return dict(seed_effects=effects.tolist(),mean_effect=float(effects.mean()),
                descriptive_bootstrap_ci95=np.quantile(draws,[.025,.975]).tolist(),bootstrap_repetitions=repetitions,
                exact_seed_sign_p=p,conditional_on_fixed_targets_and_studies=True)


def holm(values):
    values=np.asarray(values); order=np.argsort(values); result=np.ones(len(values)); previous=0.
    for rank,index in enumerate(order):
        previous=max(previous,float(values[index])*(len(values)-rank)); result[index]=min(1.,previous)
    return result.tolist()
