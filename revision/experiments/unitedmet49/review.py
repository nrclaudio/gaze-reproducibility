"""Independently replay saved posterior predictions and paired correlations."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata,spearmanr
from common import save,sha,verify


def replay(p,column,draws,seed):
    # Separate implementation of worker prediction, with explicit inverse
    # permutations. Uses no assessment outcomes.
    rng=np.random.RandomState(seed); n=len(p['w']); total=np.zeros(n)
    for i in range(draws):
        samples=rng.normal(p['w'],p['ws'])
        feature=rng.normal(p['h'][:,column],p['hs'][:,column])
        values=samples.dot(feature)+rng.gumbel(0,1,size=n)
        permutation=np.argsort(-values,kind='stable')
        ranks=np.empty(n,dtype=int); ranks[permutation]=np.arange(n)
        total+=ranks
    return 1-(total/draws+.5)/n


def review(root):
    root=Path(root).resolve(); verify(root)
    tasks=json.loads((root/'tasks.json').read_text())
    settings=json.loads((root/'protocol.json').read_text())['settings']
    with np.load(root/'inputs/assessment.npz') as z: truth=dict(z)
    with np.load(root/'inputs/fit.npz') as z: fit_input=dict(z)
    inventory=[];rows=[];outputs={}; all_predictions=[]
    for task in tasks:
        out=root/'results'/task['id']; item=dict(task=task['id'],state='missing',error='')
        try:
            status=json.loads((out/'status.json').read_text())
            if status['state']!='complete' or status['smoke']: raise ValueError('Fit is not complete.')
            if status['manifest_sha256']!=sha(root/'manifest.json'):raise ValueError('Wrong manifest.')
            assert status['task']==task
            for name,expected in status['artifacts'].items():
                if sha(out/name)!=expected: raise ValueError('Changed result '+name)
            config=json.loads((out/'config.json').read_text())
            assert config['task']==task and config['settings']==settings and not config['assessment_loaded']
            assert config['training_input_sha256']==sha(root/'inputs/fit.npz')
            labels=fit_input['met'].astype(np.float64,copy=True);labels[:,task['column']]=np.nan
            for j in range(labels.shape[1]):
                finite=np.isfinite(labels[:,j])
                if finite.any():labels[finite,j]=labels[finite,j]-np.min(labels[finite,j])+1.
            matrix=np.concatenate([labels,fit_input['rna'].astype(np.float64)],axis=1)
            assert config['matrix_sha256']==hashlib.sha256(matrix.tobytes()).hexdigest()
            assert config['steps_run']==settings['steps']
            assert config['diagnostic']['target_observations']==0
            assert config['diagnostic']['target_likelihood_gradient_max']==0
            assert config['diagnostic']['likelihood_change_on_target_perturbation']==0
            assert config['diagnostic']['observed_feature_gradient_max']>0
            with np.load(out/'posterior.npz') as z: post=dict(z)
            assert len(post['losses'])==settings['steps'] and np.isfinite(post['losses']).all()
            with np.load(out/'predictions.npz') as z: saved=dict(z)
            prediction=replay(post,task['column'],settings['posterior_draws'],settings['seed'])
            np.testing.assert_array_equal(prediction,saved['predictions'])
            np.testing.assert_array_equal(saved['sample_ids'],truth['sample_ids'])
            i=task['index']; assert str(saved['feature'])==str(truth['features'][i])==task['feature']
            assert int(saved['node'])==int(truth['nodes'][i])==task['node']
            y=truth['targets'][:,i]; observed=np.isfinite(y);g=truth['predictions'][:,i]
            assert observed.sum()>=10 and np.ptp(y[observed])>0 and np.isfinite(prediction).all()
            def score(x):
                if np.ptp(x[observed])==0:return 0.
                a=float(spearmanr(x[observed],y[observed]).statistic)
                b=float(np.corrcoef(rankdata(x[observed]),rankdata(y[observed]))[0,1])
                assert np.isclose(a,b,atol=1e-12);return a
            rows.append(dict(feature=task['feature'],node=task['node'],n_observed=int(observed.sum()),
                gaze_spearman=score(g),comparator_spearman=score(prediction),
                constant_prediction=bool(np.ptp(prediction[observed])==0)))
            all_predictions.append(prediction)
            outputs[str(out.relative_to(root))+'/status.json']=sha(out/'status.json')
            item['state']='verified'
        except Exception as exc:
            item['error']=f'{type(exc).__name__}: {exc}'
        inventory.append(item)
    dest=root/'review';dest.mkdir(exist_ok=True)
    pd.DataFrame(inventory).to_csv(dest/'inventory.csv',index=False)
    complete=len(rows)==len(tasks)==49
    summary=dict(state='complete' if complete else 'incomplete',verified=len(rows),expected=49,
        matched_outcomes=True,posterior_replay=True,manifest_sha256=sha(root/'manifest.json'),
        result_status_hashes=outputs,zero_shot_stress_test=True,gaze_refits=0)
    if complete:
        frame=pd.DataFrame(rows);frame.to_csv(dest/'paired.csv',index=False)
        np.savez_compressed(dest/'predictions.npz',predictions=np.array(all_predictions).T,
            sample_ids=truth['sample_ids'],features=truth['features'],nodes=truth['nodes'])
        summary.update(median_unitedmet_spearman=float(frame.comparator_spearman.median()),
            median_gaze_spearman=float(frame.gaze_spearman.median()),
            gaze_higher=int((frame.gaze_spearman>frame.comparator_spearman).sum()),
            constant_predictions=int(frame.constant_prediction.sum()),
            artifacts={n:sha(dest/n) for n in ['paired.csv','predictions.npz','inventory.csv']})
    save(dest/'verification.json',summary)
    print(f"UnitedMet LOMO: {len(rows)}/49 verified; {summary['state']}.",flush=True)
    if complete: print(json.dumps({k:v for k,v in summary.items() if k.startswith('median_') or k=='gaze_higher'},indent=2))
    return 0 if complete else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd())
    raise SystemExit(review(p.parse_args().root))
