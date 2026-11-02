"""Meaningful regression checks for masks, matched ablations, paired statistics and submission safety."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
import numpy as np
from common import save,sha
from protocol import VARIANTS,SEEDS,make_tasks,architecture,matched_model
from reporting_stats import correlations,paired_comparison,holm
from submit import submit,commands,STAGES

CODE=Path(__file__).resolve().parent/'code'
if not CODE.exists(): CODE=Path(__file__).resolve().parent.parent/'code'
sys.path.insert(0,str(CODE))


class ProtocolTests(unittest.TestCase):
    def panel(self): return [dict(node=n,feature='feature'+str(n),n_observed=30,historical_fold=n) for n in range(36)]
    def test_exact_inventory_and_reuse_of_core_controls(self):
        tasks=make_tasks(self.panel()); self.assertEqual(len(tasks),157)
        self.assertEqual(sum(t['stage']=='core' for t in tasks),57)
        self.assertEqual(sum(t['kind']=='sample' for t in tasks),21)
        for node in range(5):
            matching=[t for t in tasks if t.get('node')==node]
            self.assertEqual({(t['variant'],t['seed']) for t in matching},{(v,s) for v in VARIANTS for s in SEEDS})
        self.assertEqual(len({t['id'] for t in tasks}),157)

    def test_architecture_and_penalty_isolation(self):
        for v in VARIANTS:
            arch,weights=architecture(v)
            self.assertEqual(arch.use_mcr,v!='no_mcr')
            self.assertTrue(arch.use_flux_messages); self.assertTrue(arch.use_mass_balance_projection)
            self.assertEqual(arch.structural_loss_weight,1.)
            self.assertEqual(arch.to_dict()['features']['cross_modal_attention'],v!='no_attention')
            for key,default in [('balance',.5),('correlation',.1),('variance',1.)]:
                self.assertEqual(weights['lambda_'+key],0. if v in ('no_'+key,'no_aux') else default)

    def test_shared_initialization(self):
        import torch
        from src.config import ModelConfig
        models=[]; records=[]
        for v in VARIANTS:
            torch.manual_seed(42)
            cfg=ModelConfig(variant='enhanced',hidden_dim=128,num_layers=3,dropout=.2,enhanced_config=architecture(v)[0].to_dict())
            model,record=matched_model(cfg,42,embedding_dim=16,gene_dim=8,smiles_dim=16)
            models.append(model); records.append(record)
        self.assertEqual(len({r['full_reference_parameter_sha256'] for r in records}),1)
        reference=models[0].state_dict()
        for model in models[1:]:
            for key,value in model.state_dict().items():
                if key in reference and value.shape==reference[key].shape: self.assertTrue(torch.equal(value,reference[key]))
        self.assertEqual(records[0]['model_parameter_sha256'],records[3]['model_parameter_sha256'])
        self.assertNotEqual(records[0]['parameters'],records[1]['parameters'])

    def test_constant_predictions_retained_and_truth_coverage_fixed(self):
        truth=np.arange(12,dtype=float)[:,None]; prediction=np.zeros_like(truth)
        rho,mask,n,constant=correlations(truth,prediction)
        self.assertEqual(rho[0],0); self.assertTrue(mask[0]); self.assertTrue(constant[0])
        truth[:4]=np.nan
        self.assertFalse(correlations(truth,prediction)[1][0])

    def test_paired_statistics_and_order_reversal(self):
        truth=np.arange(24,dtype=float)[:,None]; groups=np.array(['a']*12+['b']*12)
        reference=np.stack([truth]*3); candidate=-reference
        same=paired_comparison(truth,reference,reference,groups,'lomo',repetitions=12)
        self.assertEqual(same['mean_effect'],0); self.assertEqual(same['descriptive_bootstrap_ci95'],[0.,0.])
        a=paired_comparison(truth,reference,candidate,groups,'sample',repetitions=12)
        b=paired_comparison(truth,candidate,reference,groups,'sample',repetitions=12)
        self.assertAlmostEqual(a['mean_effect'],-b['mean_effect'])
        self.assertEqual(a['exact_seed_sign_p'],.25)
        self.assertEqual(holm([.01,.02,.5]),[.03,.04,.5])

    def bundle(self,root):
        save(root/'tasks.json',make_tasks(self.panel()))
        save(root/'research_protocol.json',dict(training_gpu_hour_ceiling=942))
        save(root/'manifest.json',dict(files={p.name:sha(p) for p in root.iterdir() if p.is_file()}))

    def test_dry_run_never_submits_or_connects(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); self.bundle(root)
            with patch('subprocess.run',side_effect=AssertionError('Unexpected process')),patch('subprocess.check_output',side_effect=AssertionError('Unexpected process')),redirect_stdout(io.StringIO()):
                value=submit(root,dry_run=True)
            self.assertEqual(value['total_gpu_hour_ceiling'],943)
            self.assertFalse((root/'submission.json').exists())

    def test_serial_stages_cap_global_concurrency(self):
        ids={s:str(100+i) for i,s in enumerate(STAGES)}
        cmds=commands(Path('/bundle'),'gpu-l4-24g','cpu-zen4','env',8,ids)
        self.assertIn('--array=0-56%8',cmds['core']); self.assertIn('--array=0-99%8',cmds['lomo_ablation'])
        self.assertIn('--dependency=afterok:101,afterany:102',cmds['lomo_ablation'])
        self.assertIn('--dependency=afterany:102:103',cmds['aggregate'])
        for kind in ['core','lomo_ablation']: self.assertIn('--time=06:00:00',cmds[kind])

    def test_resume_reuses_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); self.bundle(root)
            def fake(cmd,**kwargs): return subprocess.CompletedProcess(cmd,0,str(100+len(calls))+'\n','')
            calls=[]
            def tracking(cmd,**kwargs): calls.append(cmd); return fake(cmd,**kwargs)
            with patch('subprocess.check_output',return_value='gpu-l4-24g cpu-zen4 cpu-skylake'),patch('subprocess.run',side_effect=tracking),redirect_stdout(io.StringIO()):
                first=submit(root); second=submit(root)
            self.assertEqual(len(calls),5); self.assertEqual(first,second)

    def test_uncertain_receipt_never_resubmits(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); self.bundle(root)
            with patch('subprocess.check_output',return_value='gpu-l4-24g cpu-zen4 cpu-skylake'),patch('subprocess.run',return_value=subprocess.CompletedProcess([],1,'','connection lost')) as run,redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError): submit(root)
                with self.assertRaises(RuntimeError): submit(root)
                self.assertEqual(run.call_count,1)

    def test_result_verifier_rejects_wrong_truth_mask_and_checkpoint(self):
        from aggregate import verify_task
        from worker import configuration,prediction_layout
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'inputs').mkdir()
            task=make_tasks(self.panel())[0]; output=root/'results'/task['id']; output.mkdir(parents=True)
            for name in ['camp_reanalysis.h5mu','graph.pt','gene_mapping.json','lomo_phase1.csv','lomo_phase2.csv']:
                (root/'inputs'/name).write_text('unit fixture only')
            split=dict(train=[0],val=[1],test=[],holdout=[],excluded=[])
            split2=dict(train=[0,1],val=[],test=[],holdout=[],excluded=[])
            save(root/'inputs/splits.json',dict(lomo={'1':split,'2':split2}))
            np.savez(root/'inputs/assessment.npz',targets=np.array([[1.],[2.]]),sample_ids=np.array(['a','b']),
                     features=np.array(['feature0']),nodes=np.array([0]),groups=np.array(['g','g']))
            save(root/'manifest.json',dict(files={str(p.relative_to(root)):sha(p) for p in (root/'inputs').iterdir()}))
            init=dict(parameters=10,full_reference_parameter_sha256='init',model_parameter_sha256='init',shared_state_entries=1)
            for phase in [1,2]:
                cfg=configuration(root,task,phase,output/f'phase{phase}',epochs=300 if phase==1 else 1)
                save(output/f'phase{phase}/experiment_config.json',cfg.to_dict())
                save(output/f'phase{phase}/sample_indices.json',split if phase==1 else split2)
                save(output/f'phase{phase}/initialization.json',init)
                save(output/f'phase{phase}/training_history.json',dict(train_loss=[.3],val_prediction_mse=[.2],val_loss=[.2]))
                save(output/f'phase{phase}/test_results.json',dict(total_epochs_run=1))
                (output/f'phase{phase}/epoch_diagnostics.csv').write_text('epoch,train_loss\n1,.3\n')
                (output/f'phase{phase}/best_model.pt').write_bytes(b'fixture checkpoint')
            (output/'phase1/best_validation_model.pt').write_bytes(b'fixture checkpoint')
            save(output/'selection.json',dict(selected_epoch=1,selection_mse=.2,objective='prediction_mse',target_exclusions=[0],initialization=init))
            save(output/'claim.json',dict(task=task,smoke=False))
            layout=prediction_layout(root,task)
            prediction=dict(**layout,predictions=np.array([[1.],[2.]]),checkpoint_sha256=np.array(sha(output/'phase2/best_model.pt')))
            def seal():
                artifacts={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and p.name!='status.json'}
                save(output/'status.json',dict(state='complete',task=task,smoke=False,manifest_sha256=sha(root/'manifest.json'),artifacts=artifacts,elapsed_seconds=1.))
            np.savez(output/'predictions.npz',**prediction); seal()
            verify_task(root,task,True)
            changed=dict(prediction,targets=layout['targets']+1)
            np.savez(output/'predictions.npz',**changed); seal()
            with self.assertRaisesRegex(ValueError,'truth mismatch'): verify_task(root,task)
            changed=dict(prediction,checkpoint_sha256=np.array('wrong checkpoint'))
            np.savez(output/'predictions.npz',**changed); seal()
            with self.assertRaisesRegex(ValueError,'checkpoint identity'): verify_task(root,task)
            np.savez(output/'predictions.npz',**prediction)
            cfg=json.loads((output/'phase1/experiment_config.json').read_text()); cfg['data']['excluded_target_nodes']=[]
            save(output/'phase1/experiment_config.json',cfg); seal()
            with self.assertRaisesRegex(ValueError,'configuration mismatch'): verify_task(root,task)


if __name__=='__main__': unittest.main()
