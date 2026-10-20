"""Fast CPU checks with synthetic data only; never trains or opens NEON data."""
from pathlib import Path
import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import shutil

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import numpy as np
import pandas as pd
import torch
import yaml
torch.set_num_threads(1)

def synthetic_tables():
    from nerrs_faith.neon_data import WATER,MET
    dates=pd.date_range('2023-01-01',periods=360,freq='15min',tz='UTC')
    tables={};scalers={}
    for site in ['CUPE','SYCA','WLOU']:
        for pos in [101,102]:
            name=f'{site}-{pos}.100.100'
            frame=pd.DataFrame(dict(date=dates,split='val'))
            for i,col in enumerate(WATER+MET):frame[col]=5+i+np.arange(360)*.001;frame[col+'_observed']=True
            frame.loc[260,'pH_observed']=False
            frame.loc[270,'dissolvedOxygen_observed']=False
            tables[name]=frame
            scalers[name]={'water':{c:dict(mean=4.,scale=2.) for c in WATER},'meteorology':{c:dict(mean=0.,scale=1.) for c in MET}}
    return tables,scalers
def dataset():
    from nerrs_faith.neon_data import NeonWindowDataset
    t,s=synthetic_tables()
    return NeonWindowDataset(t,split='val',scaling_stats=s),s
def batch():
    from nerrs_faith.hslf_data import hslf_collate
    ds,_=dataset()
    return hslf_collate([ds[0],ds[9]])

class ReleaseChecks(unittest.TestCase):
    def test_all_local_imports(self):
        for directory,base in [(ROOT/'src/nerrs_faith',ROOT/'src'),(ROOT/'supporting',ROOT),(ROOT/'analysis',ROOT)]:
            for p in directory.rglob('*.py'):
                name='.'.join(p.relative_to(base).with_suffix('').parts)
                if name.endswith('.__init__'):name=name[:-9]
                module=importlib.import_module(name)
                self.assertTrue(Path(module.__file__).resolve().is_relative_to(ROOT))

    def test_frozen_matrix(self):
        rows=json.loads((ROOT/'configs/final_runs.json').read_text())['runs']
        self.assertEqual(len(rows),36)
        self.assertEqual(len({(r['model'],r['seed']) for r in rows}),36)
        self.assertEqual({r['seed'] for r in rows},{2026,2027,2028})
        for r in rows:
            self.assertEqual(r['training']['max_epochs'],dict(v1=15,v2=25,original_frozen=5)[r['version']])
        cfg=yaml.safe_load((ROOT/'configs/neon_onsite_group_v1.yaml').read_text())
        self.assertEqual(cfg['window'],dict(seq_len=256,pred_len=96))
        self.assertEqual(cfg['splits']['train_start'],'2019-04-01T00:00:00Z')
        self.assertEqual(cfg['splits']['train_end'],'2022-12-31T23:59:59Z')
        self.assertEqual(cfg['splits']['val_start'][:4],'2023')
        self.assertEqual(cfg['splits']['test_start'][:4],'2024')

    def test_models_and_shapes(self):
        from supporting.neon_training_extension import build_model,seed_all
        rows=json.loads((ROOT/'configs/final_runs.json').read_text())['runs']
        b=batch()
        for r in rows:
            if r['seed']!=2026:continue
            seed_all(2026);model=build_model(r).eval()
            with torch.no_grad():y,_=model(b)
            self.assertEqual(tuple(y.shape),(2,96),r['model'])
            self.assertTrue(torch.isfinite(y).all(),r['model'])
            if r['group'] in ['swmf','zeroed']:self.assertEqual(sum(p.numel() for p in model.parameters()),41080)

    def test_fixed_masks_and_causality(self):
        from nerrs_faith.benchmark.models.f4_depth_double_res import build_f4_depth_double_res
        from nerrs_faith.benchmark.models.f4_tcn_factorial import build_f4_tcn_factorial
        torch.manual_seed(2026);model=build_f4_depth_double_res('f4_depth_f4_mid',station_count=6)
        self.assertEqual([b.cross_group for b in model.blocks],[False]*4+[True]*2)
        self.assertEqual([b.dilation for b in model.blocks],[1,2,4,8,16,32])
        self.assertEqual(model.station_embedding.embedding_dim,4)
        self.assertEqual(len(model.blocks),6)
        for b in model.blocks:
            self.assertTrue(torch.equal(b.block_mask[:16,16:],torch.zeros(16,16,1)))
            self.assertEqual(len(b.convs),2)
        x=torch.randn(2,32,256);z=x.clone();z[:,16:]+=10
        a=model.blocks[0](x);c=model.blocks[0](z)
        self.assertTrue(torch.equal(a[:,:16],c[:,:16]))
        z=x.clone();z[:,:,101:]+=10
        self.assertTrue(torch.equal(model.blocks[4](x)[:,:,:101],model.blocks[4](z)[:,:,:101]))
        model(batch()).sum().backward()
        for b in model.blocks[:4]:
            for conv in b.convs:self.assertTrue(torch.equal(conv.weight.grad[:16,16:],torch.zeros_like(conv.weight.grad[:16,16:])))
        torch.manual_seed(2026);old=build_f4_tcn_factorial('f4_tcn_double_res',station_count=6)
        self.assertEqual(set(model.state_dict()),set(old.state_dict()))
        for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,old.state_dict()[k]),k)

    def test_weather_zeroes_values_and_masks(self):
        from nerrs_faith.neon_fusion_depth import perturb_batch
        b=batch();c=perturb_batch(b,'met_zero',seed=0,block_steps=96)
        self.assertEqual(torch.count_nonzero(c['met']).item(),0)
        self.assertEqual(torch.count_nonzero(c['met_mask']).item(),0)
        for k in ['water','water_mask','target','target_mask']:self.assertTrue(torch.equal(b[k],c[k]))

    def test_quality_aggregation_and_scaler(self):
        from nerrs_faith.neon_pipeline import aggregate_15min,fit_training_scalers
        frame=pd.DataFrame(dict(startDateTime=pd.date_range('2023-01-01 00:01',periods=30,freq='min',tz='UTC'),v=2.,q=0))
        frame.loc[[0,1,2],'q']=1;frame.loc[[15,16,17,18],'q']=1
        avg=aggregate_15min(frame,'v','q');total=aggregate_15min(frame,'v','q',method='sum')
        self.assertEqual(avg.valid_count.tolist(),[12,11]);self.assertEqual(avg.observed.tolist(),[True,False])
        self.assertEqual(avg.value.iloc[0],2);self.assertTrue(np.isnan(avg.value.iloc[1]))
        self.assertEqual(total.value.iloc[0],24)
        t=pd.DataFrame(dict(v=[1.,3.,1000.],v_observed=[True]*3,split=['train','train','val']),index=pd.to_datetime(['2020-01-01','2021-01-01','2023-01-01'],utc=True))
        self.assertEqual(fit_training_scalers(t,['v'],train_end='2022-12-31T00:00:00Z')['v'],dict(mean=2.,scale=1.))

    def test_windows_reconstruction_and_baselines(self):
        from nerrs_faith.neon_baselines import iter_dataset_baseline_predictions
        from nerrs_faith.hslf_training import collect_predictions
        ds,scalers=dataset();item=ds[0]
        self.assertEqual(tuple(item['water'].shape),(256,4));self.assertEqual(tuple(item['target'].shape),(96,))
        self.assertEqual(item['target_mask'].sum().item(),95)
        self.assertTrue(torch.isclose(item['target'][0],torch.tensor(.0005),atol=1e-6))
        self.assertEqual(ds.station_ids['CUPE-101.100.100'],1)
        b=batch()
        class Zero(torch.nn.Module):
            def forward(self,b):return torch.zeros_like(b['target']),{}
        metadata=dict(station_ids=ds.station_ids,reserve_ids=ds.reserve_ids,scaling_stats={p:{'water':{'mean':{'DO_mgl':4.},'scale':{'DO_mgl':2.}}} for p in ds.station_ids})
        pred=collect_predictions(Zero(),[b],metadata=metadata)
        self.assertTrue(np.allclose(pred.prediction.iloc[:96],float(b['last_do_physical'][0]),atol=1e-6))
        base=next(iter_dataset_baseline_predictions(ds,chunk_origins=2))
        self.assertTrue(np.allclose(base.prediction,base.seasonal_persistence))
        self.assertTrue((pd.to_datetime(base.target_time,utc=True)-pd.to_datetime(base.origin,utc=True)).dt.total_seconds().eq(base.horizon.astype('int64')*900).all())

    def test_formal_metrics(self):
        from supporting.final_metrics import accumulate,metrics
        f=pd.DataFrame(dict(station=['a','b','b'],horizon=[1,1,1],target=[0.,2.,4.],prediction=[1.,4.,6.],target_mask=True))
        stats=np.zeros((7,2,96));accumulate(f,dict(a=0,b=1),stats);m=metrics(stats[:,:,:1])
        self.assertAlmostEqual(m['mae'],1.5);self.assertAlmostEqual(m['rmse'],1.5)
        self.assertAlmostEqual(m['r2'],1-9/8);self.assertAlmostEqual(m['mape'],75.)
        self.assertEqual(m['mape_excluded'],1)

    def test_paired_bootstrap_and_invalid_rows(self):
        from supporting.test_2024_pairing import paired_block_bootstrap
        frames=[]
        for station in ['CUPE-101.100.100','SYCA-101.100.100']:
            for month in [1,7]:
                for day in [1,2]:
                    origin=pd.Timestamp(f'2024-{month:02}-{day:02}',tz='UTC')
                    frames.append(dict(aquatic_site=station[:4],station=station,origin=origin,target_time=origin+pd.Timedelta(minutes=15),horizon=1,target_mask=True,target=2.,prediction=3.))
        left=pd.DataFrame(frames).sort_values(['aquatic_site','station','origin','target_time','horizon']);right=left.copy();right['prediction']=4.
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            a=Path(td)/'a.parquet';b=Path(td)/'b.parquet';left.to_parquet(a,index=False);right.to_parquet(b,index=False)
            result=paired_block_bootstrap(a,b,seed=2026)
            for k in ['annual','H1','H2']:
                self.assertEqual(result[k]['estimate'],-1.);self.assertEqual(result[k]['lower'],-1.);self.assertEqual(result[k]['upper'],-1.)
                self.assertEqual(result[k]['bootstrap_seed'],4052)
            right.iloc[0,right.columns.get_loc('target_mask')]=False;right.to_parquet(b,index=False)
            with self.assertRaises(ValueError):paired_block_bootstrap(a,b,seed=2026)

    def test_stopping_rules(self):
        from supporting.training_extension import StopController as V1
        from supporting.training_extension_v2 import StopController as V2
        for cls,cap in [(V1,15),(V2,25)]:
            control=cls();reason=None
            for epoch in range(1,9):reason=control.observe(epoch,1.,False)
            self.assertEqual(reason,'EARLY_STOPPED')
            control=cls()
            for epoch in range(1,cap+1):reason=control.observe(epoch,1./epoch,False)
            self.assertEqual(reason,'BUDGET_CAPPED_UNRESOLVED')

    def test_cli_and_configuration(self):
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(ROOT/'src')}
        for name in ['prepare','plan','train','predict','evaluate','bootstrap']:
            p=subprocess.run([sys.executable,'scripts/release.py',name,'--help'],cwd=ROOT,env=env,capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
        p=subprocess.run([sys.executable,'scripts/release.py','plan','--experiment','all'],cwd=ROOT,env=env,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(len(json.loads(p.stdout)),36)

    def test_cli_metrics_integration(self):
        from nerrs_faith.neon_baselines import iter_dataset_baseline_predictions
        ds,_=dataset()
        frame=pd.concat(list(iter_dataset_baseline_predictions(ds,chunk_origins=2)),ignore_index=True)
        frame['horizon']=frame.horizon.astype('int64')
        frame['prediction']=frame['target']+np.where(frame.station.str.startswith('CUPE'),1.,2.)
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            p=Path(td)/'predictions'/'synthetic_model';p.mkdir(parents=True)
            frame.to_parquet(p/'seed-2026.parquet',index=False)
            out=Path(td)/'metrics'
            cmd=[sys.executable,'scripts/release.py','evaluate','--prediction-root',str(p.parent),'--output',str(out)]
            result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            metrics=pd.read_csv(out/'metrics_by_seed.csv');overall=metrics.loc[metrics.scope=='all'].iloc[0]
            self.assertAlmostEqual(overall.mae,5/3);self.assertAlmostEqual(overall.rmse,5/3)
            self.assertEqual(len(metrics),101)
            self.assertTrue((out/'metrics_seed_summary.csv').is_file())
            position=pd.read_csv(out/'metrics_by_position.csv')
            site=pd.read_csv(out/'metrics_by_site.csv')
            self.assertEqual(len(position),606);self.assertEqual(len(site),303)
            self.assertEqual(site.loc[(site.site=='CUPE') & (site.scope=='all'),'mae'].iloc[0],1.)

    def test_trainer_configuration_without_training(self):
        from supporting.neon_training_extension import build_model
        from nerrs_faith.hslf_training import HSLFTrainer,make_hslf_loader
        from supporting.training_extension import ExtensionTrainer as V1
        from supporting.training_extension_v2 import ExtensionTrainer as V2
        ds,_=dataset();loader=make_hslf_loader(ds,batch_size=2,device='cpu')
        rows=json.loads((ROOT/'configs/final_runs.json').read_text())['runs']
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            for row in rows:
                if row['seed']!=2026:continue
                trainer=dict(v1=V1,v2=V2,original_frozen=HSLFTrainer)[row['version']](build_model(row),loader,loader,run_dir=Path(td)/row['model'],data_hash='synthetic-only',config=row['training'],device='cpu',physical_scaling_stats={i:(4.,2.) for i in ds.station_ids.values()},primary_metric_name='trajectory_physical_position_macro_mae',selection_scope='trajectory')
                self.assertEqual(trainer.config,row['training'])
                self.assertEqual(trainer.selection_scope,'trajectory')
                self.assertEqual(trainer.scheduler.factor,.5)
                self.assertEqual(trainer.scheduler.patience,2)

    def test_prediction_adapter_with_synthetic_checkpoint(self):
        # Controlled orchestration check only. No optimizer step or real data.
        from supporting import neon_training_extension as core
        from nerrs_faith.neon_baselines import iter_dataset_baseline_predictions
        import importlib.util
        spec=importlib.util.spec_from_file_location('release_cli_under_test',ROOT/'scripts/release.py')
        cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        ds,scalers=dataset()
        row=next(r for r in cli.matrix() if r['model']=='f4_depth_f4_mid' and r['seed']==2026)
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            root=Path(td);shutil.copytree(ROOT/'configs',root/'configs')
            run=root/'outputs/release_training/f4_depth_f4_mid/seed-2026';run.mkdir(parents=True)
            record=dict(model=row['model'],seed=2026,training=row['training'],data_hash='synthetic-only')
            (run/'release_run.json').write_text(json.dumps(record),encoding='utf-8')
            torch.save(dict(model_state=core.build_model(row).state_dict(),config=row['training'],data_hash='synthetic-only',selection_scope='trajectory'),run/'best.pt')
            with patch.object(cli,'ROOT',root),patch.object(cli,'prepared_identity',return_value='synthetic-only'),patch.object(core,'trainval_data',return_value=(ds,ds,scalers)):
                cli.predict(SimpleNamespace(experiment='benchmark',model=row['model'],seed=2026,split='val',device='cpu'))
            pred=root/'outputs/release_predictions/val'
            learned=pd.read_parquet(pred/'f4_depth_f4_mid/seed-2026.parquet')
            seasonal=pd.read_parquet(pred/'baselines/seasonal_24h_with_persistence_fallback.parquet')
            persistence=pd.read_parquet(pred/'baselines/persistence.parquet')
            expected=pd.concat(list(iter_dataset_baseline_predictions(ds)),ignore_index=True)
            self.assertEqual(len(learned),len(ds)*96)
            self.assertTrue(np.allclose(seasonal.prediction,expected.prediction))
            self.assertTrue(np.allclose(persistence.prediction,expected.persistence))
            self.assertTrue(learned[['station','origin','target_time','horizon','target_mask']].equals(seasonal[['station','origin','target_time','horizon','target_mask']]))

if __name__=='__main__':unittest.main(verbosity=2)
