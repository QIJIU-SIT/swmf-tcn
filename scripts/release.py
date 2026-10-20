"""Thin public CLI around the original scientific implementations.

Run from this repository root. No historical checkpoints or approvals are needed
for fresh runs; original models, data processing and trainer loops are retained.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

def read(p): return json.loads(Path(p).read_text('utf-8'))
def write(p, value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,default=str)+'\n',encoding='utf-8')
def matrix(): return read(ROOT/'configs/final_runs.json')['runs']
def select(args):
    rows=matrix()
    if args.experiment=='benchmark': rows=[r for r in rows if r['model'] in ['f4_depth_f4_mid','tcn_mv','shared_gru','lstm_mv','dlinear_mv','patchtst_mv']]
    elif args.experiment=='fusion': rows=[r for r in rows if r['group'] in ['swmf','ordinary']]
    elif args.experiment=='weather': rows=[r for r in rows if r['model'] in ['f4_depth_f4_mid','f4_depth_f4_mid_weather_zeroed']]
    if args.model: rows=[r for r in rows if r['model']==args.model]
    if args.seed: rows=[r for r in rows if r['seed']==args.seed]
    if not rows: raise ValueError('Empty model/seed selection')
    return rows
def configuration():
    import yaml
    return yaml.safe_load((ROOT/'configs/neon_onsite_group_v1.yaml').read_text('utf-8'))
def prepared_identity():
    """Verify paper data bytes independently of changed release path hashes."""
    data=ROOT/configuration()['prepared_root']/'data'
    contract=read(ROOT/'configs/data_contract.json')
    manifest=read(data/'manifest.json')
    keys=read(data/'validation_key_manifest.json')
    if keys['key_hash']!=contract['validation_key_hash'] or keys['rows']!=contract['validation_key_rows']:
        raise ValueError('Validation cohort differs from the final paper contract')
    if hashlib.sha256((data/'scaler.json').read_bytes()).hexdigest()!=contract['scaler_hash']:
        raise ValueError('Training-fitted scaler differs from the final paper contract')
    for name in ['table_hashes','axis_hashes']:
        if manifest[name]!=contract[name]:raise ValueError(f'Prepared {name} differs from the paper contract')
    # Hash actual tables as well as metadata: a stale manifest is insufficient.
    from nerrs_faith.neon_pipeline import dataframe_hash
    import pandas as pd
    for position,expected in contract['table_hashes'].items():
        if dataframe_hash(pd.read_parquet(data/(position+'.parquet')))!=expected:
            raise ValueError('Prepared table bytes differ: '+position)
    digest=hashlib.sha256()
    for name in ['manifest.json','scaler.json','validation_key_manifest.json']:digest.update((data/name).read_bytes())
    return digest.hexdigest()
def scaling(val, scalers):
    return {val.station_ids[p]:(float(scalers[p]['water']['dissolvedOxygen']['mean']),float(scalers[p]['water']['dissolvedOxygen']['scale'])) for p in val.station_ids}
def train(args):
    import torch
    from supporting import neon_training_extension as core
    from supporting import training_extension as v1, training_extension_v2 as v2
    from nerrs_faith.hslf_training import HSLFTrainer,make_hslf_loader
    from nerrs_faith.neon_onsite_experiment import StationBalancedBatchSampler,resolve_onsite_loader_config
    if args.device!='cuda': raise ValueError('Formal training retains CUDA/AMP policy; CPU is for check_release.py only')
    if not torch.cuda.is_available(): raise RuntimeError('Formal training requires CUDA')
    data_hash=prepared_identity()
    ds,val,scalers=core.trainval_data()
    common=resolve_onsite_loader_config(configuration(),device='cuda')
    for row in select(args):
        run=ROOT/'outputs/release_training'/row['model']/f"seed-{row['seed']}"
        if run.exists(): raise FileExistsError(f'Fresh run only; preserve existing run: {run}')
        core.seed_all(row['seed'])
        cfg=dict(row['training'])
        if row['version'] in ['v1','v2']:
            tl,vl=core.loaders_for(row,ds,val)
            trainer_type=v1.ExtensionTrainer if row['version']=='v1' else v2.ExtensionTrainer
        else:
            if row['model']=='tcn_mv':
                tl=make_hslf_loader(ds,batch_sampler=StationBalancedBatchSampler(ds,512,seed=row['seed']),device='cuda',**common)
            else:
                gen=torch.Generator().manual_seed(row['seed'])
                tl=make_hslf_loader(ds,batch_size=cfg['batch_size'],shuffle=True,generator=gen,device='cuda',**common)
            vl=make_hslf_loader(val,batch_size=cfg['batch_size'],shuffle=False,device='cuda',**common)
            trainer_type=HSLFTrainer
        model=core.build_model(row)
        # New release provenance hashes bind local prepared files. Historical
        # author path/config fingerprints cannot be reused after extraction.
        run.mkdir(parents=True)
        write(run/'release_run.json',dict(model=row['model'],seed=row['seed'],version=row['version'],training=cfg,data_hash=data_hash,source='configs/final_runs.json'))
        trainer=trainer_type(model,tl,vl,run_dir=run,data_hash=data_hash,config=cfg,device='cuda',physical_scaling_stats=scaling(val,scalers),primary_metric_name='trajectory_physical_position_macro_mae',selection_scope='trajectory')
        try: trainer.fit()
        finally:
            for loader in [tl,vl]:
                it=getattr(loader,'_iterator',None)
                if it is not None:it._shutdown_workers()

def predict(args):
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from supporting import neon_training_extension as core
    from nerrs_faith.hslf_training import collect_predictions,make_hslf_loader
    from nerrs_faith.neon_onsite_expanded_test import _prepare_test_dataset
    from supporting.test_2024_export import prepare_export_frame
    cfg=configuration()
    data_hash=prepared_identity()
    if args.split=='val': _,ds,scalers=core.trainval_data()
    else:
        # 2024 is forward-only; training CLI never constructs this dataset.
        context=_prepare_test_dataset(cfg,repo_root=ROOT);ds=context['dataset'];scalers=context['scalers']
        if len(ds)!=141358:raise ValueError('2024 cohort differs from the final paper contract')
    metadata=dict(station_ids=ds.station_ids,reserve_ids=ds.reserve_ids,scaling_stats={p:{'water':{'mean':{'DO_mgl':float(scalers[p]['water']['dissolvedOxygen']['mean'])},'scale':{'DO_mgl':float(scalers[p]['water']['dissolvedOxygen']['scale'])}}} for p in ds.station_ids})
    selected=select(args)
    if args.split=='test' and any(r['group']!='baseline' and r['model']!='f4_depth_f4_mid' for r in selected):
        raise ValueError('Fusion/weather controls are validation-only')
    reference=None
    for ri,row in enumerate(selected):
        run=ROOT/'outputs/release_training'/row['model']/f"seed-{row['seed']}"
        ck=read(run/'release_run.json')
        if ck['model']!=row['model'] or ck['seed']!=row['seed'] or ck['training']!=row['training'] or ck['data_hash']!=data_hash:
            raise ValueError('Checkpoint selection/configuration mismatch')
        model=core.build_model(row)
        saved=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
        if saved['config']!=ck['training'] or saved['data_hash']!=ck['data_hash'] or saved['selection_scope']!='trajectory':
            raise ValueError('Checkpoint provenance mismatch')
        model.load_state_dict(saved['model_state'],strict=True)
        destination=ROOT/'outputs/release_predictions'/args.split/row['model']/f"seed-{row['seed']}.parquet"
        if destination.exists():raise FileExistsError(destination)
        destination.parent.mkdir(parents=True,exist_ok=True)
        loader=make_hslf_loader(ds,batch_size=row['training']['batch_size'],shuffle=False,device=args.device,num_workers=0)
        baseline_dest=ROOT/'outputs/release_predictions'/args.split/'baselines'
        # Deterministic references share exactly the same target cohort.
        writers={}; digest=hashlib.sha256()
        try:
            for batch in loader:
                frame=collect_predictions(model,[batch],device=args.device,metadata={**metadata,'seed':row['seed'],'variant':row['model']},split=args.split)
                frame['aquatic_site']=frame.station.str.split('-',n=1).str[0]
                frame=prepare_export_frame(frame)
                frame['horizon']=frame.horizon.astype('int64')
                digest.update(pd.util.hash_pandas_object(frame[['station','origin','target_time','horizon','target_mask','target']],index=False).to_numpy().tobytes())
                exports={destination:frame}
                for path,part in exports.items():
                    table=pa.Table.from_pandas(part,preserve_index=False)
                    if path not in writers:
                        path.parent.mkdir(parents=True,exist_ok=True)
                        writers[path]=pq.ParquetWriter(path.with_suffix('.tmp.parquet'),table.schema,compression='zstd')
                    writers[path].write_table(table)
        finally:
            for w in writers.values():w.close()
        if reference is not None and digest.hexdigest()!=reference:raise ValueError('Prediction cohort differs across runs')
        reference=digest.hexdigest()
        for path in writers:path.with_suffix('.tmp.parquet').replace(path)
        write(destination.with_suffix('.json'),dict(model=row['model'],seed=row['seed'],split=args.split,checkpoint_sha256=hashlib.sha256((run/'best.pt').read_bytes()).hexdigest(),target_key_mask_sha256=reference))
    # Use the actual physical-unit baseline exporter used in final evaluation.
    from nerrs_faith.neon_baselines import iter_dataset_baseline_predictions
    baseline_writers={}
    try:
        for raw in iter_dataset_baseline_predictions(ds):
            for name in ['persistence','seasonal_24h_with_persistence_fallback']:
                path=baseline_dest/(name+'.parquet')
                if path.exists():continue
                frame=raw.copy()
                frame['prediction']=raw['persistence'] if name=='persistence' else raw['prediction']
                frame['origin']=pd.to_datetime(frame['origin'],utc=True)
                frame['target_time']=pd.to_datetime(frame['target_time'],utc=True)
                frame=prepare_export_frame(frame)
                frame['horizon']=frame.horizon.astype('int64')
                table=pa.Table.from_pandas(frame,preserve_index=False)
                if path not in baseline_writers:
                    path.parent.mkdir(parents=True,exist_ok=True)
                    baseline_writers[path]=pq.ParquetWriter(path.with_suffix('.tmp.parquet'),table.schema,compression='zstd')
                baseline_writers[path].write_table(table)
    finally:
        for w in baseline_writers.values():w.close()
    for path in baseline_writers:path.with_suffix('.tmp.parquet').replace(path)

def evaluate(args):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from supporting.final_metrics import accumulate,metrics
    from analysis.neon_onsite_2024.finalize import HORIZON_BANDS
    paths=sorted(Path(args.prediction_root).rglob('*.parquet'))
    if not paths:raise FileNotFoundError('No generated prediction Parquets')
    rows=[];position_rows=[];site_rows=[]
    for path in paths:
        positions=sorted(read(ROOT/'configs/data_contract.json')['table_hashes'])
        states=np.zeros((7,len(positions),96),dtype=np.float64)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=262144):accumulate(batch.to_pandas(),{p:i for i,p in enumerate(positions)},states)
        identity=dict(model=path.parent.name if path.parent.name!='baselines' else path.stem,seed=int(path.stem.split('-')[-1]) if path.stem.startswith('seed-') else None)
        for scope,steps in [('all',slice(None)),*[(k,slice(lo-1,hi)) for k,(lo,hi) in HORIZON_BANDS.items()],*[(f'step_{i}',slice(i-1,i)) for i in range(1,97)]]:
            rows.append(dict(**identity,scope=scope,**metrics(states[:,:,steps])))
            for i,position in enumerate(positions):
                position_rows.append(dict(**identity,scope=scope,station=position,**metrics(states[:,i:i+1,steps])))
            for site in sorted({p.split('-',1)[0] for p in positions}):
                indices=[i for i,p in enumerate(positions) if p.startswith(site+'-')]
                site_rows.append(dict(**identity,scope=scope,site=site,**metrics(states[:,indices,:][:,:,steps])))
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows);frame.to_csv(output/'metrics_by_seed.csv',index=False)
    frame.groupby(['model','scope'])[['mae','rmse','r2','mape']].agg(['mean','std']).to_csv(output/'metrics_seed_summary.csv')
    for level,records,key in [('position',position_rows,'station'),('site',site_rows,'site')]:
        frame=pd.DataFrame(records);frame.to_csv(output/f'metrics_by_{level}.csv',index=False)
        frame.groupby(['model','scope',key])[['mae','rmse','r2','mape']].agg(['mean','std']).to_csv(output/f'metrics_by_{level}_seed_summary.csv')

def bootstrap(args):
    from supporting.test_2024_pairing import paired_block_bootstrap
    result=paired_block_bootstrap(args.candidate,args.comparator,seed=args.seed,resamples=10000)
    write(args.output,result)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--config',default='configs/neon_onsite_group_v1.yaml')
    for name in ['plan','train','predict']:
        p=sub.add_parser(name);p.add_argument('--experiment',choices=['benchmark','fusion','weather','all'],default='benchmark');p.add_argument('--model');p.add_argument('--seed',type=int,choices=[2026,2027,2028]);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
        if name=='predict':p.add_argument('--split',choices=['val','test'],required=True)
    p=sub.add_parser('evaluate');p.add_argument('--prediction-root',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('bootstrap');p.add_argument('--candidate',required=True);p.add_argument('--comparator',required=True);p.add_argument('--seed',type=int,choices=[2026,2027,2028],required=True);p.add_argument('--output',required=True)
    args=parser.parse_args()
    if Path.cwd().resolve()!=ROOT:raise ValueError('Run from the swmf-tcn repository root')
    if args.command=='prepare':
        from scripts.build_neon_onsite_group import main as prepare
        sys.argv=[sys.argv[0],'--config',args.config];return prepare()
    if args.command=='plan':print(json.dumps(select(args),indent=2));return 0
    globals()[args.command](args)
    return 0
if __name__=='__main__':raise SystemExit(main())
