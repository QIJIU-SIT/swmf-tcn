"""Mask-aware pairing only; historical scientific bootstrap kernel is unchanged."""
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.neon_onsite_2024.finalize import PAIR_KEY_COLUMNS,_iter_batches,_bootstrap

# These are the unchanged numpy.allclose defaults used by the historical pairing.
RTOL=1e-5
ATOL=1e-8

def pair_batches(candidate,comparator,batch_rows=262144,*,audit=None):
    audit={} if audit is None else audit
    for k in ['rows','valid_rows','invalid_rows','invalid_representation_mismatches','left_invalid_nan','right_invalid_nan','left_invalid_finite','right_invalid_finite','left_invalid_inf','right_invalid_inf']:
        audit.setdefault(k,0)
    audit.setdefault('valid_max_absolute_difference',0.)
    audit.update(rtol=RTOL,atol=ATOL,candidate_source=str(candidate),comparator_source=str(comparator))
    cols=[*PAIR_KEY_COLUMNS,'target_mask','target','prediction']
    previous=None
    streams=[_iter_batches(Path(p),cols,batch_rows) for p in [candidate,comparator]]
    for left,right in zip_longest(*streams):
        if left is None or right is None:raise ValueError('paired row counts differ')
        a,b=left.to_pandas(),right.to_pandas()
        for frame in [a,b]:
            for c in ['origin','target_time']:frame[c]=pd.to_datetime(frame[c],utc=True)
            if frame.target_mask.isna().any() or not frame.target_mask.isin([True,False]).all():raise ValueError('invalid target mask')
        if not a[PAIR_KEY_COLUMNS+['target_mask']].reset_index(drop=True).equals(b[PAIR_KEY_COLUMNS+['target_mask']].reset_index(drop=True)):
            raise ValueError('paired keys or masks differ')
        order=list(a[PAIR_KEY_COLUMNS].itertuples(index=False,name=None))
        if any(x>=y for x,y in zip(order,order[1:])) or (order and previous is not None and previous>=order[0]):raise ValueError('duplicate or unordered paired keys')
        if order:previous=order[-1]
        mask=a.target_mask.to_numpy(bool);x=a.target.to_numpy(float);y=b.target.to_numpy(float)
        if not np.isfinite(x[mask]).all() or not np.isfinite(y[mask]).all():raise ValueError('nonfinite valid target')
        if not np.allclose(x[mask],y[mask],rtol=RTOL,atol=ATOL,equal_nan=True):raise ValueError('valid paired target values differ')
        audit['rows']+=len(a);audit['valid_rows']+=int(mask.sum());audit['invalid_rows']+=int((~mask).sum())
        if mask.any():audit['valid_max_absolute_difference']=max(audit['valid_max_absolute_difference'],float(np.max(np.abs(x[mask]-y[mask]))))
        audit['invalid_representation_mismatches']+=int((~np.isclose(x[~mask],y[~mask],rtol=RTOL,atol=ATOL,equal_nan=True)).sum())
        for name,values in [('left',x[~mask]),('right',y[~mask])]:
            for typ,predicate in [('nan',np.isnan),('finite',np.isfinite),('inf',np.isinf)]:audit[name+'_invalid_'+typ]+=int(predicate(values).sum())
        yield a,b

def paired_block_bootstrap(candidate_path,comparator_path,*,seed,resamples=10000,batch_rows=262144,statistic='f4_mae_minus_tcn_mv_mae',audit=None):
    # Same block construction/order/count weights as historical finalize.py.
    # Only its all-row target-equality guard is replaced by pair_batches above.
    block_sums=defaultdict(dict)
    for candidate,comparator in pair_batches(candidate_path,comparator_path,batch_rows,audit=audit):
        valid=candidate.target_mask.astype(bool)
        candidate=candidate.loc[valid].copy();comparator=comparator.loc[valid].copy()
        if candidate.empty:continue
        if not np.isfinite(candidate[['target','prediction']].to_numpy(float)).all() or not np.isfinite(comparator.prediction.to_numpy(float)).all():raise ValueError('nonfinite valid prediction')
        work=pd.DataFrame(dict(station=candidate.station.astype(str).to_numpy(),origin=pd.to_datetime(candidate.origin.to_numpy(),utc=True),
          difference=np.abs(candidate.prediction.to_numpy(float)-candidate.target.to_numpy(float))-np.abs(comparator.prediction.to_numpy(float)-comparator.target.to_numpy(float))))
        work['date']=work.origin.dt.strftime('%Y-%m-%d');work['block']=work.station+'|'+work.date
        work['half']=np.where(work.origin.dt.month.to_numpy()<=6,'H1','H2')
        for scope,scoped in [('annual',work),('H1',work.loc[work.half=='H1']),('H2',work.loc[work.half=='H2'])]:
            if scoped.empty:continue
            grouped=scoped.groupby(['station','block'],sort=False)['difference'].agg(['sum','count'])
            for (station,block),r in grouped.iterrows():
                item=block_sums[scope].setdefault((str(station),str(block)),[0.,0.]);item[0]+=float(r['sum']);item[1]+=float(r['count'])
    result={}
    for scope in ['annual','H1','H2']:
        blocks=pd.DataFrame([dict(station=s,block=b,sum=v[0],count=v[1]) for (s,b),v in block_sums[scope].items()])
        result[scope]=_bootstrap(blocks,training_seed=seed,resamples=resamples,statistic=statistic)
    return result
