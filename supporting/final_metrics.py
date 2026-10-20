"""Selected original definitions; see SOURCE_PROVENANCE.json."""
import numpy as np

def accumulate(frame,positions,stats):
    v=frame.target_mask.to_numpy(bool); y=frame.target.to_numpy(float); pred=frame.prediction.to_numpy(float)
    assert np.isfinite(y[v]).all() and np.isfinite(pred[v]).all()
    p=frame.station.map(positions).to_numpy(int)[v]; h=frame.horizon.to_numpy(int)[v]-1
    y=y[v];e=pred[v]-y;nz=y!=0;ape=np.zeros(len(y));ape[nz]=np.abs(e[nz]/y[nz])
    for a,x in zip(stats,[np.abs(e),e*e,np.ones(len(y)),y,y*y,ape,nz.astype(float)]):
        np.add.at(a,(p,h),x)

def metrics(s):
    a,q,n,y,y2,ape,an=s.sum(axis=2)
    N=n.sum(); variance=y2.sum()-y.sum()**2/N
    return dict(mae=float(np.mean(a/n)),rmse=float(np.mean(np.sqrt(q/n))),
      mape=float(100*ape.sum()/an.sum()) if an.sum() else float('nan'),
      r2=float(1-q.sum()/variance) if variance>0 else float('nan'),valid_targets=int(N),mape_excluded=int(N-an.sum()))
