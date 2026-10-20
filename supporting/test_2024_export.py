"""Validated export adapter; derive site exactly as the original exporter."""
import numpy as np
import pandas as pd

POSITIONS={f'{s}-{p}.100.100' for s in ['CUPE','SYCA','WLOU'] for p in [101,102]}
COLS=['aquatic_site','station','origin','target_time','horizon','target_mask','target','prediction']

def prepare_export_frame(frame):
    if not isinstance(frame,pd.DataFrame) or frame.empty:raise ValueError('empty or invalid prediction frame')
    required=set(COLS)-{'aquatic_site'}
    if not required.issubset(frame.columns):raise ValueError('missing prediction columns: '+str(sorted(required-set(frame.columns))))
    if not frame.station.map(lambda x:isinstance(x,str) and x in POSITIONS).all():raise ValueError('unknown or non-string monitoring position')
    # Exactly neon_onsite_expanded_test.execute_test_job's conversion, after
    # validating values so astype(str) cannot hide numeric/null station IDs.
    derived=frame['station'].astype(str).str.split('-',n=1).str[0]
    if 'aquatic_site' in frame and not frame['aquatic_site'].equals(derived):raise ValueError('aquatic_site conflicts with original station-prefix rule')
    if not pd.api.types.is_bool_dtype(frame.target_mask.dtype) or frame.target_mask.isna().any():raise ValueError('target_mask must be boolean')
    if not pd.api.types.is_integer_dtype(frame.horizon.dtype) or not frame.horizon.between(1,96).all():raise ValueError('invalid horizon type or value')
    for c in ['origin','target_time']:
        if not isinstance(frame[c].dtype,pd.DatetimeTZDtype) or frame[c].isna().any():raise ValueError('invalid timestamp contract')
    for c in ['target','prediction']:
        if not pd.api.types.is_numeric_dtype(frame[c].dtype) or pd.api.types.is_bool_dtype(frame[c].dtype):raise ValueError('target/prediction must be numeric')
    if not np.isfinite(frame.loc[frame.target_mask,['target','prediction']].to_numpy(float)).all():raise ValueError('nonfinite valid target or prediction')
    if frame.duplicated(['station','origin','target_time','horizon']).any():raise ValueError('duplicate target key')
    result=frame.copy();result['aquatic_site']=derived
    return result[COLS]
