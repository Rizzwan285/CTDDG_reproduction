"""Audit RF_Nitesh.joblib probabilities without modifying the classifier or outputs.

Example on Bhavani:
 python RF_Nitesh_diagnostic.py \
   --rf /home/muhamed/repo/CTDDG/REWORKED_CODE/Random_Forest/RF_Nitesh.joblib \
   --csv /path/to/generated_run1.csv

WARNING: joblib uses pickle. Load only the RF artifact that you trained/trust.
"""
import argparse
import csv
import json
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import sklearn


def make_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Cannot parse SMILES: {smiles!r}')
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    vec = np.zeros((2048,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, vec)
    return vec


def main():
    parser = argparse.ArgumentParser(description='Audit Ames model predict_proba on representative molecules.')
    parser.add_argument('--rf', type=Path, required=True)
    parser.add_argument('--csv', type=Path, default=None,
                        help='Previously generated_run1.csv or all_generated_attempts.csv')
    parser.add_argument('--output', type=Path, default=Path('rf_nitesh_audit.json'))
    args = parser.parse_args()
    print('scikit-learn runtime:', sklearn.__version__)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        predictor = joblib.load(args.rf)  # TRUSTED LOCAL FILE ONLY
    print('Estimator type:', type(predictor).__module__ + '.' + type(predictor).__name__)
    for w in captured:
        print('LOAD WARNING:', str(w.message))
    print('classes_:', repr(getattr(predictor, 'classes_', None)))
    print('n_features_in_:', repr(getattr(predictor, 'n_features_in_', None)))

    smiles = ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCN']
    if args.csv:
        with args.csv.open(newline='',encoding='utf-8') as f:
            rows=list(csv.DictReader(f))
        smiles=[r['smiles'] for r in rows if r.get('smiles') and str(r.get('valid')) in ('1','True','true')]
    X=np.asarray([make_fingerprint(s) for s in smiles],dtype=np.uint8)
    print('Fingerprint input:',X.shape,'dtype:',X.dtype)
    with warnings.catch_warnings(record=True) as pred_warn:
        warnings.simplefilter('always')
        preds=np.asarray(predictor.predict(X))
        raw=predictor.predict_proba(X)
    for w in pred_warn:
        print('PREDICT WARNING:',str(w.message))
    if not isinstance(raw,np.ndarray):
        print('predict_proba returned type:',type(raw).__name__)
    proba=np.asarray(raw,dtype=float)
    classes=np.asarray(getattr(predictor,'classes_',[]))
    if proba.ndim!=2:
        raise RuntimeError(f'Expected 2-D probabilities, observed shape {proba.shape}')
    good_shape=(proba.shape==(len(smiles),len(classes)))
    rowsums=proba.sum(axis=1)
    range_invalid=(~np.isfinite(proba)).any(axis=1) | (proba < 0).any(axis=1) | (proba > 1).any(axis=1)
    rowsum_invalid=~np.isclose(rowsums,1,atol=1e-5,rtol=1e-5)
    report={
        'rf_path':str(args.rf), 'sklearn_runtime':sklearn.__version__,
        'estimator_type':type(predictor).__module__+'.'+type(predictor).__name__,
        'classes':classes.tolist(),'n_features_in':getattr(predictor,'n_features_in_',None),
        'n_test_smiles':len(smiles),'predict_counts':dict((str(k),int(v)) for k,v in Counter(preds.tolist()).items()),
        'proba_shape':list(proba.shape),'proba_valid_shape':bool(good_shape),
        'proba_min':float(np.nanmin(proba)), 'proba_max':float(np.nanmax(proba)),
        'proba_row_sum_min':float(np.nanmin(rowsums)), 'proba_row_sum_max':float(np.nanmax(rowsums)),
        'n_prob_rows_out_of_range_or_nonfinite':int(range_invalid.sum()),
        'n_prob_rows_not_normalized':int(rowsum_invalid.sum()),
        'valid_probability_matrix':bool(good_shape and not range_invalid.any() and not rowsum_invalid.any()),
        'load_warnings':[str(w.message) for w in captured],
        'predict_warnings':[str(w.message) for w in pred_warn],
    }
    print(json.dumps(report, indent=2))
    if good_shape:
        for i,(s,pred,prob) in enumerate(zip(smiles,preds,proba)):
            if i>=5: break
            print(f'example[{i}] SMILES={s[:75]} | predict={pred} | predict_proba={prob.tolist()}')
    args.output.write_text(json.dumps(report,indent=2))
    print('Saved non-destructive report:',args.output)
    if not report['valid_probability_matrix']:
        print('FAILED: Do not use the saved predicted Ames-toxic percentages as a validated result.')
    else:
        print('PASS: Probability mathematics are valid. Feature provenance, class meaning, '
              'and independent toxicity generalization still require separate verification.')

if __name__=='__main__':
    main()
