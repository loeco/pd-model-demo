import os, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

def load_data(path='data/credit_synth.csv'):
    return pd.read_csv(path)

def build_model(v='0.1'):
    if v=='0.1':
        use = ['age','income','utilization']; clf = LogisticRegression(max_iter=200); calibrate=False
    elif v=='0.2':
        use = ['age','income','utilization','delinquency_count']; clf = LogisticRegression(C=0.5,max_iter=300); calibrate=False
    else: # v0.3
        use = ['age','income','utilization','delinquency_count']; clf = LogisticRegression(C=0.5,max_iter=300); calibrate=True
    return use, clf, calibrate

def train(version='0.2'):
    df = load_data()
    use, clf, calibrate = build_model(version)
    X, y = df[use], df['default']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    if calibrate:
        base = clf.fit(Xtr, ytr)
        clf = CalibratedClassifierCV(base, method='isotonic').fit(Xtr, ytr)
    else:
        clf = clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte, proba)
    brier = brier_score_loss(yte, proba)
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(clf, f'artifacts/pd_model_v{version}.joblib')
    print({'version':version,'auc':round(auc,3),'brier':round(brier,3)})

if __name__ == '__main__':
    import sys
    v = sys.argv[1] if len(sys.argv)>1 else '0.1'
    train(v)
