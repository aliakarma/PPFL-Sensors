import json, os, numpy as np
results_dir = 'results/logs/'
metrics = []
for root, dirs, files in os.walk(results_dir):
    if 'summary.json' in files:
        with open(os.path.join(root, 'summary.json')) as f:
            d = json.load(f)
            t = d.get('config', {}).get('training', {})
            dt = d.get('config', {}).get('dataset', {})
            # Get actual values 
            ens_grp = t.get('n_ensemble_groups') if t.get('aggregation')=='ensemble' else 'None'
            metrics.append({
                'agg': t.get('aggregation'),
                'groups': ens_grp,
                'data': dt.get('name'),
                'fl_acc': d.get('final_fl_accuracy',0),
                'att_acc': d.get('mean_best_attack_accuracy',0),
                'priv': d.get('final_privacy_score',0)
            })

for g in ['fedavg']:
    for d_name in ['har', 'synthetic']:
        filt = [m for m in metrics if m['agg']==g and m['data']==d_name]
        if filt:
            print(f"{g} {d_name} FL={np.mean([m['fl_acc'] for m in filt]):.4f} +/- {np.std([m['fl_acc'] for m in filt]):.4f} ATT={np.mean([m['att_acc'] for m in filt]):.4f}  PRIV={np.mean([m['priv'] for m in filt]):.4f}")

for g in [2, 3, 5]:
    filt = [m for m in metrics if m['agg']=='ensemble' and m['groups']==g and m['data']=='har']
    if filt:
        print(f"ensemble {g} groups HAR FL={np.mean([m['fl_acc'] for m in filt]):.4f} ATT={np.mean([m['att_acc'] for m in filt]):.4f}  PRIV={np.mean([m['priv'] for m in filt]):.4f}")

