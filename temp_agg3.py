import os, json
import pandas as pd

rows = []
for d in os.listdir('results/logs'):
    p = os.path.join('results/logs', d)
    if os.path.isdir(p) and os.path.exists(os.path.join(p, 'summary.json')):
        with open(os.path.join(p, 'summary.json')) as f: s = json.load(f)
        with open(os.path.join(p, 'config.json')) as f: c = json.load(f)
        
        name = c.get('experiment', 'HAR baseline')
        if name == 'default':
            d_name = c.get('dataset', {}).get('name', 'har')
            if d_name == 'synthetic':
                name = 'Synthetic'
        
        rows.append({
            'name': name,
            'dataset_name': c.get('dataset', {}).get('name', 'none'),
            'ens': c.get('training', {}).get('aggregation', 'none'),
            'grp': c.get('training', {}).get('n_ensemble_groups', 3),
            'sigma': c.get('defense', {}).get('noise', {}).get('sigma', 0.0),
            'fl_acc': s['final_fl_accuracy'],
            'attack_acc': s['mean_best_attack_accuracy'],
            'privacy': s['final_privacy_score']
        })

df = pd.DataFrame(rows)
if len(df) > 0:
    res = df.groupby(['name', 'dataset_name', 'ens', 'grp', 'sigma']).agg({'fl_acc': ['mean', 'std', 'count'], 'attack_acc': 'mean', 'privacy': 'mean'})
    print(res)
