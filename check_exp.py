import os, json
import pandas as pd

lines = []
for d in os.listdir('results/logs'):
    p = os.path.join('results/logs', d)
    if os.path.isdir(p) and os.path.exists(os.path.join(p, 'summary.json')):
        with open(os.path.join(p, 'summary.json')) as f: s = json.load(f)
        with open(os.path.join(p, 'config.json')) as f: c = json.load(f)
        
        lines.append({
            'run_id': d,
            'dataset': c.get('dataset', {}).get('name', 'unk'),
            'agg': c.get('training', {}).get('aggregation', 'unk'),
            'grp': c.get('training', {}).get('n_ensemble_groups', 'unk'),
            'fl_acc': s['final_fl_accuracy'],
            'att': s['mean_best_attack_accuracy'],
            'priv': s['final_privacy_score']
        })

df = pd.DataFrame(lines)
df = df.groupby(['dataset', 'agg', 'grp']).agg({'fl_acc':'mean', 'att':'mean', 'priv':'mean', 'run_id':'count'})
print(df)
