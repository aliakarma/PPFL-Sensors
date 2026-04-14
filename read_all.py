import os, json
import pandas as pd

lines = []
for d in os.listdir('results/logs'):
    p = os.path.join('results/logs', d)
    if os.path.isdir(p) and os.path.exists(os.path.join(p, 'summary.json')):
        with open(os.path.join(p, 'summary.json')) as f: s = json.load(f)
        with open(os.path.join(p, 'config.json')) as f: c = json.load(f)
        
        d_name = c.get('dataset', {}).get('name', 'har')
        ens = c.get('training', {}).get('aggregation', 'fedavg')
        grp = c.get('training', {}).get('n_ensemble_groups', 3)
        exp = c.get('experiment_name', 'unk')
        
        name = f"{d_name}_{ens}_{grp}_{exp}_{os.path.getmtime(os.path.join(p, 'summary.json'))}"
        
        lines.append({
            'd_name': d_name, 'ens': ens, 'grp': grp, 'fl_acc': s['final_fl_accuracy'], 'att': s['mean_best_attack_accuracy']
        })

df = pd.DataFrame(lines)
print("\n--- SYNTHETIC ---")
print(df[df['d_name'] == 'synthetic'].agg('mean'))
print("\n--- ENSEMBLE ---")
print(df[df['ens'] == 'ensemble'].groupby('grp').agg('mean'))
print("\n--- BASELINE HAR ---")
print(df[(df['d_name'] == 'har') & (df['ens'] == 'fedavg') & (df['grp'] == 3)].agg('mean'))
