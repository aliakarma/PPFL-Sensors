import os, json

print("Extracting summary metrics...")

for d in os.listdir('results/logs'):
    p = os.path.join('results/logs', d)
    if os.path.isdir(p) and os.path.exists(os.path.join(p, 'summary.json')):
        with open(os.path.join(p, 'summary.json')) as f: s = json.load(f)
        with open(os.path.join(p, 'config.json')) as f: c = json.load(f)
        
        d_name = c.get('dataset', {}).get('name', 'har')
        train = c.get('training', {})
        ens = train.get('aggregation', 'fedavg')
        grp = train.get('n_ensemble_groups', 3)
        defense = c.get('defense', {}).get('noise', {})
        noise_enabled = defense.get('enabled', False)
        sigma = defense.get('sigma', 0.0)
        
        cond = "BASE_HAR"
        if d_name == 'synthetic': cond = "SYNTHETIC"
        elif ens == 'ensemble': cond = f"ENSEMBLE_{grp}"
        elif noise_enabled: cond = f"NOISE_{sigma}"
        
        print(f"[{cond}] fl_acc: {s.get('final_fl_accuracy'):.4f}, attack: {s.get('mean_best_attack_accuracy'):.4f}, priv: {s.get('final_privacy_score'):.4f}")
