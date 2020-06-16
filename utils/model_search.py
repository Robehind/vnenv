import os

def search_newest_model(exps_dir, exp_name):
    _, dirs, _ = next(os.walk(exps_dir))
    for d in dirs[::-1]:
        if d.split('-')[0] == exp_name:
            _, _, files = next(os.walk(os.path.join(exps_dir, d)))
            for f in files[::-1]:
                if f.split('.')[-1] == 'dat':
                    return os.path.join(exps_dir, d, f)
    return None
