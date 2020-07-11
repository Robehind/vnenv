import os

def search_newest_model(exps_dir, exp_name):
    _, dirs, _ = next(os.walk(exps_dir))
    tmp = -1
    dd = []
    for d in dirs:
        sp = d.split('_')
        if sp[0] == exp_name:
            dd.append(d)
    if dd == []:
        return None

    ff = None
    fd = None
    tmp = -1
    dd.sort()
    print(dd)
    #stop = False
    for d in dd[::-1]:
        _, _, files = next(os.walk(os.path.join(exps_dir, d)))
        for f in files:
            if f.split('.')[-1] == 'dat':
                frame = int(f.split('_')[1])

                if frame > tmp:
                    tmp = frame
                    ff = f
                    fd = d
        if ff != None and fd != None:
            return os.path.join(exps_dir, fd, ff)
    return None
    
