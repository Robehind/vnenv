import random
def get_scene_names(train_scenes):
    tmp = {}
    #mapping = {"kitchen":0, "living_room":1, "bedroom":2, "bathroom":3}
    for k in train_scenes.keys():
        ss = [int(x) for x in train_scenes[k].split('-')]
        tmp[k] = [make_scene_name(k, i) for i in range(ss[0], ss[-1]+1)]
    return tmp

def make_scene_name(scene_type, num):
    mapping = {"kitchen":'', "living_room":'2', "bedroom":'3', "bathroom":'4'}
    front = mapping[scene_type]
    endd = '_physics' if (front == '' or front == '2') else ''
    if num >= 10 or front == '':
        return "FloorPlan" + front + str(num) + endd
    return "FloorPlan" + front + "0" + str(num) + endd

def random_divide(total_epi, chosen_scenes, n):
    """输入的chosen scenes是从get_scene_names得到的还打包过一次的场景名"""
    scenes = [x for i in chosen_scenes.values() for x in i]
    out = []
    random.shuffle(scenes)
    if n > len(scenes):
        epi_nums = [total_epi//n for _ in range(n)]
        for i in range(0, total_epi%n):
            epi_nums[i%n]+=1
        out = [scenes for _ in range(n)]
        return out, epi_nums
    step = len(scenes)//n
    mod = len(scenes)%n
    
    for i in range(0, n*step, step):
        out.append(scenes[i:i + step])
    
    for i in range(0, mod):
        out[i].append(scenes[-(i+1)])

    num_per_epi = total_epi/len(scenes)
    epi_nums = [round(len(x)*num_per_epi) for x in out]
    epi_nums[0] += total_epi-sum(epi_nums)
    return out, epi_nums

if __name__ == "__main__":
    train_scenes = {
        'kitchen':'1-20',
        'living_room':'5',
    }
    cc = get_scene_names(train_scenes)
    #cc = [[1],[2,3,4,5],[6,7,8,9,10,11,12]]
    print(random_divide(100,cc, 7))
