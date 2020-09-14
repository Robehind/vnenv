class ScalarMeanTracker(object):
    """用来算平均数的一个类，记录数据用"""
    def __init__(self) -> None:
        self._sums = {}
        self._counts = {}

    def add_scalars(self, scalars, count = True):
        #输入的是dict，其中值可以是一个list，list长度可以不一样
        for k in scalars:
            if k not in self._sums:
                self._sums[k] = scalars[k]
                if isinstance(scalars[k], list):
                    self._counts[k] = [1 for _ in range(len(scalars[k]))]
                else:
                    self._counts[k] = 1
            else:
                if isinstance(scalars[k], list):
                    self._list_add(k, scalars[k])
                else:
                    self._sums[k] += scalars[k]
                    self._counts[k] += count
    
    def _list_add(self, label, in_list):
        a = self._sums[label]
        b = in_list
        c = self._counts[label]
        
        if len(a) < len(b): 
            c += [1 for _ in range(len(a),len(b))]
            a,b = in_list,self._sums[label]
        for i in range(len(b)):
            c[i] += 1
        self._counts[label] = c
        tmp = [a[i]+b[i] for i in range(len(b))] + a[len(b):len(a)]
        self._sums[label] = tmp
        
    def pop_and_reset(self, no_div_list = []):
        for k in no_div_list:
            self._counts[k] = 1 if not isinstance(self._counts[k], list) else [1 for _ in range(len(self._counts[k]))]
        means = {}
        for k in self._sums:
            if isinstance(self._sums[k], list):
                means[k] = [a/b for a,b in zip(self._sums[k],self._counts[k])]
            else:
                means[k] = self._sums[k] / self._counts[k]
        self._sums = {}
        self._counts = {}
        return means

class LabelScalarTracker(object):
    """带标签的用来算平均数的一个类，记录数据用"""
    def __init__(self):
        self.trackers = {}

    def __getitem__(self, key):
        if key in self.trackers:
            return self.trackers[key]
        else:
            self.trackers[key] = ScalarMeanTracker()
            return self.trackers[key]

    def items(self):
        return self.trackers.items()

    def pop_and_reset(self, no_div_list = []):
        out = {}
        for k in self.trackers:
            out[k] = self.trackers[k].pop_and_reset(no_div_list)
        self.trackers = {}
        return out
