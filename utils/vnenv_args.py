import json

class VNENVargs:
    def __init__(self, args_dict = None, **kwargs):
        '''注意kwargs中的参数会覆盖args_dict中的'''
        self.update(args_dict, **kwargs)

    def update(self, args_dict = None, **kwargs):
        '''注意kwargs中的参数会覆盖args_dict中的'''
        if args_dict != None:
            for k in args_dict:
                setattr(self, k, args_dict[k])
        if kwargs !=None:
            for k in kwargs:
                setattr(self, k, kwargs[k])

    def save_args(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, indent=4)

if __name__ == "__main__":
    args = VNENVargs(a = 1, s= 2, vsvv = 3)
    args.save_args('./test.json')



        

