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

if __name__ == "__main__":
    args = VNENVargs(dict(vvv=1), a = 1, s= 2, vvv = 3)
    print(args.__dict__)



        

