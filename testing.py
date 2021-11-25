from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support

def func(a, b):
    return a + b

def main():
    a_args = [1,2,3]
    second_arg = 1
    with Pool() as pool:
        print("test")
        L = pool.starmap(func, [(1, 1), (2, 1), (3, 1)])
        M = pool.starmap(func, zip(a_args, repeat(second_arg)))
        N = pool.map(partial(func, b=second_arg), a_args)
        assert L == M == N

#if __name__=="__main__":
    #freeze_support()
    #main()


class test():
    def __init__(self, x):
        self.x=x
    
    def func2(self):
        a_args = [1,2,3]
        second_arg = 1
        with Pool() as pool:
            print("test")
            L = pool.starmap(func, [(1, 1), (2, 1), (3, 1)])
            M = pool.starmap(func, zip(a_args, repeat(second_arg)))
            N = pool.map(partial(func, b=second_arg), a_args)
            print(L)
            assert L == M == N
            
        
        return


xx=test(3)
xx.func2()