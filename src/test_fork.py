import os

def test_func(x):
    print(x)


"""
pid = os.fork()
for i in range(3):
    if pid>1:
        pid = os.fork()
    elif pid == 0:
        test_func(i)
        print('Stopp')
        exit()

"""

for i in range(3):
    pid = os.fork()
    if pid == 0:
        test_func(i)
        sys.exit()


"""
pid=os.fork()
if pid>0:
    pid=os.fork()
    if pid>0:
        test_func(0)
    else:
        test_func(1)
else:
    test_func(2)
"""