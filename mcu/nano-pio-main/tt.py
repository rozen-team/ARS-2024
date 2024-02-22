from time import time 

stim = time()
while True:
    tim = time() - stim

    print((time() // 0.2) % 2)

