import numpy as np
landscape = np.random.randint(1,high=100,size=(5,5))
print(landscape)
pos = [2,2]
k = landscape[pos[0]][pos[1]]
def top():
    a1 = pos[0]
    a2 = pos[1]
    if a1-1<0:
        return 0
    return landscape[a1-1][a2]
def bot():
    a1 = pos[0]
    a2 = pos[1]
    if a1+1>2:
        return 0
    return landscape[a1+1][a2]
def left():
    a1 = pos[0]
    a2 = pos[1]
    if a2-1<0:
        return 0
    return landscape[a1][a2-1]
def right():
    a1 = pos[0]
    a2 = pos[1]
    if a2+1>2:
        return 0
    return landscape[a1][a2+1]

def st():
    k = landscape[pos[0]][pos[1]]
    k1 = top()
    k2 = bot()
    k3 = left()
    k4 = right()
    k5 = max(k1,k2,k3,k4)
    if(k5==k or k5<k):
        return 1
    if k5==k1:
        pos[0] = pos[0]-1
        return 0
    if k5==k2:
        pos[0] = pos[0]+1
        return 0
    if k5==k3:
        pos[1] = pos[1]-1
        return 0
    if k5==k4:
        pos[1] = pos[1]+1
        return 0
u = 0
print("current best pos",landscape[pos[0]][pos[1]])
while u==0:
    u  = st()
    print("current best pos",landscape[pos[0]][pos[1]])
print("final best pos",landscape[pos[0]][pos[1]])
