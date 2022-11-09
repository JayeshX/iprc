graph = [
    [99,4,1,3],
    [4,99,2,3],
    [1,2,99,5],
    [3,3,5,99]
]
k = min(graph[0][:])
def mt(i):
    global k
    k = min(graph[i][:])
def lop(j):
    for i in range(0,4):
        if k == graph[i][j]:
            print(k)
            graph[i][j]=99
            graph[i][j]=99
            mt(i)
            return i
u=0
for i in range(0,4):
    u= lop(u)
