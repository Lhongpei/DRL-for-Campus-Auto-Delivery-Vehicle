from itertools import chain, combinations
import numpy as np
import matplotlib.pyplot as plt
from code.data import Location, Box
def power(lis):
    s = list(lis)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)))

def dis(loc1,loc2):
    return (loc1.x-loc2.x)**2+(loc1.y-loc2.y)**2

def generW(setOfBox, num_loc):
    W=[]
    for i in setOfBox:
        target=i.target
        loc=i.loc
        target.append(loc)
        W.append([i in target for i in range(num_loc)])
    W=np.matrix(W)
    return W

def draw_locations_opt(locations, x_opt, V, V_div_init, save=False, name='optimal.png'):
    plt.figure(figsize=(16, 9))
    for loc in locations:
        plt.scatter(loc.x, loc.y, marker='o')
        plt.text(loc.x - 3, loc.y + 1, loc.id)
    for i in V:
        for j in V_div_init:
            if x_opt[i,j]>=0.6:
                x_line=(locations[i].x,locations[j].x)
                y_line=(locations[i].y,locations[j].y)
                plt.plot(x_line,y_line)
    plt.show()
    if save:
        plt.savefig(name)
        
def rand_create_Boxes(ids,loc_num,numOfBox):#prefer is 1
    set_box=[]
    np.random.seed(seed=210)
    for j,i in enumerate(ids):
        setofloc=[i for i in range(loc_num)]
        setofloc.remove(i)
        lis=np.random.choice([j for j in setofloc],numOfBox).tolist()
        set_box.append(Box(j,i,lis,1))
    return set_box

def rand_gener_map(scale, num_loc):
    location=[]
    np.random.seed(seed=200)
    for i in range(num_loc):
        location.append(Location(i,np.random.rand(1)*scale,np.random.rand(1)*scale))
    return location