import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import coptpy as cp
from coptpy import quicksum
from coptpy import COPT
from code.utils import dis, generW, power, draw_locations_opt, rand_create_Boxes, rand_gener_map

class AllocVeh:
    """
    AllocVeh class represents a vehicle allocation problem.

    Attributes:
        init_point (int): The initial point of the vehicle.
        locs (list): List of locations.
        boxes (list): List of boxes.
        velocity (float): The velocity of the vehicle.
        T (float): Time constraint for the problem.
        complete (bool): Indicates if the input data is complete.
        constructed (bool): Indicates if the model is constructed.
        model (cp.Model): The optimization model.
        num_locs (int): The number of locations.
        num_boxes (int): The number of boxes.
        dists (np.matrix): Matrix of distances between locations.
        box_idxes (list): List of box indices.
        loc_idxes (list): List of location indices.
        loc_div_init (list): List of location indices excluding the initial point.
        W (np.matrix): Matrix representing the relationship between boxes and locations.
        Wset (list): List of location indices for each box.

    Methods:
        __init__(self, init_point=None, locs=None, boxes=None, velocity=None, T=None):
            Initializes the AllocVeh object.
        input_data(self, init_point, locs, boxes, velocity, T=None):
            Sets the input data for the problem.
        _preprocess(self):
            Preprocesses the input data.
        construct_model(self):
            Constructs the optimization model.
        solve(self, return_opt=False):
            Solves the optimization model.
        draw_locations_opt(self):
            Draws the optimal locations.
    """
    def __init__(self, init_point=None, locs=None, boxes=None, velocity=None, T=None):
        """
        Initializes the AllocVeh object.

        Args:
            init_point (int, optional): The initial point of the vehicle. Defaults to None.
            locs (list, optional): List of locations. Defaults to None.
            boxes (list, optional): List of boxes. Defaults to None.
            velocity (float, optional): The velocity of the vehicle. Defaults to None.
            T (float, optional): Time constraint for the problem. Defaults to None.
        """
        env = cp.Envr()
        self.complete = False
        self.constructed = False
        self.model = env.createModel()
        self.locs = locs
        self.boxes = boxes
        self.num_locs = len(locs) if locs is not None else 0
        self.num_boxes = len(boxes) if boxes is not None else 0
        self.init_point = init_point
        self.velocity = velocity
        self.T = T
        if init_point is not None and locs is not None and boxes is not None and velocity is not None:
            self.complete = True
        if self.complete:
            self._preprocess()
        print(f'Initializing model, {self.num_locs} locations and {self.num_boxes} boxes')
    
    def input_data(self, init_point, locs, boxes, velocity, T=None):
        """
        Sets the input data for the problem.

        Args:
            init_point (int): The initial point of the vehicle.
            locs (list): List of locations.
            boxes (list): List of boxes.
            velocity (float): The velocity of the vehicle.
            T (float, optional): Time constraint for the problem. Defaults to None.
        """
        self.locs = locs
        self.boxes = boxes
        self.num_locs = len(locs)
        self.num_boxes = len(boxes)
        self.init_point = init_point
        self.velocity = velocity
        print(f'Input data, {self.num_locs} locations and {self.num_boxes} boxes')
        self._preprocess()
        self.complete = True
        
    def _preprocess(self):
        """
        Preprocesses the input data.
        """
        self.dists = np.matrix(np.array([[dis(self.locs[i], self.locs[j]) for j in range(self.num_locs)]\
            for i in range(self.num_locs)]))
        self.box_idxes=[i for i in range(self.num_boxes)]
        self.loc_idxes=[i for i in range(self.num_locs)]
        self.loc_div_init=self.loc_idxes.copy()
        self.loc_div_init.remove(self.init_point)
        self.W = generW(self.boxes, self.num_locs)
        for i in self.box_idxes:
            self.W[i,self.init_point]=1
        self.Wset = [[i for i in self.loc_idxes if self.W[t,i]==1] for t in self.box_idxes]
        
    def construct_model(self):
        """
        Constructs the optimization model.
        """
        assert self.complete, 'Input data is not complete!'
        x = self.model.addVars(self.num_locs, self.num_locs, vtype=COPT.BINARY, nameprefix='x')
        y = self.model.addVars(self.num_boxes, vtype=COPT.BINARY, nameprefix='y')
        
        sub = [list(power(i)) for i in self.Wset]
        
        # Constraints
        # Avoid subcycle
        self.model.addConstrs(quicksum(x[i,j] for i in S for j in S)<=len(S)-1\
            for t in self.box_idxes for S in sub[t])
        
        # We need first go to get box
        self.model.addConstrs(x[self.init_point, self.boxes[t].loc] == y[t]\
            for t in self.box_idxes)
        
        # We must pass by sites in W t when we choose box t 
        self.model.addConstrs(x.sum('*',j) == quicksum(self.W[t,j]*y[t]\
            for t in self.box_idxes) for j in self.loc_idxes)
        self.model.addConstrs(x.sum(j,'*') == quicksum(self.W[t,j]*y[t]\
            for t in self.box_idxes) for j in self.loc_idxes)
        
        # We only load one box
        self.model.addConstr(y.sum() == 1)
        
        # Battery constraint
        if self.T is not None:
            self.model.addConstrs(quicksum(x[i,j]*self.dists[i,j] for i in self.loc_idxes)\
                <=self.T for j in self.loc_idxes)
            
        self.model.setObjective(quicksum(self.dists[i,j]*x[i,j]*self.dists[i,j] for i in self.loc_div_init\
            for j in self.loc_div_init)+quicksum(self.dists[i,self.init_point]*x[i,self.init_point]\
            for i in self.loc_idxes)) 
        
        print('Model constructed successfully')
        self.constructed = True
        
    def solve(self, return_opt=False):
        """
        Solves the optimization model.

        Args:
            return_opt (bool, optional): Whether to return the optimal solution. Defaults to False.

        Returns:
            tuple: Tuple containing the optimal x and y values (if return_opt is True).
        """
        assert self.constructed, 'Model is not constructed!'
        self.model.solve()
        if self.model.status != COPT.OPTIMAL:
            print("[Error] fail to get optimal solution!")
            return
        print(f'Model solved')
        self.x_opt = np.array([[self.model.getVarByName("x({},{})".format(i,j)).x \
            for j in self.loc_idxes] for i in self.loc_idxes])
        self.y_opt = np.array([self.model.getVarByName("y({})".format(i)).x for i in self.box_idxes])
        if return_opt:
            return self.x_opt, self.y_opt
    
    def draw_locations_opt(self):
        """
        Draws the optimal locations.
        """
        assert self.x_opt is not None, 'Optimal solution is not available!'
        draw_locations_opt(self.locs, self.x_opt, self.loc_idxes, self.loc_div_init, True, 'optimal.png')
        
if __name__ == '__main__':
     # You Need Change Boxes, locs, init_point, velocity to Real Data
    boxes= rand_create_Boxes([13,19],20,5)
    locs = rand_gener_map(100, 20)
    init_point = 8
    velocity = velocity = [[1 for i in range(20)] for i in range(20)]
    alloc_class = AllocVeh(init_point, locs, boxes, velocity)
    alloc_class.construct_model()
    alloc_class.solve(False)
    alloc_class.draw_locations_opt()