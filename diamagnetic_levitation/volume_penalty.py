import numpy as np
from scipy.special import erf

class Ellipse:
    
    def __init__(self,a,b,h,fun='erf'):
        
        self.a = a
        self.b = b
        self.h = h
        
        self.__name__ = 'Ellipse'
        
        if fun == 'erf':
            self.f  = lambda x: erf(x*np.sqrt(np.pi)/2)
            self.df = lambda x: np.exp(-(x*np.sqrt(np.pi)/2)**2)
        
        if fun == 'tanh':
            self.f  = np.tanh
            self.df = lambda x: 1 - np.tanh(x)**2


    def __call__(self,x,y,s):
        x, y = self.rotate(x,y,s)
        j = 0.5*(1 - self.f(self.E(x,y)/self.h))
        return j

    def grad(self,x,y,s):
        x, y = self.rotate(x,y,s)
        M, (Mx, My) = self.E(x,y), self.dE(x,y)
        M = -0.5*self.df(M/self.h)/self.h
        Mx, My = self.rotate(Mx,My,-s)
        return M*Mx, M*My
    
    def rotate(self,x,y,s):
        c, s = np.cos(s), np.sin(s)
        return x*c + y*s, y*c - x*s

    def E(self,x,y):
        return (x/self.a)**2 + (y/self.b)**2 - 1

    def dE(self,x,y):
        return (2/self.a**2)*x, (2/self.b**2)*y

    def Area(self):
        return np.pi*self.a*self.b
    
    def Inertia(self):
        return (self.a**2 + self.b**2)*self.Area()/4


class Body:
    
    def __init__(self, mask, domain, position  = (None,None),
                                     velocity  = (None,[0,0]),
                                     angle     = ('theta',0),
                                     frequency = ('omega',0)):
        
        self.mask    = mask
        self.inertia = mask.Inertia()
        
        self.dimension  = 2
        self.cross_prod = [-1,1]
        
        if position[0] == None:
            position = (tuple(map(str,domain.bases)),position[1])
        
        if position[1] == None:
            position = (position[0],[sum(b.interval)/2 for b in domain.bases])
        
        if velocity[0] == None:
            velocity = (tuple('v'+x for x in position[0]),velocity[1])
        
        self.position  = position[0]
        self.velocity  = velocity[0]
        self.angle     = angle[0]
        self.frequency = frequency[0]
        
        self.names = position[0] + velocity[0] + (angle[0] , frequency[0])
        self._state = np.array( position[1] + velocity[1] + [angle[1], frequency[1]] )
        
        self._grids    = domain.grids
        
        self._maps = [lambda r: r for i in range(self.dimension)]
        for i, basis in enumerate(domain.bases):
            if basis.__class__.__name__ == 'Fourier':
                a,b = basis.interval
                self._maps[i] = lambda r: ( (r - a) % (b-a) ) + a


    def step(self,dt,force,torque):
    
        self._state[0]  += dt*self._state[2]
        self._state[1]  += dt*self._state[3]
        
        self._state[2]  += dt*force[0]
        self._state[3]  += dt*force[1]
        
        self._state[4] += dt*self._state[5]
        self._state[5] += dt*torque
    

    def __getitem__(self,item):
        
        if item == self.position:
            return tuple(self[x] for x in self.position)
    
        if item == self.velocity:
            return tuple(self[v] for v in self.velocity)

        if item in self.names: item = self.names.index(item)
        return self._state[item]

    def __setitem__(self,item,value):
        
        if item in self.names: item = self.names.index(item)
        self._state[item] = value


    def field(self,item,*argv,**kwargs):
    
        if item in self.position:
            i = self.position.index(item)
            return self._maps[i](self._grids(*argv,**kwargs)[i] - self._state[i])
        
        if item in self.velocity:
            V, i = self[item], self.velocity.index(item)
            X = self.field(self.position[1-i],*argv,**kwargs)
            return V + X*self.cross_prod[i]*self._state[-1]

        if tuple(item) in (self.position,self.velocity):
            return tuple(self.field(i,*argv,**kwargs) for i in item)

        if item == 'interior':
            return self.mask(*self.field(self.position,*argv,**kwargs),self._state[-2])

        if item == 'boundary':
            return self.mask.grad(*self.field(self.position,*argv,**kwargs),self._state[-2])
