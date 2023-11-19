import numpy as np
import torch
        

class Flipper(object):
    '''
        Flipping data via three axis
        data is of dim: channel x width x height x depth
        axis: set to 1, 2, 3 for flipping on width, height, or depth, respectively
    '''
    def __call__(self, data):
        
        num_ops = 6
        odds = np.ones(num_ops) / num_ops
        op = np.random.choice(np.arange(num_ops), p=odds)
        if op == 0:
            data = self.flip(data, 1)
            data = np.ascontiguousarray(data)   # copy to avoid negative strides of numpy arrays
        elif op == 1:
            data = self.flip(data, 2)
            data = np.ascontiguousarray(data) 
        elif op == 2:
            data = self.flip(data, 3)
            data = np.ascontiguousarray(data) 
        elif op == 3:
            data = self.flip(data, 1)
            data = self.flip(data, 2)
            data = np.ascontiguousarray(data) 
        elif op == 4:
            data = self.flip(data, 1)
            data = self.flip(data, 3)
            data = np.ascontiguousarray(data) 
        elif op == 5:
            data = self.flip(data, 2)
            data = self.flip(data, 3)
            data = np.ascontiguousarray(data) 

        return  data

    
    def flip(self, data, axis):
        output = np.asarray(data).swapaxes(axis, 0)
        output = output[::-1, ...]
        output = output.swapaxes(0, axis)
        return output
    
class Inserter(object):
    '''
            Insert the data into a fixed size
            data is of dim: channel x width x height x depth
            return torch tensor
            size: set to be the largest of all data
    '''
    def __init__(self, size):
        super(Inserter, self).__init__()
        self.size = size
    def __call__(self, data):
        
        c, w, h, d = data.shape
        ww, hh, dd = self.size
        output = np.zeros((c, ww, hh, dd)).astype(np.float32)
        x = np.random.randint(0, ww - w + 1)
        y = np.random.randint(0, hh - h + 1)
        z = np.random.randint(0, dd - d + 1)
        output[:, x:x+w, y:y+h, z:z+d] = data
        return torch.from_numpy(output)


