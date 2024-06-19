from numpy.linalg import inv
from numpy.linalg import inv
from numpy.linalg import multi_dot
import numpy as np

class COV:
    def __init__(self, x:list[float], sigma:float):
        self.x = x 
        self.sigma = sigma

    @staticmethod
    def __get_xtx_gl(x):
        x_m = np.array([np.ones(len(x)), x]).transpose()
        x_m_t = x_m.transpose()
        x_c = np.matmul(x_m_t, x_m)
        return x_c
            
    @staticmethod
    def __get_xtx_global_cubic(x):
        x_m = np.array([np.ones(len(x)), x, x * x, x * x * x]).transpose()
        x_m_t = x_m.transpose()
        x_c = np.matmul(x_m_t, x_m)
        return x_c
    
    # linear regression
    @staticmethod
    def GLCov(x, sigma):
        ### Getting transpose of x and multiplying with x 
        x_c = COV.__get_xtx_gl(x)
        x_c_inv = inv(x_c)

        ### Multiplying with sigma square         
        x_c_inv = x_c_inv * (sigma * sigma)
        return x_c_inv
    
    # global cubic spline
    @staticmethod
    def GlobalCubicCov(x, sigma):
        ### Getting transpose of x and multiplying with x 
        x_c = COV.__get_xtx_global_cubic(x)
        x_c_inv = inv(x_c)
        
        ### Multiplying with sigma square
        x_c_inv = x_c_inv * (sigma * sigma)
        return x_c_inv
