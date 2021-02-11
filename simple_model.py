import numpy as np
from scipy.optimize import curve_fit

class simple_model:
    
    def _convert_to_lin(self, x):
        return 10**(x/10)
    
    def _convert_to_db(self, x):
        return 10*np.log10(x)
    
    def _obj_func(self, p_in, a, b, c):
        return 1/(a*p_in**(-1) + b*p_in**2) + c 
    
    def fit(self, p_in, y, init_guess):  
        p_in = 1e-3*self._convert_to_lin(p_in) # W
        y = self._convert_to_lin(y)
        opt_res = curve_fit(self._obj_func, p_in, y, init_guess)
        return opt_res
    
    def predict(self, p_in, a, b, c):
        p_in = 1e-3*self._convert_to_lin(p_in) # W
        return self._convert_to_db(1/(a*p_in**(-1) + b*p_in**2) + c )
