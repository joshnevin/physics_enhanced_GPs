import numpy as np
def convert_to_lin(x):
    return 10**(x/10)
def convert_to_db(x):
    return 10*np.log10(x)
def convert_to_dbm(x):
    return 10*np.log10(x/1e-3)
def convert_to_lin_dbm(x):
    return 1e-3*convert_to_lin(x)
def apply_b2b(snr, b2b):
    return convert_to_db(1/(1/convert_to_lin(snr) + 1/convert_to_lin(b2b)))
