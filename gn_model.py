import numpy as np
class GN_model:
    def __init__(self, span_len, lam_op, num_lam, nl_cof, disp_cof,  alpha, r_sym, bchrs, grid_sp, n_fig, num_spans):
        self.span_len = span_len
        self.lam_op = lam_op
        self.num_lam = num_lam
        self.nl_cof = nl_cof
        self.disp_cof = disp_cof
        self.alpha = alpha
        self.r_sym = r_sym
        self.bchrs = bchrs
        self.grid_sp = grid_sp
        self.n_fig = n_fig
        self.num_spans = num_spans
        self.freq = 299792458/(self.lam_op*1e-9)
        self.al_lin = np.log((10**(self.alpha/10)))/2 # fibre loss [1/km] 
        self.beta2 = (self.disp_cof*(self.lam_op**2))/(2*np.pi*299792.458) # dispersion coefficient at given wavelength [ps^2/km]
        self.l_eff = (1 - np.exp(-2*self.al_lin*self.span_len ))/(2*self.al_lin) # effective length [km]
        self.l_eff_as = 1/(2*self.al_lin) # the asymptotic effective length [km]  
        self.h = 6.63*1e-34 # Planck's constant [Js]
        self.Bwdm = self.bchrs * self.num_lam ** ( self.bchrs / self.grid_sp )  # channel BW [GHz]
        self.epsilon = 0.3*np.log(1 + ( 6 * self.l_eff_as ) / ( self.span_len * np.arcsinh( 0.5*np.pi**2 * self.beta2 *self.l_eff_as * self.Bwdm**2 )  )  )
    def predict_snr(self, p_ch):
        Gwdm = (1e-3*self.convert_to_lin(p_ch))/(self.bchrs*1e9)  # [W]
        Gnli = (1e24*(8/27)*(self.nl_cof**2)*(Gwdm**3)*(self.l_eff**2) ) /(np.pi*self.beta2*self.l_eff_as)  *  (np.arcsinh((np.pi**2)*0.5*self.beta2*self.l_eff_as*(self.bchrs**2)*(self.num_lam**((2*self.bchrs)/self.grid_sp))  ) )*self.num_spans**(1 + self.epsilon)                                                                             
        Pase = self.n_fig*self.h*self.freq*(self.convert_to_lin(self.alpha*self.span_len) - 1)*self.bchrs*1e9*self.num_spans       
        Pch = 1e-3*10**(p_ch/10) 
        snr = (Pch/(Pase + Gnli*self.bchrs*1e9))
        snr = self.apply_trx_b2b(snr, 16.5)
        #snr = ( snr**(-1) + (self.convert_to_lin(16.5))**(-1) )**(-1)
        return self.convert_to_db(snr)
    def find_pch_opt(self):  # return optimal Pch in dBm
        PchdBm = np.linspace(-6,6,500)  # 500 datapoints for higher resolution of Pch
        numpch = len(PchdBm)
        Pchsw = 1e-3*10**(PchdBm/10)  # convert from dBm to linear units [W]
        Gwdmsw = Pchsw/(self.bchrs*1e9)
        Gnlisw = (1e24*(8/27)*(self.nl_cof**2)*(Gwdmsw**3)*(self.l_eff**2) ) /(np.pi*self.beta2*self.l_eff_as)  *  (np.arcsinh((np.pi**2)*0.5*self.beta2*self.l_eff_as*(self.bchrs**2)*(self.num_lam**((2*self.bchrs)/self.grid_sp))  ) )                                                                            
        G = self.alpha*self.span_len
        NFl = 10**(self.n_fig/10) # convert to linear noise figure
        Gl = 10**(G/10) # convert to linear gain
        Pasesw = NFl*self.h*self.freq*(Gl - 1)*self.bchrs*1e9 # [W] the ASE noise power in one Nyquist channel across all spans
        snrsw = (Pchsw)/(Pasesw*np.ones(numpch) + Gnlisw*self.bchrs*1e9)
        return PchdBm[np.argmax(snrsw)]
    def apply_trx_b2b(self, snr, snr_pen):
        return ( snr**(-1) + (self.convert_to_lin(16.5))**(-1) )**(-1)
    def convert_to_lin(self, x):
        return 10**(x/10)
    def convert_to_db(self, x):
        return 10*np.log10(x)
