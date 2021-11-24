import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
from helper_functions import convert_to_lin, convert_to_db, convert_to_dbm, convert_to_lin_dbm, apply_b2b
def train_phys_gp(pch, pch_phys, snr, snr_phys, pch_pred):
    pch = pch.reshape(-1, 1)  # needed for SK learn input
    pch_phys = pch_phys.reshape(-1,1)
    pch_pred = pch_pred.reshape(-1,1)
    snr = snr.reshape(-1,1)
    snr_phys = snr_phys.reshape(-1,1)
    kernel_sk = C(1, (1e-5, 1e5)) * RBF(1, (1e-8, 1e5)) +  W(1, (1e-5,1e5))
    gpr_phys = GaussianProcessRegressor(kernel=kernel_sk, n_restarts_optimizer = 20, normalize_y=True)
    gpr_phys.fit_phys(pch, pch_phys, snr, snr_phys)
    mu_sk_phys, std_sk_phys = gpr_phys.predict(pch_pred, return_std=True)
    std_sk_phys = np.reshape(std_sk_phys,(np.size(std_sk_phys), 1))
    theta_phys = gpr_phys.kernel_.theta
    lml_phys = gpr_phys.log_marginal_likelihood()
    return mu_sk_phys, std_sk_phys, theta_phys, lml_phys
def train_standard_gp(pch, snr, pch_pred):
    pch = pch.reshape(-1, 1)  # needed for SK learn input
    snr = snr.reshape(-1,1)
    pch_pred = pch_pred.reshape(-1,1)
    kernel_sk = C(1, (1e-5, 1e5)) * RBF(1, (1e-5, 1e5)) +  W(1, (1e-8,1e5))
    gpr = GaussianProcessRegressor(kernel=kernel_sk, n_restarts_optimizer = 20, normalize_y=True)
    gpr.fit(pch, snr)
    mu_sk, std_sk = gpr.predict(pch_pred, return_std=True)
    std_sk = np.reshape(std_sk,(np.size(std_sk), 1))
    theta = gpr.kernel_.theta
    lml = gpr.log_marginal_likelihood()
    return mu_sk, std_sk, theta, lml
def snr_simple_gen(p_in, a, b, rseed, sig):
        np.random.seed(rseed)
        p_in = p_in + np.random.normal(0, sig, len(p_in))
        p_in = convert_to_lin_dbm(p_in) # W
        return convert_to_db(1/( (a*p_in**(-1) + b*p_in**2) + 1/convert_to_lin(14.8)  ))
def gen_phys_targets(pch, num_pts_phys, rseed, sig, a_opt, b_opt):
    pch_gn = np.linspace(pch[0], pch[-1], num_pts_phys)
    #snr_phys = (model.predict_snr(pch_gn)).reshape(-1,1)
    snr_phys = snr_simple_gen(pch_gn, a_opt, b_opt, rseed, sig)
    return pch_gn, snr_phys
def hyp_var_sig(sig_range, pch, num_pts_phys, snr, pch_pred, a_opt, b_opt):
    mus = []
    ls = []
    sigs = []
    lmls = []
    pred_stds = []
    mses = []
    for sig in sig_range:
        #nsr_s = convert_to_db(sig*convert_to_lin(-16))
        pch_gn, snr_phys = gen_phys_targets(pch, num_pts_phys, 1, sig, a_opt, b_opt)
        pred_mean, pred_std, theta_phys, lml_phys = train_phys_gp(pch, pch_gn, snr, snr_phys, pch_pred)
        mus.append(theta_phys[0])
        ls.append(theta_phys[1])
        sigs.append(theta_phys[2])
        lmls.append(lml_phys)
        pred_stds.append(np.mean(pred_std))
        mses.append(calc_mae(snr, pred_mean))
    return mus, ls, sigs, lmls, pred_stds, mses
def hyp_var_num_pts(sigma, pch, num_pts_range, snr, pch_pred, a_opt, b_opt):
    mus = []
    ls = []
    sigs = []
    lmls = []
    pred_stds = []
    mses = []
    for num_pts in num_pts_range:
        pch_gn, snr_phys = gen_phys_targets(pch, num_pts, 2, sigma, a_opt, b_opt)
        pred_mean, pred_std, theta_phys, lml_phys = train_phys_gp(pch, pch_gn, snr, snr_phys, pch_pred)
        mus.append(theta_phys[0])
        ls.append(theta_phys[1])
        sigs.append(theta_phys[2])
        lmls.append(lml_phys)
        pred_stds.append(np.mean(pred_std))
        mses.append(calc_mae(snr, pred_mean))
    return mus, ls, sigs, lmls, pred_stds, mses
def calc_mae(data, y):
    return np.mean(((data - y)**2)**0.5)
