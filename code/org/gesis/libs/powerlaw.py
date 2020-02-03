# https://www.johndcook.com/blog/2015/11/24/estimating-the-exponent-of-discrete-power-law-data/

from scipy import log
from scipy.optimize import bisect
from scipy.special import zeta
from scipy import sqrt

T = None

def log_zeta(x):
    return log(zeta(x, 1))

def log_deriv_zeta(x):
    h = 1e-5
    return (log_zeta(x+h) - log_zeta(x-h))/(2*h)

def objective(x):
    global T

    return log_deriv_zeta(x) - T

def zeta_prime(x, xmin=1):
    h = 1e-5
    return (zeta(x+h, xmin) - zeta(x-h, xmin))/(2*h)

def zeta_double_prime(x, xmin=1):
    h = 1e-5
    return (zeta(x+h, xmin) -2*zeta(x,xmin) + zeta(x-h, xmin))/h**2

def sigma(n, alpha_hat, xmin=1):
    z = zeta(alpha_hat, xmin)
    temp = zeta_double_prime(alpha_hat, xmin)/z
    temp -= (zeta_prime(alpha_hat, xmin)/z)**2
    return 1/sqrt(n*temp)

def get_exponent(x):
    global T

    a, b = 1.01, 10

    xmin = x.min()
    if xmin == 0:
        x = x+1
        xmin = 1

    n = x.shape[0]
    T = -sum(log(x / xmin)) / n

    alpha_hat = bisect(objective, a, b, xtol=1e-6)
    s = sigma(n, alpha_hat, xmin)

    return alpha_hat, s