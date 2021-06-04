
import numpy as np
import scipy.integrate as spi
import MyTicToc as mt
import matplotlib.pyplot as plt
from collections import namedtuple

# plot figures inline
# %matplotlib inline

# plot figures as interactive graphs...
# %matplotlib qt

### Definition of functions
# Then we need to define the functions which we will be using:
# BndTTop for calculating the rain as a function of time;
# watertFlux for calculating all water flows in the domain;
# DivwaterFlux for calculating the divergence of the waterflow across the cells in the domain.


# make the temperature viscosity table
counnt = 0


def TempVis(T):
    import scipy.interpolate as spint
    Tini = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    mu = [1.787, 1.519, 1.307, 1.002, 0.798, 0.653, 0.547, 0.467, 0.404, 0.355, 0.315, 0.282]  # MPa/s
    p = spint.interp1d(Tini, mu, fill_value='extrapolate')
    mu1 = p(T - 273.15) * 1e-3
    return mu1


# make Ksat          #maybe make array from this?
def Ksat(T, mDim):
    #Ksat = sPar.Ksat1 / TempVis(T) * sPar.rhoW * 9.81
    #Ksat = sPar.Ksat1 * (TempVis(298.15) / TempVis(T))
    Ksat=np.ones(np.shape(mDim.zN))*0.05
    return Ksat


def lambda_fun(hw, sPar, mDim):  # Gaat dit goed met de definitie theta? Moeten we die msischien eerder oproepen? check als we het runnen
    nN = mDim.nN
    nIN = mDim.nIN

    thetaw = Theta(hw, sPar)
    Sw = thetaw / sPar.theta_sat
    por = sPar.theta_sat  # porosity n

    ls = sPar.lambdas
    lw = sPar.lambdaw
    la = sPar.lambdada + (sPar.lambdava * Sw)
    g1 = 0.015 + (0.333 - 0.015) * Sw
    g3 = 1 - g1 - g1

    Fw = sPar.Fw
    Fa = 1 / 3 * (
            ((1 + (la / lw - 1) * g1) ** -1) + ((1 + (la / lw - 1) * g1) ** -1) + ((1 + (la / lw - 1) * g3) ** -1))
    Fs = 1 / 3 * (
            ((1 + (ls / lw - 1) * g1) ** -1) + ((1 + (ls / lw - 1) * g1) ** -1) + ((1 + (ls / lw - 1) * g3) ** -1))

    lamtot = (Fs * ls * (1 - por) + Fw * lw * por * Sw + Fa * la * por * (1 - Sw)) \
             / (Fs * (1 - por) + Fw * por * Sw + Fa * por * (1 - Sw))

    lamtotIN = np.zeros(np.shape(mDim.zIN), dtype=hw.dtype)
    lamtotIN[0,0] = lamtot[0,0]
    i = np.arange(1, nIN - 1)
    lamtotIN[i,0] = np.minimum(lamtot[i - 1,0], lamtot[i,0])
    lamtotIN[nIN - 1,0] = lamtot[nIN - 2,0]

    return lamtotIN * 24 * 3600


def zeta_fun(hw, sPar):  # zet zetaSol, zetaWat en zetaAir in assignment_3.py of leest het heat diff.?
    hw = np.real(hw.copy())
    thetaw = Theta(hw, sPar)
    Sw = thetaw / sPar.theta_sat
    rhoS = 2650  # [kg/m3] density of solid phase
    rhoB = 1700  # %[kg/m3] dry bulk density of soil
    por = 1 - rhoB / rhoS  # [-] porosity of soil = saturated water content.
    zetas = sPar.zetaSol
    zetaw = sPar.zetaWat  # the heat capacity of air phase is neglected: zetaAir is neglected

    zetatot = zetas * (1 - por) + zetaw * Sw * por
    return zetatot


# making S effective
def SEFF(hw, sPar):
    hc = -hw  # ha is zero
    Seff = (1 + ((hc * (hc > 0)) * sPar.alpha) ** sPar.n) ** -sPar.m
    return Seff


# making theta
def Theta(hw, sPar):
    Seff = SEFF(hw, sPar)
    theta_w = Seff * (sPar.theta_sat - sPar.theta_res) + sPar.theta_res
    return theta_w


# making krw (relative permeability) for every hw
def krwfun(hw, sPar, mDim):
    nIN = mDim.nIN
    Seff = SEFF(hw, sPar)
    nr, nc = hw.shape

    krw = Seff ** 3
    i = np.arange(1, nIN - 1)
    Krw = np.zeros((nIN, nc))
    Krw[0] = krw[0]
    Krw[i] = np.minimum(krw[i - 1], krw[i])
    Krw[-1] = krw[-1]
    return Krw


# boundary top condition
def BndwTop(t, bPar):  # t> 25 and 200 -0.001
    bndT = -0.001 * (t > bPar.tMin) * (t < bPar.tMax)
    return bndT


def BndTTop(t, bPar):
    bndT = bPar.avgT - bPar.rangeT * np.cos(2 * np.pi
                                            * (t - bPar.tMin) / 365.25)
    return bndT


# part of the richardson equation Ksat*krw(gradient*hw+flux*z))
def waterFlux(t, T, hw, sPar, mDim, bPar):
    nIN = mDim.nIN

    dzN = mDim.dzN

    ksat = Ksat(T, mDim)
    krw = krwfun(hw, sPar, mDim)
    nr, nc = hw.shape
    q = np.zeros((nIN, nc))

    # Flux in all intermediate nodes
    i = np.arange(1, nIN - 1)
    q[i] = -ksat[i] * krw[i] * ((hw[i] - hw[i - 1]) / dzN[i - 1] + 1)

    # Lower boundary:
    if bPar.botCon == "ROBIN":
        q[0] = -bPar.krob * (hw[0] - bPar.h0)
    if bPar.botCon == "GRAVITY":
        q[0] = -ksat[0] * krw[0]

    # top boundary conditions
    # bndw = BndwTop(t, bPar)
    q[nIN - 1] = 0  # or bndw
    return q


def HeatFlux(t, T, hw, sPar, mDim, bPar):
    nIN = mDim.nIN
    nN = mDim.nN
    dzN = mDim.dzN
    lambdaIN = lambda_fun(hw, sPar, mDim)
    zetaWat = sPar.zetaWat
    nr, nc = T.shape
    qw = waterFlux(t, T, hw, sPar, mDim, bPar)

    ql = np.zeros([nIN, nc])
    qz = np.zeros([nIN, nc])

    # Temperature at top boundary
    bndT = BndTTop(t, bPar)

    # Calculate heat flux in domain
    # Bottom layer Robin condition
    ql[0] = 0.0
    ql[0] = -bPar.lambdaRobBot * (T[0] - bPar.TBndBot)

    qz[0] = 0.0
    qz[0] = qw[0] * zetaWat[0] * (bPar.TBndBot * (qw[0] >= 0)
                                  + T[0] * (qw[0] < 0))

    # Flux in all intermediate nodes
    i = np.arange(1, nIN - 1)
    ql[i] = -lambdaIN[i] * ((T[i] - T[i - 1])
                            / dzN[i - 1])

    qz[i] = zetaWat[0] * qw[i] * (T[i - 1] * (qw[i] >= 0) +
                                  T[i] * (qw[i] < 0))

    # Top layer

    # Robin condition
    ql[nIN - 1] = -bPar.lambdaRobTop * (bndT - T[nN - 1])
    qz[nIN - 1] = zetaWat[0] * qw[nIN - 1] * (T[nN - 1] * (qw[nIN - 1] >= 0) +
                                              bndT * (qw[nIN - 1] < 0))

    qh = ql + qz
    return qh


# def Cfun(sPar, hw):
#    hc=-hw
#    Seff=SEFF(sPar,hw)
#    dSeffdh = sPar.alpha*sPar.m/ (1-sPar.m)*Seff**(1/sPar.m)*\(1-Seff**(1/sPar.m))**sPar.m*(hc>0)+(hc<=0) * 0
#    return (sPar.theta_sat-sPar.theta_res)*dSeffdh


# Part of the richardson equation (C(hw)+Sw*Ss**w)
def Cfuncmplx(hw, sPar):
    dh = np.sqrt(np.finfo(float).eps)
    if np.iscomplexobj(hw):
        hcmplx = hw.real + 1j * dh
    else:
        hcmplx = hw.real + 1j * dh
    theta = Theta(hcmplx, sPar)
    C = theta.imag / dh
    return C


def Caccentfun(hw, sPar, mDim):
    theta = Theta(hw, sPar)
    Sw = theta / sPar.theta_sat
    Chw = Cfuncmplx(hw, sPar)
    beta = 4.5e-10
    Sws = sPar.rhoW * 9.81 * (sPar.Cv + sPar.theta_sat * beta)
    C = Chw + Sw * Sws

    C[mDim.nN - 1] = 1 / mDim.dzIN[mDim.nIN - 2] * (hw[mDim.nN - 1] > 0) + C[mDim.nN - 1] * (hw[mDim.nN - 1] <= 0)
    return C


def dhwdtFun(t, T, hw, sPar, mDim, bPar):
    nr, nc = hw.shape
    nN = mDim.nN
    dzIN = mDim.dzIN

    divqW = np.zeros([nN, nc]).astype(hw.dtype)

    C = Caccentfun(hw, sPar, mDim)

    qW = waterFlux(t, T, hw, sPar, mDim, bPar)

    # Calculate divergence of flux for all nodes
    i = np.arange(0, nN)
    divqW[i] = -(qW[i + 1] - qW[i]) / (dzIN[i] * C[i])

    return divqW

def FillmMatHeat(t, T, hw, sPar, mDim, bPar):
    zetaBN = sPar.zetaBN
    if bPar.topCond.lower() == 'dirichlet':
        zetaBN[mDim.nN - 1, 0] = 0
    mMat = np.diag(zetaBN.squeeze(), 0)
    return mMat

def DivHeatFlux(t, T, hw, sPar, mDim, bPar):
    nN = mDim.nN
    dzIN = mDim.dzIN
    lochw = hw.copy()
    locT = T.copy()
    zetaBN= np.diag(FillmMatHeat(t, locT, lochw, sPar, mDim, bPar))
    nr, nc = T.shape

    # Calculate heat fluxes accross all internodes
    qH = HeatFlux(t, lochw, locT, sPar, mDim, bPar)

    divqH = np.zeros((nN, 1), dtype=hw.dtype)
    # Calculate divergence of flux for all nodes
    i = np.arange(0, nN - 1)
    divqH[i,0] = -(qH[i + 1,0] - qH[i,0]) \
               / (dzIN[i,0] * zetaBN[i])

    # Top condition is special
    i = nN - 1
    if bPar.topCond.lower() == 'dirichlet':
        divqH[i,0] = 0
    else: 
        divqH[i,0] = -(qH[i + 1,0] - qH[i,0]) / (dzIN[i,0] * zetaBN[i])
    
    ii = np.arange(0, nN-1)
    dzetadt = np.zeros((nN,1), dtype=hw.dtype)
    dthetadt = dhwdtFun(t, locT, lochw, sPar, mDim, bPar)
    dzetadtheta = 4.154e6
    dzetadt[ii,0]=dzetadtheta*dthetadt[ii,0]
    
    divqHRet = np.zeros((nN,1), dtype = hw.dtype)
    divqHRet[ii,0] = divqH[ii,0] -locT[ii,0] * dzetadt[ii,0] / zetaBN[i]
    
    #top condition is special
    ii = nN-1
    if bPar.topCond.lower() == 'dirichlet':
        divqHRet[ii,0] = 0
    else:
        divqHRet[ii,0] = divqH[ii,0] - locT[ii,0] * dzetadt[ii,0] / zetaBN[ii]

    return divqHRet

# def DivHeatFlux(t, T, hw, sPar, mDim, bPar):
#     nN = mDim.nN
#     dzIN = mDim.dzIN
#     zeta = zeta_fun(hw, sPar)
#     nr, nc = T.shape

#     # Calculate heat fluxes accross all internodes
#     qH = HeatFlux(t, hw, T, sPar, mDim, bPar)

#     divqH = np.zeros([nN, nc])
#     # Calculate divergence of flux for all nodes
#     i = np.arange(0, nN - 1)
#     divqH[i] = -(qH[i + 1] - qH[i]) \
#                / (dzIN[i] * zeta[i])

#     # Top condition is special
#     i = nN - 1
#     divqH[i] = -(qH[i + 1] - qH[i]) \
#                    / (dzIN[i] * zeta[i])

#     divqHRet = divqH  # .reshape(T.shape)
#     return divqHRet


# delete this when it is not necessary
def divCoupled(t, y, sPar, mDim, bPar):
    zN = mDim.zN
    nN = mDim.nN
    dzIN = mDim.dzIN
    sdim = y.shape

    if len(sdim) == 1:
        y = y.reshape((2 * nN, 1))

    divqw = np.zeros([nN, 1], dtype=y.dtype)
    divqh = np.zeros([nN, 1], dtype=y.dtype)

    dhwdt = np.zeros([nN, 1], dtype=y.dtype)
    dTdt = np.zeros([nN, 1], dtype=y.dtype)

    totaldif = np.zeros([2 * nN, 1], dtype=y.dtype)

    hw = y[0:nN, 0].reshape(mDim.zN.shape)
    T = y[nN:2 * nN, 0].reshape(mDim.zN.shape)

    # pressure head differential equation
    qw = waterFlux(t, T, hw, sPar, mDim, bPar)
    C = Caccentfun(hw, sPar, mDim)
    i = np.arange(0, nN)
    divqw[i, 0] = -(qw[i + 1, 0] - qw[i, 0]) \
                  / (dzIN[i, 0])
    dhwdt = divqw / C

    # temperature differential equation
    qh = HeatFlux(t, T, hw, sPar, mDim, bPar)
    zeta = zeta_fun(hw, sPar)
    i = np.arange(0, nN)
    divqh[i, 0] = -(qh[i + 1, 0] - qh[i, 0]) \
                  / (dzIN[i, 0])
    dTdt = (divqh - (T[i] * (sPar.zetaWat - sPar.zetaAir) * divqw)) / zeta[i]

    totaldif = np.vstack([dhwdt, dTdt])

    return totaldif


# def main():
# Then we start running our model.
# First we require the domain discretization

# Domain
nIN = 101
# soil profile until 1 meters depth
zIN = np.linspace(-1, 0, num=nIN).reshape(nIN, 1)
# nIN = np.shape(zIN)[0]
zN = np.zeros(nIN - 1).reshape(nIN - 1, 1)
zN[0, 0] = zIN[0, 0]
zN[1:nIN - 2, 0] = (zIN[1:nIN - 2, 0] + zIN[2:nIN - 1, 0]) / 2
zN[nIN - 2, 0] = zIN[nIN - 1]
nN = np.shape(zN)[0]

i = np.arange(0, nN - 1)
dzN = (zN[i + 1, 0] - zN[i, 0]).reshape(nN - 1, 1)
dzIN = (zIN[1:, 0] - zIN[0:-1, 0]).reshape(nIN - 1, 1)

# collect model dimensions in a namedtuple: modDim
modDim = namedtuple('ModDim', ['zN', 'zIN', 'dzN', 'dzIN', 'nN', 'nIN'])
mDim = modDim(zN=zN,
              zIN=zIN,
              dzN=dzN,
              dzIN=dzIN,
              nN=nN,
              nIN=nIN)

# ## Definition of material properties
# In this section of the code we define the material properties

# Soil Properties SAND from table 2.4 Mayer & Hassanizadeh ch 2

rhoW = 1000  # [kg/m3] density of water
theta_res = 0.045
theta_sat = 0.38
n = 3
m = 1 - (1 / 3)
alpha = 2  # m^-1
Cv = 1e-8
Ksat1 = 0.05  # m/day
lambdas = 6
lambdaw = 0.57
lambdada = 0.025
lambdava = 0.0736
zetaSol = 2.235e6
zetaWat = 4.154e6
Fw = 1

# collect soil parameters in a namedtuple: soilPar
soilPar = namedtuple('soilPar',
                     ['rhoW', 'n', 'm', 'alpha', 'Cv', 'Ksat1', 'theta_res', 'theta_sat', 'lambdas', 'lambdaw',
                      'lambdada', 'lambdava', 'zetaSol', 'zetaWat', 'Fw', 'zetaBN'])
sPar = soilPar(rhoW=np.ones(np.shape(zN)) * rhoW, n=np.ones(np.shape(zN)) * n, m=np.ones(np.shape(zN)) * m,
               alpha=np.ones(np.shape(zN)) * alpha, Cv=np.ones(np.shape(zN)) * Cv, Ksat1=np.ones(np.shape(zN)) * Ksat1,
               theta_res=np.ones(np.shape(zN)) * theta_res,
               theta_sat=np.ones(np.shape(zN)) * theta_sat, lambdas=np.ones(np.shape(zN)) * lambdas,
               lambdaw=np.ones(np.shape(zN)) * lambdaw, lambdada=np.ones(np.shape(zN)) * lambdada,
               lambdava=np.ones(np.shape(zN)) * lambdava, zetaSol=np.ones(np.shape(zN)) * zetaSol,
               zetaWat=np.ones(np.shape(zN)) * zetaWat, Fw=np.ones(np.shape(zN)) * Fw, 
               zetaBN=np.ones(np.shape(zN)) * ((1 - n) * zetaSol + n * zetaWat))

# ## Definition of the Boundary Parameters
# boundary parameters
# collect boundary parameters in a named tuple boundpar...
boundPar = namedtuple('boundPar',
                      ['avgT', 'rangeT', 'tMin', 'tMax', 'topCond', 'botCon', 'h0', 'krob', 'lambdaRobTop', 'lambdaRobBot',
                       'TBndBot'])
bPar = boundPar(avgT=273.15 + 10,
                rangeT=20,
                tMin=25,
                tMax=300,
                topCond='ROBIN',
                botCon='ROBIN',
                h0=-1,
                krob=0.01,
                lambdaRobTop=1,
                lambdaRobBot=0,
                TBndBot=273.15 + 10)

# ## Initial Conditions
# Initial Conditions
HIni = -0.75 - zN
TIni = np.ones(np.shape(zN)) * (10.0 + 273.15)  # K
YIni = np.vstack([HIni[0], TIni[0]])

# Time Discretization
tOut = np.linspace(0, 600, 120)  # time
nOut = np.shape(tOut)[0]

# ## Implement using the built-in ode solver

mt.tic()


def intFun(t, y):
    hw = y[0:nN]
    T = y[nN:2 * nN]
    nf = dhwdtFun(t, T, hw, sPar, mDim, bPar)
    nv = DivHeatFlux(t, T, hw, sPar, mDim, bPar)
    dhwdT = np.vstack([nf, nv])
    return dhwdT

def jacFun(t, y):
    if len(y.shape) == 1:
        y = y.reshape(mDim.nN, 1)

    nr, nc = y.shape
    dh = np.sqrt(np.finfo(float).eps)
    ycmplx = np.repeat(y, nr, axis=1).astype(complex)
    c_ex = np.eye(nr) * 1j * dh
    ycmplx = ycmplx + c_ex
    dfdy = intFun(t, ycmplx).imag / dh  # make T complex
    return spi.coo_matrix(dfdy)

# def jac_complex(t, y):
#     if len(y.shape) == 1:
#         y = y.reshape(mDim.nN, 1)
        
#     nr, nc = y.shape
#     dh = np.sqrt(np.finfo(float).eps)
#     ycmplx = np.repeat(y, nr, axis=1).astype(complex)
#     jac=np.zeros((nr,nr))
#     for i in np.arange(nr):
#         ycmplx=y.copy().astype(complex)
#         ycmplx[i]=ycmplx[i]+1j*dh
#         dfdy=intFun(t, ycmplx).imag/dh
#         jac[:,i]=dfdy.squeeze()
#     return jac  

# H0 = HIni.squeeze()
# print(H0)
HODE = spi.solve_ivp(intFun, [tOut[0], tOut[-1]], YIni.squeeze(), method='BDF',
                     t_eval=tOut, vectorized=True, rtol=1e-6)
mt.toc()

hw = HODE.y[0:nN,:]
T = HODE.y[nN:2 * nN,:]

plt.close('all')
fig1, ax1 = plt.subplots(figsize=(7, 4))
for ii in np.arange(0, nN, 10):
    ax1.plot(HODE.t, hw[ii,:], '-', label=ii)
ax1.legend()
ax1.set_title('Water pressure head against time')
ax1.grid(b=True)
ax1.set_xlabel('time [days]')
ax1.set_ylabel('Water pressure head [m]')

fig2, ax2 = plt.subplots(figsize=(4, 7))

for ii in np.arange(0, nN, 10):
    ax2.plot(HODE.t, T[ii,:], '-')
ax2.set_title('Time against Temperature')
# ax2.legend()
ax2.grid(b=True)
ax2.set_xlabel('time [days]')
ax2.set_ylabel('Temperature [Kelvin]')


fig3, ax3 = plt.subplots(figsize=(7, 7))
for ii in np.arange(0, nOut, 10):
    ax3.plot(hw[:,ii], zN, '-', label=ii)
ax3.grid(b=True)
# ax3.legend()
ax3.set_title('Water pressure head against depth')
ax3.set_xlabel('water pressure head [m]')
ax3.set_ylabel('depth [m]')
plt.show()

fig4, ax4 = plt.subplots(figsize=(7, 7))
for ii in np.arange(0, nOut, 10):
    ax4.plot(T[:,ii], zN, '-', label=ii)
ax4.grid(b=True)
# ax3.legend()
ax4.set_title('Temperature against depth ')
ax4.set_xlabel('Temperature [Kelvin]')
ax4.set_ylabel('depth [m]')
plt.show()
# plt.savefig('myfig.png')

# if __name__ == "__main__":
# main()
