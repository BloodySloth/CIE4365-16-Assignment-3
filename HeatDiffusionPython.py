
# coding: utf-8

# # Heat Diffusion in Soils
# 
# This Jupyter Notebook gives an example how to implement a 1D heat diffusion model in Python.
# 
# First we need to import the packages which we will be using:
# 

# In[1]:


import numpy as np
import scipy.integrate as spi
import MyTicToc as mt
import matplotlib.pyplot as plt
from collections import namedtuple


# plot figures inline
#%matplotlib inline

# plot figures as interactive graphs...
#%matplotlib qt

# ## Definition of functions
# Then we need to define the functions which we will be using:
# BndTTop for calculating the top temperature as a function of time;
# HeatFlux for calculating all heat fluxes in the domain;
# DivHeatFlux for calculating the divergence of the heat flux across the cells in the domain.


def BndTTop(t, bPar):
    bndT = bPar.avgT - bPar.rangeT * np.cos(2 * np.pi
                                            * (t - bPar.tMin) / 365.25)
    return bndT


def HeatFlux(t, T, sPar, mDim, bPar):
    nIN = mDim.nIN
    nN = mDim.nN
    dzN = mDim.dzN
    lambdaIN = sPar.lambdaIN

    q = np.zeros((nIN, 1))

    # Temperature at top boundary
    bndT = BndTTop(t, bPar)
    # Implement Dirichlet Boundary  Python can mutate a list because these are
    # passed by reference and not copied to local scope...
    # locT = np.ones(np.shape(T)) * T
    locT = T.copy()
    if bPar.topCond.lower() == 'dirichlet':
        locT[nN - 1, 0] = bndT

    # Calculate heat flux in domain
    # Bottom layer Robin condition
    q[0, 0] = 0.0
    q[0, 0] = -bPar.lambdaRobBot * (T[0, 0] - bPar.TBndBot)

    # Flux in all intermediate nodes
    ii = np.arange(1, nIN - 1)
    q[ii, 0] = -lambdaIN[ii, 0] * ((locT[ii, 0] - locT[ii - 1, 0])
                                   / dzN[ii - 1, 0])
    # Top layer
    if bPar.topCond.lower() == 'dirichlet':
        # Temperature is forced, so we ensure that divergence of flux in top
        # layeris zero...
        q[nIN-1, 0] = q[nIN - 2, 0]
    else:
        # Robin condition
        q[nIN-1, 0] = -bPar.lambdaRobTop * (bndT - T[nN-1, 0])

    return q


def DivHeatFlux(t, T, sPar, mDim, bPar):
    nN = mDim.nN
    dzIN = mDim.dzIN
    locT = T.copy()
    zetaBN = np.diag(FillmMatHeat(t, locT, sPar, mDim, bPar ))

    # Calculate heat fluxes accross all internodes
    qH = HeatFlux(t, locT, sPar, mDim, bPar)

    divqH = np.zeros([nN, 1])
    # Calculate divergence of flux for all nodes
    ii = np.arange(0, nN-1)
    divqH[ii, 0] = -(qH[ii + 1, 0] - qH[ii, 0]) \
                   / (dzIN[ii, 0] * zetaBN[ii])

    # Top condition is special
    ii = nN-1
    if bPar.topCond.lower() == 'dirichlet':
        divqH[ii,0] = 0
    else:
        divqH[ii, 0] = -(qH[ii + 1, 0] - qH[ii, 0]) \
                       / (dzIN[ii, 0] * zetaBN[ii])


    divqHRet = divqH # .reshape(T.shape)
    return divqHRet


# ## functions to make the implicit solution easier
# 
# In order to facilitate an easy implementation of the implicit (matrix) solution I implemented three functions:
# Fill_kMat_Heat, Fill_mMat_Heat and Fill_yVec_Heat.


def FillkMatHeat(t, T, sPar, mDim, bPar):
    lambdaIN = sPar.lambdaIN
    zetaBN = sPar.zetaBN

    nN = mDim.nN
    nIN = mDim.nIN
    dzN = mDim.dzN
    dzIN = mDim.dzIN

    lambdaRobTop = bPar.lambdaRobTop
    lambdaRobBot = bPar.lambdaRobBot

    a = np.zeros([nN, 1])
    b = np.zeros([nN, 1])
    c = np.zeros([nN, 1])

    # Fill KMat
    # lower boundary
    # Robin Boundary condition

    a[0, 0] = 0
    b[0, 0] = -(lambdaRobBot / dzIN[0, 0] + lambdaIN[1, 0] / (
                dzIN[0, 0] * dzN[0, 0]))
    c[0, 0] = lambdaIN[1, 0] / (dzIN[0, 0] * dzN[0, 0])

    # middel nodes
    ii = np.arange(1, nN - 1)
    a[ii, 0] = lambdaIN[ii, 0] / (dzIN[ii, 0] * dzN[ii - 1, 0])

    b[ii, 0] = -(lambdaIN[ii, 0] / (dzIN[ii, 0] * dzN[ii - 1, 0])
                 + lambdaIN[ii + 1, 0] / (dzIN[ii, 0] * dzN[ii, 0]))

    c[ii, 0] = lambdaIN[ii + 1, 0] / (dzIN[ii, 0] * dzN[ii, 0])

    # Top boundary
    if bPar.topCond.lower() == 'dirichlet':
        a[nN-1, 0] = 0
        b[nN-1, 0] = -1
        c[nN-1, 0] = 0
    else:
        # Robin condition
        a[nN-1,0] = lambdaIN[nIN-2, 0] / (dzIN[nIN-2, 0] * dzN[nN-2, 0])
        b[nN-1, 0] = -(lambdaIN[nIN-2, 0] / (dzIN[nIN-2, 0] * dzN[nN-2, 0])
                       + lambdaRobTop / dzIN[nIN-2, 0])
        c[nN-1, 0] = 0

    kMat = np.diag(a[1:nN, 0], -1) + np.diag(b[0:nN, 0], 0) + np.diag(c[0:nN - 1, 0], 1)
    return kMat


def FillmMatHeat(t, T, sPar, mDim, bPar):
    zetaBN = sPar.zetaBN
    if bPar.topCond.lower() == 'dirichlet':
        zetaBN[mDim.nN - 1, 0] = 0
    mMat = np.diag(zetaBN.squeeze(), 0)
    return mMat


def FillyVecHeat(t, T, sPar, mDim, bPar):
    nN = mDim.nN

    yVec = np.zeros([nN, 1])

    # Bottom Boundary
    yVec[0, 0] = bPar.lambdaRobBot / mDim.dzIN[0,0] * bPar.TBndBot

    # Top Boundary (Known temperature)
    if bPar.topCond.lower() == 'dirichlet':
        yVec[nN-1, 0] = BndTTop(t, bPar)
    else:
        # Robin condition
        yVec[nN-1, 0] = bPar.lambdaRobTop / mDim.dzIN[mDim.nIN-2,0] * BndTTop(t,bPar)

    return yVec


def JacHeat(t, T, sPar, mDim, bPar):
    # Function calculates the jacobian matrix for the Richards equation
    nN = mDim.nN
    locT = T.copy().reshape(nN, 1)
    kMat = FillkMatHeat(t, locT, sPar, mDim, bPar)
    massMD = np.diag(FillmMatHeat(t, locT, sPar, mDim, bPar)).copy()

    a = np.diag(kMat, -1).copy()
    b = np.diag(kMat, 0).copy()
    c = np.diag(kMat, 1).copy()

    if bPar.topCond.lower() == 'dirichlet':
        # massMD(nN-1,1) = 0 so we cannot divide by massMD but we know that the
        # Jacobian should be zero so we set b[nN-1,0] to zero instead and
        # massMD[nN-1,0] to 1.
        b[nN-1] = 0
        massMD[nN-1] = 1

    jac = np.zeros((3, nN))
    a = a / massMD[1:nN]
    b = b / massMD[0:nN]
    c = c / massMD[0:nN - 1]
    # jac[0,0:nN-1] = a[:]
    # jac[1,0:nN] = b[:]
    # jac[2,0:nN-1] = c[:]
    jac = np.diag(a, -1) + np.diag(b, 0) + np.diag(c, 1)
    return jac



def main():
    # Then we start running our model.
    # First we require the domain discretization

    # Domain
    nIN = 151
    # soil profile until 15 meters depth
    zIN = np.linspace(-15, 0, num=nIN).reshape(nIN, 1)
    # nIN = np.shape(zIN)[0]
    zN = np.zeros(nIN - 1).reshape(nIN - 1, 1)
    zN[0, 0] = zIN[0, 0]
    zN[1:nIN - 2, 0] = (zIN[1:nIN - 2, 0] + zIN[2:nIN - 1, 0]) / 2
    zN[nIN - 2, 0] = zIN[nIN - 1]
    nN = np.shape(zN)[0]

    ii = np.arange(0, nN - 1)
    dzN = (zN[ii + 1, 0] - zN[ii, 0]).reshape(nN - 1, 1)
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

    # Soil Properties
    # [J/(m3 K)] volumetric heat capacity of soil solids
    zetaSol = 2.235e6
    # [J/(m3 K)] volumetric heat capacity of water (Fredlund 2006)
    zetaWat = 4.154e6

    # rhoW = 1000  # [kg/m3] density of water
    rhoS = 2650  # [kg/m3] density of solid phase
    rhoB = 1700  # %[kg/m3] dry bulk density of soil
    n = 1 - rhoB / rhoS  # [-] porosity of soil = saturated water content.
    q = 0.75  # quartz content

    # [W/(mK)] thermal conductivity of water (Remember W = J/s)
    lambdaWat = 0.58
    lambdaQuartz = 6.5  # [W/(mK)] thermal conductivity of quartz
    lambdaOther = 2.0  # [W/(mK)] thermal conductivity of other minerals

    lambdaSolids = lambdaQuartz ** q * lambdaOther ** (1 - q)
    lambdaBulk = lambdaWat ** n * lambdaSolids ** (1 - n)

    # collect soil parameters in a namedtuple: soilPar
    soilPar = namedtuple('soilPar', ['zetaBN', 'lambdaIN'])
    sPar = soilPar(zetaBN=np.ones(np.shape(zN)) * ((1 - n) * zetaSol
                                                    + n * zetaWat),
                   lambdaIN=np.ones(np.shape(zIN)) * lambdaBulk * (24 * 3600))


    # ## Definition of the Boundary Parameters
    # boundary parameters
    # collect boundary parameters in a named tuple boundpar...
    boundPar = namedtuple('boundPar', ['avgT', 'rangeT', 'tMin', 'topCond',
                                       'lambdaRobTop','lambdaRobBot',
                                       'TBndBot'])
    bPar = boundPar(avgT=273.15 + 10,
                    rangeT=20,
                    tMin=46,
                    topCond='Dirichlet',
                    lambdaRobTop=1,
                    lambdaRobBot=0,
                    TBndBot=273.15 + 10)


    # ## Initial Conditions
    # Initial Conditions
    TIni = np.ones(np.shape(zN)) * (10.0 + 273.15)  # K
    # Time Discretization
    tOut = np.linspace(0, 365.25 * 10, 365)  # time
    nOut = np.shape(tOut)[0]

    # ## Implement using the built-in ode solver

    mt.tic()

    def intFun(t, y):
        nf = DivHeatFlux(t, y, sPar, mDim, bPar)
        return nf

    def jacFun(t, y):
        jac = JacHeat(t, y, sPar, mDim, bPar)
        return (jac)


    T0 = TIni.copy().squeeze()
    TODE = spi.solve_ivp(intFun, [tOut[0], tOut[-1]], T0, method='BDF',
                         t_eval=tOut, vectorized=True, rtol=1e-8, jac=jacFun)
    # Dirichlet boundary condition: write boundary temperature to output.
    for iiOut in np.arange(0, nOut):
        TODE.y[nN - 1, iiOut] = BndTTop(tOut[iiOut], bPar)
    mt.toc()

    plt.close('all')
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    for ii in np.arange(0, nN, 20):
        ax1.plot(tOut, TODE.y[ii, :], '-')
    ax1.set_title('Temperature (ODE)')

    fig2, ax2 = plt.subplots(figsize=(4, 7))
    for ii in np.arange(0, nOut, 20):
        ax2.plot(TODE.y[:, ii], zN, '-')

    ax2.set_title('Temperature vs. depth (ODE)')
    # plt.show()

    # In order to run own implementation (slow) chage RunMyOwn to True
    RunMyOwn = False
    if RunMyOwn:
        # ## My own explicit implementation based on the Runge Kutta  Solver
        # RungeKutta
        # Initialize output vector
        TRK = np.zeros([nN, nOut], dtype=float)
        dtMax = 0.1
        dtMin = 1e-9
        # dtMin = 1e-11
        t = tOut[0]
        iiOut = 0
        # Initialize problem
        mt.tic()
        T = TIni.copy()
        # Write initial values to output vector
        TRK[:, iiOut] = T.reshape(1, nN)

        # Write initial condition to top boundary
        TRK[nN - 1, iiOut] = BndTTop(t, bPar)

        while t < tOut[nOut - 1]:
            # check time steps
            R1 = DivHeatFlux(t, T, sPar, mDim, bPar)

            minT = bPar.avgT-bPar.rangeT
            idx = (R1 == 0)

            dtEst = -0.5 * (T[~idx] - minT) / R1[~idx]
            dtRate = dtEst[np.isfinite(dtEst)]
            # dtRate = np.sign(dtRate) * (np.isinf(dtRate) * 1000)
            # We only need to take the positive rates in to account
            dtRate = (dtRate < 0) * dtMax + (dtRate >= 0) * dtRate
            dtOut = tOut[iiOut + 1] - t
            # dt = min(min(dtRate), dtOut, dtMax)
            dt = min(min(dtRate), dtOut, dtMax)
            dt = max(dtMin, dt)

            tmpT = T+R1*dt/2
            R2 = DivHeatFlux(t + dt/2, tmpT, sPar, mDim, bPar)

            tmpT = T + R2*dt/2
            R3 = DivHeatFlux(t + dt/2, tmpT, sPar, mDim, bPar)

            tmpT = T + R3*dt
            R4 = DivHeatFlux(t + dt, tmpT, sPar, mDim, bPar)

            T = T + (R1 + 2*R2 + 2*R3 + R4)*dt/6
            # T = T + k1
            t = t + dt

            if np.abs(tOut[iiOut + 1] - t) < 1e-5:
                iiOut += 1
                TRK[:, iiOut] = T[:, 0]

                # Dirichlet boundary condition: write boundary
                # temperature to output.
                TRK[nN-1, iiOut] = BndTTop(t, bPar)

        mt.toc()

        fig3, ax1 = plt.subplots(figsize=(7, 4))
        for ii in np.arange(0, nN, 20):
            ax1.plot(tOut, TRK[ii, :], '-')
        ax1.set_title('Temperature (RK)')

        fig4, ax2 = plt.subplots(figsize=(4, 7))
        for ii in np.arange(0, nOut, 20):
            ax2.plot(TRK[:, ii], zN, '-')

        ax2.set_title('Temperature vs. depth (RK)')
        # plt.show()
        # plt.savefig('myfig.png')

        # Fully implict own implementation
        # Initialize output vector
        TImp = np.zeros([nN, nOut], dtype=float)
        dtMax = 0.1
        dtMin = 1e-9
        # dtMin = 1e-11
        t = tOut[0]
        iiOut = 0
        # Initialize problem
        mt.tic()
        T = TIni
        # Write initial values to output vector
        TImp[:, iiOut] = T.reshape(1, nN)

        # Write initial condition to top boundary
        TImp[nN - 1, iiOut] = BndTTop(t, bPar)

        convcrit = 1e-11

        while t < tOut[nOut - 1]:
            # not yet implemented
            dtRate = dtMax

            dtRate = (dtRate < 0) * dtMax + (dtRate >= 0) * dtRate
            dtOut = tOut[iiOut + 1] - t

            dt = min(dtRate, dtOut, dtMax)
            dt = max(dtMin, dt)

            # Fill the values for the K matrix, Mass matrix and yVector
            kMat = FillkMatHeat(t + dt, T, sPar, mDim, bPar)
            yVec = FillyVecHeat(t + dt, T, sPar, mDim, bPar)
            mMat = FillmMatHeat(t + dt, T, sPar, mDim, bPar)

            # Integrate by solving the matrix equation
            M1 = (mMat / dt - kMat)
            M2 = np.dot(mMat / dt, T) + yVec

            T = np.linalg.solve(M1, M2)
            t = t + dt

            if np.abs(tOut[iiOut + 1] - t) < 1e-5:
                iiOut += 1
                TImp[:, iiOut] = T[:, 0]

                # Dirichlet boundary condition: write boundary
                # temperature to output.
                TImp[nN - 1, iiOut] = BndTTop(t, bPar)

        mt.toc()

        fig5, ax1 = plt.subplots(figsize=(7, 4))
        for ii in np.arange(0, nN, 20):
            ax1.plot(tOut, TImp[ii, :], '-')
        ax1.set_title('Temperature (Implicit)')

        fig6, ax2 = plt.subplots(figsize=(4, 7))
        for ii in np.arange(0, nOut, 20):
            ax2.plot(TImp[:, ii], zN, '-')

        ax2.set_title('Temperature vs. depth (Implicit)')

        residuals = TImp - TODE.y
        squared_res = residuals*residuals
        ssr = np.sum(squared_res,0)
        fig7, ax2 = plt.subplots(figsize=(7, 4))
        ax2.plot(tOut, ssr)

        ax2.set_title('sum of squared residuals (Implicit v.s. ODE)')

    plt.show()
    # plt.savefig('myfig.png')

if __name__ == "__main__":
    main()
