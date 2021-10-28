# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema

LENTO20_XY = np.loadtxt('Matrices/20%LENTO.txt',delimiter= ',', skiprows= 3, usecols = (0,1,2,3))
LENTO36_XY = np.loadtxt('Matrices/36%LENTO.txt',delimiter= ',', skiprows= 3, usecols = (0,1,2,3))
LENTO50_XY = np.loadtxt('Matrices/50%LENTO.txt',delimiter= ',', skiprows= 3, usecols = (0,1,2,3))
RAPIDO20_XY = np.loadtxt('Matrices/20%RAPIDO.txt',delimiter= ',', skiprows= 3, usecols = (0,1,2,3))
RAPIDO36_XY = np.loadtxt('Matrices/36%RAPIDO.txt',delimiter= ',', skiprows= 3, usecols = (0,1,2,3))
RAPIDO50_XY = np.loadtxt('Matrices/50%RAPIDO.txt',delimiter= ',', skiprows= 3, usecols = (0,1,2,3))
#en la columna 0 tenemos la posicion: X
#en la columna 1 tenemos la posicion: Y
#en la columna 2 tenemos la velocidad en X: U
#en la columna 2 tenemos la velocidad en Y: V
#centros de las matrices
centros = np.loadtxt('Matrices/centrosMatrices.txt',delimiter= ',', skiprows= 3, usecols = (0,1))

#Para hacerlo mas agil despues, definimos unas funciones ahora
#Funcion que transforma las matrices de pos y vel en cartesianas a polares
def converTo_polares(MXY):
    MRT= np.empty((len(MXY), 4), float)
    for i in range(len(MXY)):
        x = MXY[i][0]
        y = MXY[i][1]
        u = MXY[i][2]
        v = MXY[i][3]
        
        #el modulo de (x,y)
        r = np.sqrt(x**2 + y**2)
        #el angulo en radianes de (x,y)
        t = np.angle(x + y*1j)
        
        MRT[i][0] = r
        MRT[i][1] = t
        
        #defino las velocidades como 0 inicialmente
        vrad = 0
        vtan = 0
        
        #hay que tener cuidado con la posicion de la matriz que esta en el origen, pues ahi r = 0
        if r != 0 :
            #vel radial
            vrad = (1/r)*(x*u + y*v) 
            #vel tangencial r*(titapunto)
            vtan = (1/r)*(x*v - y*u)
            
        MRT[i][2] = vrad
        MRT[i][3] = vtan
    return MRT

def centrarX_aux(MXY):
    heightMXY = len(MXY)
    duplaX_sumV = []
    sumVcol = 0
    for i in range(heightMXY):
        Vtot_ix =  np.sqrt(MXY[i][2]**2 + MXY[i][3]**2)
        sumVcol+= Vtot_ix
        if i != heightMXY-1 and MXY[i][0] != MXY[i+1][0]:
            duplaX_sumV.append([MXY[i][0], sumVcol])
            sumVcol = 0
    MX_sumV = np.array(duplaX_sumV)
    plt.figure()
    plt.title('Obtencion del X_o',fontsize = 15)
    plt.plot(MX_sumV[:,0], MX_sumV[:,1], '.')
    plt.xlabel("X [m]", fontsize=15)
    plt.ylabel("Suma de |V| de la columna en x [m/s]", fontsize = 15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)     
       
def centrarY_aux(MXY):
    heightMXY = len(MXY)
    MXYys = np.array(sorted(MXY , key=lambda x: x[1]))
    duplaY_sumV = []
    sumVfil = 0
    for i in range(heightMXY):
        Vtot_iy = np.sqrt(MXYys[i][2]**2 + MXYys[i][3]**2)
        sumVfil+= Vtot_iy
        if i!=heightMXY-1 and MXYys[i][1] != MXYys[i+1][1]:
            duplaY_sumV.append([MXYys[i][1], sumVfil])
            sumVfil = 0
    MY_sumV = np.array(duplaY_sumV)
    plt.figure()
    plt.title('Obtencion del Y_o',fontsize = 15)
    plt.plot(MY_sumV[:,0], MY_sumV[:,1], '.')
    plt.xlabel("Y [m]", fontsize=15)
    plt.ylabel("Suma de |V| de la fila en y [m/s]", fontsize = 15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
#esta funcion homogeneiza los graficos que tienen mil de dispersion
def homogVtan_R(MRT, p):
    heightMRT = len(MRT)
    #MRT es la matriz que contiene en la primer y cuarta columna los datos a promediar
    #p es la cantidad de datos que quiero obtener del promediado, si p->inf se reobtiene los datos originales
    
    #cuantos puntos quiero para la curva?, se define al definir la cantidad de bloques en los que se promedia
    blockBorders = np.linspace(min(MRT[:,0]), max(MRT[:,0]), p)
    #se devuelve esta lista transformada a array
    r_prom_Vtan = []
    
    for j in range(p):
        rs_toprom = []
        vt_toprom = []
        for i in range(heightMRT):
            if j!= p - 1 and blockBorders[j] < MRT[i][0] < blockBorders[j+1]:     
                vt_toprom.append(MRT[i][3])
                rs_toprom.append(MRT[i][0])     
            #print(blockBorders[j])
            if j == p - 1 and blockBorders[j-2] < MRT[i][0] :
                vt_toprom.append(MRT[i][3])
                rs_toprom.append(MRT[i][0])               
        
        if len(rs_toprom) != 0:
            r_prom_Vtan.append([np.mean(rs_toprom),np.mean(vt_toprom)])
        
        matrizRp_Vtp = np.array(r_prom_Vtan)
    return matrizRp_Vtp

#Funcion de ajuste de V_tita(r)
def func(x,a,b,c):
    return ((a/(x-c))*(1 - np.exp(-((x-c)**2/b**2))))

#%%
#centramos las matrices
matrices_XY = [LENTO20_XY, LENTO36_XY, LENTO50_XY, RAPIDO20_XY, RAPIDO36_XY, RAPIDO50_XY]
for i in range(len(matrices_XY)):
    heightMi = len(matrices_XY[i])
    #centramos las matrices
    matrices_XY[i][:,0] = matrices_XY[i][:,0] - (np.ones(heightMi, float))*centros[i][0]
    matrices_XY[i][:,1] = matrices_XY[i][:,1] - (np.ones(heightMi, float))*centros[i][1]
    #Por alguna razon vienen del PIV con los vectores de vel al revez.
    matrices_XY[i][:,2]= -1*matrices_XY[i][:,2]
    matrices_XY[i][:,3]= -1*matrices_XY[i][:,3]
    #pasamos a cm
    matrices_XY[i] = matrices_XY[i]*(-100)

#pasamos todas a polares
matrices_RT = []
for i in range(len(matrices_XY)):
    matrices_RT.append(converTo_polares(matrices_XY[i]))

#tomamos promedio espacial (sobre cada circunferencia)
matricesProm_Rp_Vtp = []
for i in range(len(matrices_RT)):
    matricesProm_Rp_Vtp.append(homogVtan_R(matrices_RT[i], int(len(matrices_RT[i])/25)))
    
#%%
#ajustes del set LENTOS (los promediados)
#los initual guess se sacaron del ajuste de los sin promediar que eran mas faciles ajustar debido a la dispersion
initialGuess_20Lento = [5.55807,1.11104,0.300924]
initialGuess_36Lento = [4.86007,1.14225,-0.0706614]
initialGuess_50Lento = [5.469,0.867945,0.00216094]

popt_20Lento, pcov_20Lento = curve_fit(func, matricesProm_Rp_Vtp[0][:,0], matricesProm_Rp_Vtp[0][:,1], initialGuess_20Lento)
popt_36Lento, pcov_36Lento = curve_fit(func, matricesProm_Rp_Vtp[1][:,0], matricesProm_Rp_Vtp[1][:,1], initialGuess_36Lento)
popt_50Lento, pcov_50Lento = curve_fit(func, matricesProm_Rp_Vtp[2][:,0], matricesProm_Rp_Vtp[2][:,1], initialGuess_50Lento)

#dominio para la curva de ajuste
rD_20Lento = np.linspace(min(matricesProm_Rp_Vtp[0][:,0]), max(matricesProm_Rp_Vtp[0][:,0]), 1000)
rD_36Lento = np.linspace(min(matricesProm_Rp_Vtp[1][:,0]), max(matricesProm_Rp_Vtp[1][:,0]), 1000)
rD_50Lento = np.linspace(min(matricesProm_Rp_Vtp[2][:,0]), max(matricesProm_Rp_Vtp[2][:,0]), 1000)

plt.figure()
plt.plot(matricesProm_Rp_Vtp[0][:,0],matricesProm_Rp_Vtp[0][:,1], '.', label = '20% concentracion', color ='blue')
plt.plot(rD_20Lento,func(rD_20Lento,*popt_20Lento), label='ajuste',color ='blue')
plt.plot(matricesProm_Rp_Vtp[1][:,0],matricesProm_Rp_Vtp[1][:,1], '.', label = '36% concentracion',color ='red')
plt.plot(rD_36Lento,func(rD_36Lento,*popt_36Lento), label='ajuste',color ='red')
plt.plot(matricesProm_Rp_Vtp[2][:,0],matricesProm_Rp_Vtp[2][:,1], '.', label = '50% concentracion',color ='green')
plt.plot(rD_50Lento,func(rD_50Lento,*popt_50Lento), label='ajuste',color ='green')
plt.xlabel("Radios [cm]", fontsize=15)
plt.ylabel("Velocidad tangencial [cm/s]", fontsize = 15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.rcParams['legend.fontsize'] = 14
plt.legend()

#%%
modelPredictions_20Lento = func(matricesProm_Rp_Vtp[0][:,0], *popt_20Lento) 
modelPredictions_36Lento = func(matricesProm_Rp_Vtp[1][:,0], *popt_36Lento)
modelPredictions_50Lento = func(matricesProm_Rp_Vtp[2][:,0], *popt_50Lento) 

absError_20Lento = modelPredictions_20Lento -  matricesProm_Rp_Vtp[0][:,1]
absError_36Lento = modelPredictions_36Lento -  matricesProm_Rp_Vtp[1][:,1]
absError_50Lento = modelPredictions_50Lento -  matricesProm_Rp_Vtp[2][:,1]

SE_20Lento = np.square(absError_20Lento) # squared errors
SE_36Lento = np.square(absError_36Lento) # squared errors
SE_50Lento = np.square(absError_50Lento) # squared errors


MSE_20Lento = np.mean(SE_20Lento) # mean squared errors
MSE_36Lento = np.mean(SE_36Lento) # mean squared errors
MSE_50Lento = np.mean(SE_50Lento) # mean squared errors

RMSE_20Lento = np.sqrt(MSE_20Lento) # Root Mean Squared Error, RMSE
RMSE_36Lento = np.sqrt(MSE_36Lento) # Root Mean Squared Error, RMSE
RMSE_50Lento = np.sqrt(MSE_50Lento) # Root Mean Squared Error, RMSE

Rsquared_20Lento = 1.0 - (np.var(absError_20Lento) / np.var(matricesProm_Rp_Vtp[0][:,1]))
Rsquared_36Lento = 1.0 - (np.var(absError_36Lento) / np.var(matricesProm_Rp_Vtp[1][:,1]))
Rsquared_50Lento = 1.0 - (np.var(absError_50Lento) / np.var(matricesProm_Rp_Vtp[2][:,1]))
print('RMSE_20Lento:', RMSE_20Lento)
print('RMSE_36Lento:', RMSE_36Lento)
print('RMSE_50Lento:', RMSE_50Lento)
print('R-squared-20Lento:', Rsquared_20Lento)
print('R-squared-36Lento:', Rsquared_36Lento)
print('R-squared-50Lento:', Rsquared_50Lento)


#%%
#ajustes del set RAPIDOS
#no quiero ajustar para radios menores que 1.10, por eso primero creo otro set de promediados
subMatrix20Rapido = matricesProm_Rp_Vtp[3][matricesProm_Rp_Vtp[3][:,0] >1.10]
subMatrix36Rapido = matricesProm_Rp_Vtp[4][matricesProm_Rp_Vtp[4][:,0] >1.10]
subMatrix50Rapido = matricesProm_Rp_Vtp[5][matricesProm_Rp_Vtp[5][:,0] >1.10]

initialGuess_20Rapido = [17.9488,1.69327,1.34539]
initialGuess_36Rapido = [21.7568,-2.31272,0.928999]
initialGuess_50Rapido = [46.6142,2.94499,0.069019]

popt_20Rapido, pcov_20Rapido = curve_fit(func, subMatrix20Rapido[:,0], subMatrix20Rapido[:,1], initialGuess_20Rapido)
popt_36Rapido, pcov_36Rapido = curve_fit(func, subMatrix36Rapido[:,0], subMatrix36Rapido[:,1], initialGuess_36Rapido)
popt_50Rapido, pcov_50Rapido = curve_fit(func, subMatrix50Rapido[:,0], subMatrix50Rapido[:,1], initialGuess_50Rapido)

#dominio para la curva de ajuste
rD_20Rapido = np.linspace(min(subMatrix20Rapido[:,0]), max(subMatrix20Rapido[:,0]), 1000)
rD_36Rapido= np.linspace(min(subMatrix36Rapido[:,0]), max(subMatrix36Rapido[:,0]), 1000)
rD_50Rapido= np.linspace(min(subMatrix50Rapido[:,0]), max(subMatrix50Rapido[:,0]), 1000)

plt.figure()
plt.plot(matricesProm_Rp_Vtp[3][:,0],matricesProm_Rp_Vtp[3][:,1], '.', label = '20% concentracion', color ='blue')
plt.plot(rD_20Rapido,func(rD_20Rapido,*popt_20Rapido), label='ajuste',color ='blue')
plt.plot(matricesProm_Rp_Vtp[4][:,0],matricesProm_Rp_Vtp[4][:,1], '.', label = '36% concentracion',color ='red')
plt.plot(rD_36Rapido,func(rD_36Rapido,*popt_36Rapido), label='ajuste',color ='red')
plt.plot(matricesProm_Rp_Vtp[5][:,0],matricesProm_Rp_Vtp[5][:,1], '.', label = '50% concentracion',color ='green')
plt.plot(rD_50Rapido,func(rD_50Rapido,*popt_50Rapido), label='ajuste',color ='green')
plt.xlabel("Radios [cm]", fontsize=15)
plt.ylabel("Velocidad tangencial [cm/s]", fontsize = 15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.rcParams['legend.fontsize'] = 10
plt.legend()
#%%
modelPredictions_20Rapido= func(subMatrix20Rapido[:,0], *popt_20Rapido) 
modelPredictions_36Rapido= func(subMatrix36Rapido[:,0], *popt_36Rapido)
modelPredictions_50Rapido= func(subMatrix50Rapido[:,0], *popt_50Rapido) 

absError_20Rapido= modelPredictions_20Rapido -  subMatrix20Rapido[:,1]
absError_36Rapido= modelPredictions_36Rapido -  subMatrix36Rapido[:,1]
absError_50Rapido= modelPredictions_50Rapido -  subMatrix50Rapido[:,1]

SE_20Rapido= np.square(absError_20Rapido) # squared errors
SE_36Rapido= np.square(absError_36Rapido) # squared errors
SE_50Rapido= np.square(absError_50Rapido) # squared errors


MSE_20Rapido= np.mean(SE_20Rapido) # mean squared errors
MSE_36Rapido= np.mean(SE_36Rapido) # mean squared errors
MSE_50Rapido= np.mean(SE_50Rapido) # mean squared errors

RMSE_20Rapido= np.sqrt(MSE_20Rapido) # Root Mean Squared Error, RMSE
RMSE_36Rapido= np.sqrt(MSE_36Rapido) # Root Mean Squared Error, RMSE
RMSE_50Rapido= np.sqrt(MSE_50Rapido) # Root Mean Squared Error, RMSE

Rsquared_20Rapido= 1.0 - (np.var(absError_20Rapido) / np.var(subMatrix20Rapido[:,1]))
Rsquared_36Rapido= 1.0 - (np.var(absError_36Rapido) / np.var(subMatrix36Rapido[:,1]))
Rsquared_50Rapido= 1.0 - (np.var(absError_50Rapido) / np.var(subMatrix50Rapido[:,1]))
print('RMSE_20Rapido:', RMSE_20Rapido)
print('RMSE_36Rapido:', RMSE_36Rapido)
print('RMSE_50Rapido:', RMSE_50Rapido)
print('R-squared-20Rapido:', Rsquared_20Rapido)
print('R-squared-36Rapido:', Rsquared_36Rapido)
print('R-squared-50Rapido:', Rsquared_50Rapido)