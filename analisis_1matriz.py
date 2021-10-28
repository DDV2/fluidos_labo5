# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
from scipy.io import loadmat
from scipy.optimize import curve_fit


matrizXY = np.loadtxt('Matrices/50%RAPIDO.txt',delimiter= ',', skiprows= 3, usecols = (0,1,2,3))
heightM = len(matrizXY)
matrizXY = matrizXY*100
#aveces la matriz aparece con velocidad al revez, lo ajustamos siempre para que quede con velocidad angular positiva
matrizXY[:,2] = -1*matrizXY[:,2]
matrizXY[:,3] = -1*matrizXY[:,3]
plt.quiver(matrizXY[:,0], matrizXY[:,1], matrizXY[:,2], matrizXY[:,3], color='b')
plt.xlabel('Distancia en x al centro [cm]', fontsize=15)
plt.ylabel('Distancia en y al centro [cm]', fontsize=15)
plt.grid() 
plt.show()  

#Para hacerlo mas agil despues, definimos una funcion ahora
#Funcion que transforma las matrices de pos y vel en cartesianas a polares
#En la col0 esta el Radio, en la col1 esta el Angulo, en la col2 esta la Vel Rad, en la col3 esta la Vel Tan
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
    plt.xlabel("X [cm]", fontsize=15)
    plt.ylabel("Suma de |V| de la columna en x [cm/s]", fontsize = 15)
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
    plt.xlabel("Y [cm]", fontsize=15)
    plt.ylabel("Suma de |V| de la fila en y [cm/s]", fontsize = 15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
#esta funcion homogeneiza los graficos que tienen mucha dispersion
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
def func(x,a,b,c,d):
    return ((a/(x-d))*(1 - np.exp(-((x-d)**2/b**2))))+c

#%%
#en esta celda se centra la matriz usando las dos funciones auxiliares definidas en la celda anterior
#se abren dos graficos y hay que seleccionar a ojo la posicion del minilo local
centrarX_aux(matrizXY)
x0 = plt.ginput(1)[0][0]
plt.close()
centrarY_aux(matrizXY)
y0 = plt.ginput(1)[0][0]
plt.close()

#centrado de la matriz con la posicion seleccionada
matrizXY_centrada = matrizXY.copy()
matrizXY_centrada[:,0]= matrizXY_centrada[:,0] - (np.ones(heightM, float))*x0
matrizXY_centrada[:,1]= matrizXY_centrada[:,1] - (np.ones(heightM, float))*y0

#pasamos ambas a polares
matrizRT = converTo_polares(matrizXY)
matrizRT_centrada = converTo_polares(matrizXY_centrada)
matrizProm_Rp_Vtp = homogVtan_R(matrizRT_centrada, 100)

#%%
#graficamos ambas al mismo tiempo para ver que tan bien hicimos el centrado
plt.plot(matrizRT[:,0], matrizRT[:,3], '.', label='sin centrar')
plt.plot(matrizRT_centrada[:,0], matrizRT_centrada[:,3], '.', label='centrada')
plt.plot(matrizProm_Rp_Vtp[:,0],matrizProm_Rp_Vtp[:,1], '.', label='centrado promediado angular')
plt.xlabel("Radios [cm]", fontsize=15)
plt.ylabel("Velocidad tangencial [cm/s]", fontsize = 15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend()
#si la dispersion disminuyo entonces seguimos con la centrada, sino volver a intentar la celda anterior

#%%

#Grafico de Velocidad Tangencial en funcion del radio
#elijo una submatriz del promediado para hacer el ajuste mas limpio ( no quiero los radios menores que 1)
initialGuess = [0,(0.002)*100,(0.02)*100,(0.01)*100] 
popt, pcov = curve_fit(func, matrizRT_centrada[:,0], matrizRT_centrada[:,3], initialGuess)
rD = np.linspace(min(matrizRT_centrada[:,0]),max(matrizRT_centrada[:,0]),len(matrizRT_centrada[:,0]))

plt.figure()
plt.plot(matrizRT_centrada[:,0], matrizRT_centrada[:,3], '.',label='centrado')
plt.plot(rD,func(rD,*popt), label='ajuste')
plt.xlabel("Radios [cm]", fontsize=15)
plt.ylabel("Velocidad tangencial [cm/s]", fontsize = 15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend()

#%%
#Grafico de Vel. Tangencial en funcion del radio PERO PARA EL PROMEDIADO ESPACIAL
#dado que el set rapido tiene problemas al ajustar para radios menores a 1, restringimos el espacio de datos
#(para el set lento setear rmin = 0)
rmin= 1
subMatrix = matrizProm_Rp_Vtp[matrizProm_Rp_Vtp[:,0] >rmin]

initialGuess_prom = popt
popt_prom, pcov_prom = curve_fit(func, subMatrix[:,0], subMatrix[:,1], initialGuess)
rD_prom = np.linspace(min(subMatrix[:,0]),max(subMatrix[:,0]),len(subMatrix[:,0]))

plt.figure()
plt.title('V_tangencial(r)')
plt.plot(matrizProm_Rp_Vtp[:,0], matrizProm_Rp_Vtp[:,1], '.',label='datos')
plt.plot(rD_prom,func(rD_prom,*popt_prom), label='ajuste')
plt.xlabel("Radios [cm]", fontsize=15)
plt.ylabel("Velocidad tangencial [cm/s]", fontsize = 15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend()

modelPredictions = func(subMatrix[:,0], *popt) 

absError = modelPredictions -  subMatrix[:,1]

SE = np.square(absError) # squared errors
MSE = np.mean(SE) # mean squared errors
RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
Rsquared = 1.0 - (np.var(absError) / np.var(subMatrix[:,1]))
print('RMSE:', RMSE)
print('R-squared:', Rsquared)



#%%
#Grafico de Velocidad Radial en funcion del radio
#Para chequear alineacion del centro escogido, velocidades tangenciales en funcion del angulo
titas = []
vtangs = []
for i in range(len(matrizRT)):
    r = matrizRT[i][0]
    #elijo el radio en un intervalo
    if  0.057 < r < 0.063:
        titas.append(matrizRT[i][1])
        vtangs.append(matrizRT[i][3])

plt.figure()
plt.title('V_tangencial(tita)')
plt.plot(titas,vtangs,'.', label="0.057 < r < 0.063")
plt.xlabel("Angulo [rad]", fontsize=15)
plt.ylabel("Velocidad tangencial [cm/s]", fontsize = 15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend()


#%%
#Grafico de velocidad radial en funcion del radio
radios = matrizRT[:,0]
velRad = matrizRT[:,2]

plt.figure()
plt.title('V_r(r)')
plt.plot(radios, velRad, '.')
plt.xlabel("Radios [cm]", fontsize=15)
plt.ylabel("Velocidad radial[cm/s]", fontsize = 15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


#%%
#Metodo 2 para encontrar el centro adecuado (el metodo no es tan efectivo como pensabamos)
#Se intenta encontrar el centro ajustando un circulo de vectores que tienen una velocidad total arriba de un umbral
totMtx = matrizXY.copy()
circMtx = []
for i in range(heightM):
    vTot = np.sqrt(totMtx[i][2]**2 + totMtx[i][3]**2)
    if vTot > 0.032:
        circMtx.append(totMtx[i])
circMatrix = np.array(circMtx)
plt.quiver(circMatrix[:,0], circMatrix[:,1], circMatrix[:,2], circMatrix[:,3], color='b')
plt.xlabel('Distancia en x al centro [m]', fontsize=15)
plt.ylabel('Distancia en y al centro [m]', fontsize=15)
plt.grid() 
plt.show()  
#%%
#Metodo 1 para encontrar el centro adecuado
#Se intenta minimizar el ancho de dispersion para un radio particular, el problema es que ensancha la disp. en otras zonas
xCors = np.linspace(-.015,.015,10)
yCors = np.linspace(-.015,.015,10)
elecciones = np.empty((len(xCors)*len(yCors), 3), float)
for k in range(len(yCors)):    
    for i in range(len(xCors)):
        mXYPrueba = matrizXY.copy()
        mXYPrueba[:,0] = mXYPrueba[:,0] - (np.ones(heightM, float))*xCors[i]
        mXYPrueba[:,1] = mXYPrueba[:,1] - (np.ones(heightM, float))*yCors[k]
        mRTPrueba = converTo_polares(mXYPrueba)    
        vtangsPrueba = []
        for j in range(heightM):
            r = mRTPrueba[j][0]
            if  0.057 < r < 0.063:
                vtangsPrueba.append(mRTPrueba[j][3])
        anchoError = max(vtangsPrueba) - min(vtangsPrueba)
        elecciones[i+k*len(xCors)][0]= anchoError
        elecciones[i+k*len(xCors)][1]= xCors[i]
        elecciones[i+k*len(xCors)][2]= yCors[k]
    print(k)
minimo = []
for i in range(len(elecciones)):
    if(elecciones[i][0] == min(elecciones[:,0])):
        print(i)
        minimo.append((elecciones[i][0], elecciones[i][1], elecciones[i][2]))
theMatrix = matrizXY.copy()
theMatrix[:,0] = theMatrix[:,0] - (np.ones(heightM, float))*minimo[0][1]
theMatrix[:,1] = theMatrix[:,1] - (np.ones(heightM, float))*minimo[0][2]
theMatrixRT = converTo_polares(theMatrix)
plt.figure()
plt.title('Comparacion de V_tangencial(r)')
plt.plot(matrizRT[:,0], matrizRT[:,3], '.', label='normal')
plt.plot(theMatrixRT[:,0], theMatrixRT[:,3], '.', label='Optimizado')
plt.xlabel("Radios [m]", fontsize=15)
plt.ylabel("Velocidad tangencial [m/s]", fontsize = 15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend()
titasM = []
vtangsM = []
for i in range(len(theMatrixRT)):
    r = theMatrixRT[i][0]
    #elijo el radio en un intervalo
    if  0.057 < r < 0.063:
        titasM.append(theMatrixRT[i][1])
        vtangsM.append(theMatrixRT[i][3])
plt.figure()
plt.title('Comparacion de V_tita(tita)')
plt.plot(titas_R,velsTan_R, '.', label="0.057 < r < 0.063")
plt.plot(titasM,vtangsM, '.', label="0.057 < r < 0.063, Optimizado")
plt.xlabel("Angulo [rad]", fontsize=15)
plt.ylabel("Velocidad tangencial [m/s]", fontsize = 15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend()