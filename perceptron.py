#!/usr/bin/python3
# -*- coding: utf-8 -*- 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import perceptron
from sklearn.datasets import make_classification
import matplotlib.animation as animation
import sys
import os


       # OK
def activation(x,exp,weights,bias):
    X = np.array(x)
    W = np.array(weights)
    y = X.dot(W)+bias
    #print('valorFuncao '+str(y))
    value = 0 if y < 0 else 1
    #print('value '+str(value))
    return exp-value
    
    # OK
def refreshValues(weights,bias,error,x):
    x = np.multiply(x,error)
    weights=np.add(weights,x)
    bias = bias + error
    return weights,bias

def calcLine(limits,w,bias):
    ymin, ymax = limits[0],limits[1]
    a = -w[0] / w[1]
    xx = np.linspace(ymin, ymax)
    yy = a * xx - (bias) / w[1]
    return xx,yy
# Setando o grafico
fig = plt.figure()
ax = plt.axes()
# ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
line, = ax.plot([], [],linestyle=':',color='k')

def animate(i):

    line.set_data(i[0], i[1])
    return line,

def main():
    #generate data for classifier
    X,Y= make_classification(n_samples=int(sys.argv[1]),n_features=2,
    n_redundant=0, n_informative=1,n_clusters_per_class=1,
    shift=float(sys.argv[2]), scale=float(sys.argv[3]))
    # X=[[2,2],[-2,-2],[-2,2],[-1,1]]
    # Y=[0,1,0,1]
    # Preparing the values from plot
    x = [item[0] for item in X]
    y = [item[1] for item in X]
    ##setting graph props
    #defining limits plot
    ax.set_xlim(np.min(x)-2,np.max(x)+2)
    ax.set_ylim(np.min(y)-2,np.max(y)+2)
    # # End preparation
    colormap = np.array(['r', 'b'])
    plt.scatter(x,y,c=colormap[Y],s=10)
    weights=[float(sys.argv[4]), float(sys.argv[5])]
    bias=0
    value = 0
    vidas = 0
    train = 2
    error_all = list()
    trace = list()
    while (np.any(error_all) or (vidas <= 2)) :
        del error_all[:]
        for i,j in zip(X,Y):
            value = activation(i,j,weights,bias)
            if value != 0:
                weights,bias = refreshValues(weights,bias,value,i)
            error_all.append(value)
       
        trace.append(calcLine(ax.get_xlim(),weights,bias))
        vidas+=1
        if (vidas > len(x)*train):
            next = os.system('zenity --question --text="É necessário mais treinamento! Proceder?"')
            if next == 0 :
                train +=train
                print(vidas)
            else:
                os.system('zenity --error --text="Treinamento não foi suficiente, o problema pode ser não linearmente separável" --ellipsize')
                plt.show()
                exit()
    #getting velocidade da animação a ser exibida
    a = os.popen('zenity --scale --text "Escolha a velocidade para a exibição da busca da fronteira, em milissegundos" --min-value=100 --max-value=1000 --value=500 --step 1').readlines()
    anim = animation.FuncAnimation(fig, animate,
                               frames=trace, interval=int(str(a[0]).split()[0]),repeat=False ,blit=True)
    plt.show()

if __name__ == '__main__':
    if(len(sys.argv)!=6):
        exit("Use python3 perceptron.py <amostras> <escala> <stride> <peso1> <peso2>")
    main()