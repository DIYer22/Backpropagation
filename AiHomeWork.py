# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 00:36:24 2016

@author: yl
"""
from __future__ import unicode_literals
import numpy as np
from aiTools import *
import os
import scipy as sp
from random import shuffle 

array= lambda x :np.array(range(x)) if isinstance(x,int) else np.array(x)
repeatv = lambda row, m: np.array([row for _ in range(m)])
def plotFun(fun):
    x = np.linspace(0,1,100)
    plt.plot(x,fun(x))
#    plt.show()
#%%
def drawPridectLine(*l):
    xs = np.linspace(0,1,100)
    xs = xs.reshape((len(xs),1))
    ys = net.forwards(xs)
    plt.plot(xs[...,0],ys[...,0]);plt.show()

x = np.array([1.,-1.])
y = [1.,1]

netshape = [2,5,2]
ws = None
rate = 0.5
bs = None

x = np.array([.05,.10])
y = [.01,.99]

netshape = [2,2,2]
ws = [
      np.array([0]),
      np.array([
                [0.15,0.20],
                [0.25,0.30],
                ]),
      np.array([
                [0.40,0.45],
                [0.50,0.55],
                ]),
                ]
rate = 0.1
bs = np.array([0,.35,.60])

class Net:
    
    def __init__(self, shape,rate,ws=None,bs=None):
        netshape = shape
        self.shape = shape
        self.rate = rate
        self.fun = lambda x:1./(1+np.e**(-x))
        if bs is None:
            bs = [0 for n in netshape]
        if ws is None:
            ws = [0]*len(netshape)
            for i,n in enumerate(netshape[1:]):
                i += 1
#                ws[i] = np.zeros((n,netshape[i-1]))
#                ws[i] = np.ones((n,netshape[i-1]))
                ws[i] = np.random.random((n,netshape[i-1]))-0.5
        self.ws = ws
        self.bs = bs
        
        self.nets = [np.zeros((n,)) for n in netshape]
        self.outs = [np.zeros((n,)) for n in netshape]
        self.fais = [0]*len(netshape)
    def forward(self, inp):
        netshape,rate,ws,bs,nets,outs,fais = self.shape,self.rate,self.ws,self.bs,self.nets,self.outs,self.fais
        outs[0] = np.array(inp)
        for i in range(1,len(netshape)):
            tmp = np.array([outs[i-1] for _ in range(netshape[i])])
#            print tmp.shape,ws[i].shape,bs[i]
            nets[i] = (tmp*ws[i]).sum(1)+bs[i]
            outs[i] = self.fun(nets[i])
        return outs[-1]
    def forwards(self, xs):
        ys = map(self.forward,xs)
        return np.array(ys)
    def backward(self,y):
        netshape,rate,ws,bs,nets,outs,fais = self.shape,self.rate,self.ws,self.bs,self.nets,self.outs,self.fais
        
        out = outs[-1]
        t = np.array(y)
        fais[-1] = ((out-t)*out*(1-out))
        for i in range(1,len(netshape)-1)[::-1]:
            tmp = repeatv(fais[i+1],netshape[i])*ws[i+1].T
            fais[i] = tmp.sum(1)*outs[i]*(1-outs[i])
        
        for i in range(1,len(netshape)):
            tmp = repeatv(fais[i],netshape[i-1]).T * repeatv(outs[i-1],netshape[i])
#            print ws[i],rate,tmp,i
            ws[i] -= rate*tmp
        etotal = (1/2.*((y - out)**2)).sum()
        return etotal
    def trainone(self, x, y):
        out = self.forward(x)
        etotal = self.backward(y) 
        return out, etotal
    def train(self, xs, ys, batch, callback):
#        tmp = np.array(zip(xs,ys))
#        np.random.shuffle(tmp)
#        xs,ys = tmp[:,0],tmp[:,1]
#        print tmp,xs[:2],ys[:2],tmp.shape
#        print '\n'*3,tmp,xs[:2],ys[:2]
        es = []
        for i,x,y in zip(range(len(xs)),xs, ys):
            out, etotal = self.trainone(x,y)
            es += [etotal]
            if(i==0):
                callback(es,batch)
            if (i+1)%batch == 0:
                print '\n batch %dth \n' % ((i+1)//batch)
                callback(es[-batch:],batch)
        if (i+1)%batch:
            callback(es[-((i+1)%batch):],batch)
net = Net(netshape,rate,ws,bs)
#print net.trainone(x,y)
#%%
kind = 1


def getcb(fun):
    l = []
    def avgLoss(es,batch):
        es = np.array(es)
        l.append(es.mean())
        plt.plot([i*batch for i in range(len(l))],l)
        plt.show()
        plotFun(fun)
#        plt.show()
        drawPridectLine()
    return avgLoss
# 当kind 为1时候 拟合 `y=sin(x * pi)`
if kind==1:
    shape = [1,100,50,1]
    shape = [1,10,1]
    rate = 1
    net = Net(shape,rate)
    
    fun = lambda x:np.sin(x*np.pi)
    #fun = lambda x:x*0.5+0.2
    
    trainnum = 100000
    #xs = np.linspace(0,1,trainnum)
    #np.random.shuffle(xs)
    xs = np.random.random(trainnum)
    ys = fun(xs)
    #plt.plot(xs,ys);plt.show()
    
    xs = xs.reshape((len(xs),1))
    ys = ys.reshape((len(ys),1))
    batch = len(xs)/30
    net.train(xs, ys, batch, getcb(fun))
    #net.train(xs, ys, batch, getcb(fun))
    #net.train(xs, ys, batch, getcb(fun))
    
    #plt.plot(xs,ys);plt.show()

if kind == 2:
    shape = [1,100,50,1]
    shape = [1,50,1]
    rate = 0.5
    net = Net(shape,rate)
    
    fun = lambda x:(x**2)
    #fun = lambda x:x*0.5+0.2
    
    trainnum = 10000
    #xs = np.linspace(0,1,trainnum)
    #np.random.shuffle(xs)
    xs = np.random.random(trainnum)
    ys = fun(xs)
    #plt.plot(xs,ys);plt.show()
    
    xs = xs.reshape((len(xs),1))
    ys = ys.reshape((len(ys),1))
    batch = len(xs)/30
    net.train(xs, ys, batch, getcb(fun))
    
if kind == 3:
    shape = [2,50,1]
    rate = 1
    ws = None
    bs = None
    net = Net(shape,rate,ws,bs)
    
    fun = lambda x1,x2:x1+x2
    
    trainnum = 1000
    x1 = np.linspace(0,1,trainnum)
    x2 = np.linspace(0,1,trainnum)
    x1 = np.random.random(trainnum)
    x2 = np.random.random(trainnum)
    x1,x2 = np.meshgrid(x1,x2)
    y = fun(x1,x2)
#    draw3dSurface(x1,x2,y)
    x1s,x2s,ys = map(lambda x:x.flatten(),[x1,x2,y])
    xs = zip(x1s,x2s)
    
    
    
    
    batch = len(x1s)//200
    def getAvgLoss():
        g = {'l':[]}
        def avgLoss(es,batch):
            l = g['l']
            npes = np.array(es)
            l+=[np.mean(es)]
            plt.plot(l)
            plt.show()
        return avgLoss
        
    callback = getAvgLoss()
    ys = ys.reshape((len(ys),1))
    net.train(xs, ys, batch, callback)

# MINST
if kind == 4:
    shape = [784,784*2,10]
    rate = 1
    net = Net(shape,rate,ws=_ws)
    
    xs = imgs[:,0]
    ys = []
    for row in labels:
        l = [0]*10
        l[row[0]]=1
        ys += [l]
    ys = np.array(ys)
    def getAvgLoss():
        g = {'l':[]}
        def avgLoss(es,batch):
            l = g['l']
            npes = np.array(es)
            l+=[np.mean(es)]
            plt.plot(l)
            plt.show()
        return avgLoss
        
    callback = getAvgLoss()
    batch = len(xs)//200
    net.train(xs,ys,batch,callback)
