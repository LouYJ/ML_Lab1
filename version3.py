#!/usr/bin/python
# coding=utf-8
#此版本加入了梯度下降优化

import numpy
import random
import matplotlib.pyplot as pt
import math

fig = pt.figure()
ax = fig.add_subplot(111)

#定义函数阶数
order = 4
#定义步长a
a=0.01
#极小值e
e=1e-3
#定义参数向量theta
theta=[]
#定义梯度向量
delta=[]

for i in range(0,order+1):
	theta.append(0.0)#初始化所有参数均为0
	delta.append(0.0)#初始化梯度向量
#theta=[ -2.70734496e-01,3.77014150e+00,-8.36016933e+00,1.07926487e+01,-7.58712667e+00,  3.04762200e+00,  -7.27774747e-01,   1.02392057e-01,-7.83301069e-03,   2.51048228e-04]
theta = 200*numpy.random.random(size=order+1)-100#-100,100
#print theta
theta=numpy.array(theta)
delta=numpy.array(delta)


def addGaussian(num, s, sigma): # mu=0
	mu = 0
	offset_Guassian = numpy.random.normal(mu, sigma, num)
	s_res = s + offset_Guassian
	return s_res

def genData(num=20, xrange=1, yrange=1, sigma=0.1):
	interv = xrange*1.0/num
	t = numpy.arange(0, xrange, interv)
	s = yrange*numpy.sin(2*numpy.pi*t)
	s = addGaussian(s.shape[0], s, sigma)
	return t, s
	
def estimateFun(x,theta,order):
	sum=0.0
	for i in range(0,order+1):
		sum+=theta[i]*numpy.power(x,i)
	return sum
	
def lossFun(X,Y,theta,order):
	sum=0.0
	for i in range(0,len(X)):
		x=X[i]
		y=Y[i]
		value=numpy.power(estimateFun(x, theta, order)-y,2)
		sum+=value
	return sum
	
		
	
def updateTheta(theta,delta,X,Y,order,a):
#	print theta
	for i in range(0,order+1):
		sum=0.0
#		print 'in'
		for j in range(0,len(X)):
			value=numpy.power(X[j],i)*(estimateFun(X[j],theta,order)-Y[j])
#			print estimateFun(X[j],theta,order)-Y[j]
			sum+=value
		delta[i]=sum
		theta[i]=theta[i]-a*sum

X,Y = genData(100)
ax.plot(X,Y,color='r',linestyle='',marker='.')
count=0
while True:
	tmp1=lossFun(X,Y,theta, order)
#	print theta
#	print tmp1
	updateTheta(theta, delta, X, Y, order, a)
#	print theta
	tmp2=lossFun(X,Y,theta, order)
	count+=1
	print (tmp1-tmp2)
	if abs(tmp2-tmp1)<e:
		break

#画出拟合后的曲线
x_t= numpy.arange(0,6,0.01)
y_t=[]
for i in range(0,len(x_t)):
	value=estimateFun(x_t[i],theta,order)
	y_t.append(value)	
	
#print theta
ax.plot(x_t,y_t,color='g',linestyle='-',marker='')
ax.axis([0, 1, -2, 2])
ax.legend()
#展示图像
pt.show()