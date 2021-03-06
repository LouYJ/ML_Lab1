#!/usr/bin/python
# coding=utf-8
#此版本使用了梯度下降法进行曲线拟合，加入惩罚项进行优化

import numpy
import random
import matplotlib.pyplot as pt
import math

fig = pt.figure()
ax = fig.add_subplot(111)


order = 9#定义函数阶数
a=0.01#定义步长a
e=1e-3#极小值e
lamb=100#惩罚项参数
num=100#定义样本数量
theta=[]#定义权值向量theta
delta=[]#定义梯度向量


for i in range(0,order+1):
	theta.append(0.0)#初始化所有参数均为0
	delta.append(0.0)#初始化梯度向量

theta = 200*numpy.random.random(size=order+1)-100#自动生成一组theta，范围介于-100到100
theta=numpy.array(theta)
delta=numpy.array(delta)

#函数功能：加入高斯噪声
def addGaussian(num, s, sigma):
	mu = 0
	for i in range(0,len(s)):
		offset_Guassian = random.gauss(mu, sigma)
		s[i] = s[i] + offset_Guassian
	return s

#函数功能：用于生成样本数据
def generateData(num, xrange=1, yrange=1, sigma=0.12):
	interv = xrange*1.0/num
	tmp=numpy.arange(0, xrange, xrange*1.0/100)
	x = numpy.arange(0, xrange, interv)
	t = yrange*numpy.sin(2*numpy.pi*tmp)
	y = yrange*numpy.sin(2*numpy.pi*x)
	y = addGaussian(y.shape[0], y, sigma)
	return x, y, t,tmp
	
#函数功能：估计函数
def estimateFun(x,theta,order):
	sum=0.0
	for i in range(0,order+1):
		sum+=theta[i]*numpy.power(x,i)
	return sum
	
#函数功能：损失函数
def lossFun(X,Y,theta,order):
	sum=0.0
	for i in range(0,len(X)):
		x=X[i]
		y=Y[i]
		value=numpy.power(estimateFun(x, theta, order)-y,2)
		sum+=value
	return sum
	
#函数功能：利用最速下降法更新权值参数theta
def updateTheta(theta,delta,X,Y,order,a):
	for i in range(0,order+1):
		sum=0.0
		for j in range(0,len(X)):
			value=numpy.power(X[j],i)*(estimateFun(X[j],theta,order)-Y[j])
#			print estimateFun(X[j],theta,order)-Y[j]
			sum+=value
		delta[i]=sum+lamb*theta[i]
		theta[i]=theta[i]-a*sum

#mian
X,Y,tmp,t = generateData(num)
count=0
while True:
	tmp1=lossFun(X,Y,theta, order)
#	print theta
#	print tmp1
	updateTheta(theta, delta, X, Y, order, a)
#	print theta
	tmp2=lossFun(X,Y,theta, order)
	count+=1
	print 'J=',tmp2,', count=',count,', delta=',abs(tmp2-tmp1)
	if abs(tmp2-tmp1)<e:
		break
	if count>100000:
		print '迭代次数过多，请修改参数'
		break

#画出拟合后的曲线
x_t= numpy.arange(0,6,0.01)
y_t=[]
for i in range(0,len(x_t)):
	value=estimateFun(x_t[i],theta,order)
	y_t.append(value)	
	
#print theta
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Gradient Descent Method(Add penalty term)')

point=ax.plot(X,Y,color='r',linestyle='',marker='.')
line1=ax.plot(x_t,y_t,color='green',linestyle='-',marker='')
#line2=ax.plot(x_t,temp,color='blue',linestyle='-',marker='')
line3=ax.plot(t,tmp,color='black',linestyle='-',marker='')

ax.axis([0, 1, -1.5, 1.5])
ax.legend( (line1[0], line3[0], point[0]), ('Fitted curve', 'Sin(x)', 'Data Sample') )
#展示图像
pt.show()