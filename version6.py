#!/usr/bin/python
# coding=utf-8
#此版本使用了共轭梯度法优化,并且加入了惩罚项

import numpy
import random
import matplotlib.pyplot as pt
import math
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
	return x, y, t, tmp
	
#函数功能：估计函数
def estimateFun(x,theta,order):
	sum=0.0
	for i in range(0,order+1):
		sum+=theta[i]*numpy.power(x,i)
	return sum
	
#函数功能：损失函数
def lossFun(X,Y,theta,order,lamb):
	sum=0.0
	for i in range(0,len(X)):
		x=X[i]
		y=Y[i]
		value=numpy.power(estimateFun(x, theta, order)-y,2)
		sum+=value
	sum+=lamb/2*calNorm(theta)
	return sum

#计算由当前数据样本所构成的黑塞矩阵
def Hes(x, order, lamb):
	h=[]
	for row in range(0, order+1):
		h_row = []
		for col in range(0, order+1):
			sum = 0
			for j in range(0, x.shape[0]):
				sum += numpy.power(x[j], row+col)
			h_row.append(sum)
		h.append(h_row)
	for i in range(0, order+1):
		h[i][i]+=lamb
	h=numpy.array(h)
	return h
	
#函数功能：计算当前的梯度
def calGradient(m, n, theta, order,lamb):
	gra = []
	for j in range(0, order+1):
		sum = 0
		for i in range(0, m.shape[0]):
			sum += numpy.power(m[i], j)*(estimateFun(m[i], theta, order)-n[i])
		gra.append(sum+lamb*theta[j])
	gra=numpy.array(gra)
	return gra
	
#函数功能：计算向量的二范数的平方
def calNorm(x):
	norm=numpy.dot(x.T,x)
	return norm

#函数功能：实现共轭梯度放方法
def conjugateGradient(t, s, weight, max_power, lamb):
	
	residue = -calGradient(t, s, weight, max_power, lamb)#计算最速下降方向
	direction=residue#定义初始搜索方向为初始点的最速下降方向

	gra0 = calNorm(residue)#表示该点的梯度的范数的平方
	beta0 = gra0
	
	iterUp = max_power+1#迭代上限，即每迭代iterUp次，搜索方向重新改为当前点的最速下降方向
	renew_theta = weight#定义迭代过程的权重向量

	i=0
	j=0
	iMax=10#定义theta迭代的上限
	jMax=5#定义线性迭代的上限
	e = 1e-12#定义一个无穷小量

	
	print "renew_theta init:", renew_theta
	count=0
	while i<iMax and gra0 > e*beta0:
		count+=1
		i += 1
		
		norm = calNorm(direction)
		alpha = (e+1)**2

		j=0
		while j<jMax and alpha**2*norm>e:#线性迭代，使得在该方向取得最小值
			#计算步长alpha
			alpha = -numpy.dot(calGradient(t, s, renew_theta, max_power,lamb).T,direction) / (numpy.dot(direction.T,numpy.dot(Hes(t, max_power,lamb),direction)))
#			print "alpha=", alpha
			renew_theta = renew_theta + alpha * direction#theta的迭代方程
			j += 1
		
		#计算用于迭代搜索方向的参数beta
		residue = -calGradient(t, s, renew_theta, max_power,lamb)#计算当前点的最速下降方向
		gra1 = gra0
		gra0 = calNorm(residue)
		beta = gra0/gra1#参数Beta计算公式 beta=||residue_new||^2 / ||residue_old||^2

		#根据前一次的搜索方向以及该点的最速下降方向，计算该点的搜索方向
		direction = residue + beta*direction

		#重新开始
		if count==iterUp or numpy.dot(residue.T,direction)<=0:
			direction=residue
			count = 0
		print '------'
		tmp=lossFun(X,Y,renew_theta,order,lamb)
		print 'J=',tmp,', count=',i
		
	return renew_theta
	

#main#############################################################################
fig = pt.figure()
ax = fig.add_subplot(111)

#定义函数阶数
order = 9
#定义数据量
num = 10
#定义惩罚项系数
lamb=100

#初始化一组权值向量theta
theta=[]#定义权值向量theta
for i in range(0,order+1):
	theta.append(0.0)#初始化所有参数均为0
theta = 200*numpy.random.random(size=order+1)-100#自动生成一组theta，范围介于-100到100
theta=numpy.array(theta)

X,Y,tmp,t = generateData(num)#生成数据

#利用共轭梯度法进行拟合
theta=conjugateGradient(X, Y, theta,order,lamb)
#画出拟合后的曲线
x_t= numpy.arange(0,6,0.01)
y_t=[]
for i in range(0,len(x_t)):
	value=estimateFun(x_t[i],theta,order)
	y_t.append(value)
	
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Conjugate gradient method(Add penalty term)')

point=ax.plot(X,Y,color='r',linestyle='',marker='.')
line1=ax.plot(x_t,y_t,color='green',linestyle='-',marker='')
#line2=ax.plot(x_t,temp,color='blue',linestyle='-',marker='')
line3=ax.plot(t,tmp,color='black',linestyle='-',marker='')

ax.axis([0, 1, -1.5, 1.5])
ax.legend( (line1[0], line3[0], point[0]), ('Fitted curve', 'Sin(x)', 'Data Sample') )
#展示图像
pt.show()