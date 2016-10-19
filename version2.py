#!/usr/bin/python
# coding=utf-8
#此版本在最小二乘法的基础上加入了惩罚项优化

import numpy
import random
import matplotlib.pyplot as pt
import math

fig = pt.figure()
ax = fig.add_subplot(111)

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

order = 9#定义函数阶数
num = 10#定义样本数量
lamda = 1#定义权重lamda

X,Y,t,tmp=generateData(num)

# 求解系数矩阵，设方程组为X·A=Y，此时X矩阵对角线上各元素都加上了lamda
matrix_X=[]#设为矩阵X
for i in range(0,order+1):
	row=[]#定义矩阵X的每一行
	for j in range(0,order+1):
		element=0.0#定义单个元素
		#计算X[i][j]
		for k in range(0,len(X)):
			mul=1.0
			for l in range(0,j+i):
				mul=mul*X[k]
			element+=mul
		row.append(element)
	matrix_X.append(row)
	
#这里是加惩罚项后的核心
for i in range(0,order+1):
	matrix_X[i][i]+=lamda
	
#print(len(X))
#print(matrix_X[0][0])
matrix_X=numpy.array(matrix_X)

matrix_Y=[]#设为矩阵Y
for i in range(0,order+1):
	row=0.0
	for k in range(0,len(X)):
		element=1.0
		for l in range(0,i):
			element=element*X[k]
		row+=Y[k]*element
	matrix_Y.append(row)
 
matrix_Y=numpy.array(matrix_Y)

#求解矩阵A
matrix_A=numpy.linalg.solve(matrix_X,matrix_Y)

#画出拟合后的曲线
#print(matrix_A)
x_t= numpy.arange(0,2*numpy.pi,0.01)
y_t=[]
for i in range(0,len(x_t)):
	value=0.0
	for j in range(0,order+1):
		dy=1.0
		for k in range(0,j):
			dy*=x_t[i]
		dy*=matrix_A[j]
		value+=dy
	y_t.append(value)
	
print matrix_A

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Least Squares Method(Add penalty term)')

point=ax.plot(X,Y,color='r',linestyle='',marker='.')
line1=ax.plot(x_t,y_t,color='green',linestyle='-',marker='')
#line2=ax.plot(x_t,temp,color='blue',linestyle='-',marker='')
line3=ax.plot(tmp,t,color='black',linestyle='-',marker='')

ax.axis([0, 1, -1.5, 1.5])
ax.legend( (line1[0], line3[0], point[0]), ('Fitted curve', 'Sin(x)', 'Data Sample') )
#展示图像
pt.show()