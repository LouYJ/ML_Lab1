#!/usr/bin/python
# coding=utf-8
#此版本在最小二乘法的基础上加入了惩罚项优化

import numpy
import random
import matplotlib.pyplot as pt
import math

fig = pt.figure()
ax = fig.add_subplot(111)

# 在0-2*pi的区间上生成100个点作为输入数据
X = numpy.linspace(0,2*numpy.pi,100,endpoint=True)
Y = numpy.sin(X)

# 对输入数据加入gauss噪声
mu = 0# 定义gauss噪声的均值
sigma = 0.12# 定义gauss噪声的方差
for i in range(X.size):
	X[i] += random.gauss(mu,sigma)
	Y[i] += random.gauss(mu,sigma)
	
# 画出加入gauss噪声的所有数据点
ax.plot(X,Y,linestyle='',marker='.')
#pt.show()

# 定义函数阶数
order = 9

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
	
ax.plot(x_t,y_t,color='g',linestyle='-',marker='')

ax.legend()
#展示图像
pt.show()