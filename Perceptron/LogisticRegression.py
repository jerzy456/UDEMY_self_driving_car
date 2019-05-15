import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
n_pts=100
b=np.ones(n_pts)

rand_x2_val = np.random.normal(4, 2, n_pts)
rand_y2_val = np.random.normal(6, 2, n_pts)
top_region = np.array([np.random.normal(10, 2, n_pts),np.random.normal(12, 2, n_pts), b]).T
low_region = np.array([np.random.normal(5, 2, n_pts),np.random.normal(6, 2, n_pts), b]).T
all_points=np.vstack((top_region, low_region))

w1= -0.2
w2= -0.35
bias = 3.5
line_parameters = np.matrix([w1, w2, bias])
x1 = np.arrey([low_region[:, 0].min(), low_region[:, 0].max()])
#w1x1+w2x2+b=0
#x2= -b/w2 +x1*(-w1/w2)
x2= -b /w2 +x1 * (-w1/w2)


#plot stuff
plt.subplot(111)
plt.scatter(top_region[:,0], top_region[:,1], color='r')
plt.scatter(low_region[:,0], low_region[:,1],color='b')
plt.show()