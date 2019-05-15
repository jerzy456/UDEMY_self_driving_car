import numpy as np
import matplotlib.pyplot as plt


def draw(x1,x2):
    ln = plt.plot(x1,x2)

def sigmoid (score):
    return 1/(1+ np.exp(-score))

def calculate_error(line_parameters, points, y):
    p = sigmoid(linear_combination)
    m=points.shape[0]
    return (np.log(p).T*y +np.log(1-p).T*(1-y))*(1/m)


np.random.seed(0)
n_pts=10
bias=np.ones(n_pts)

rand_x2_val = np.random.normal(4, 2, n_pts)
rand_y2_val = np.random.normal(6, 2, n_pts)
top_region = np.array([np.random.normal(10, 2, n_pts),np.random.normal(12, 2, n_pts), bias]).T
low_region = np.array([np.random.normal(5, 2, n_pts),np.random.normal(6, 2, n_pts), bias]).T
all_points=np.vstack((top_region, low_region))

w1= -0.2
w2= -0.35
b = 3.5
line_parameters = np.matrix([w1,w2,b]).T
x1=np.array([low_region[:,0].min(), top_region[:,0].max()])
x2= -b/w2 + (x1*(-w1/w2))

linear_combination= all_points*line_parameters
y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts*2,1)
cross_error = calculate_error(line_parameters, all_points, y)
print(cross_error)

#plot stuff
plt.subplot(111)
plt.scatter(top_region[:,0], top_region[:,1], color='r')
plt.scatter(low_region[:,0], low_region[:,1],color='b')
draw(x1,x2)
plt.show()