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

def gradient_decent(line_parameters, points, y, alpha):
    m = points.shape[0]
    alpha = 0.01
    for i in range(500):
        p = sigmoid(points*line_parameters)
        grad = (points.T * (p-y))*(alpha/m)
        line_parameters = line_parameters - grad
        w1= line_parameters.item(0)
        w2 = line_parameters.item(1)
        b = line_parameters.item(2)

        x1 = np.array([points[:,0].min(),points[:,0].max()])
        x2 = - b/w2 +x1 *( - w1 /w2)

np.random.seed(0)
n_pts=10
bias=np.ones(n_pts)

rand_x2_val = np.random.normal(4, 2, n_pts)
rand_y2_val = np.random.normal(6, 2, n_pts)
top_region = np.array([np.random.normal(10, 2, n_pts),np.random.normal(12, 2, n_pts), bias]).T
low_region = np.array([np.random.normal(5, 2, n_pts),np.random.normal(6, 2, n_pts), bias]).T
all_points=np.vstack((top_region, low_region))


line_parameters = np.matrix([np.zeros(3)]).T
gradient_decent(line_parameters, )

y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts*2,1)
cross_error = calculate_error(line_parameters, all_points, y)
print(cross_error)

#plot stuff
plt.subplot(111)
plt.scatter(top_region[:,0], top_region[:,1], color='r')
plt.scatter(low_region[:,0], low_region[:,1],color='b')

plt.show()