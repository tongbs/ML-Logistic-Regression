import random
import math
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

def gaussian_data_generator(mean, variance):
    # Box-Muller transform
    U = random.random()
    V = random.random()
    X = math.sqrt((-2)*math.log(U)) * (math.cos(2*math.pi*V))
    return math.sqrt(variance)*X + mean

def pred_y(data_X, weight):
    pred_y_val = 1/(1+np.exp(np.dot(-1*data_X,weight)))
    #print('pred_y',pred_y_val)
    for val in pred_y_val:
        if val[0] < 0.5:
            val[0] = 0
        else:
            val[0] = 1
    return pred_y_val

def draw_confusion_matrix(data_X, data_Y, weight):
    pred_y_val = pred_y(data_X, weight)
    tn, fp, fn, tp = confusion_matrix(data_Y, pred_y_val).ravel()
    sensitivity = tp/(tp+fn)
    specificity = tn/(fp+tn)

    conf_mat = pd.DataFrame([[tp,fn],[fp,tn]],index=['Is cluster 1','Is cluster 2'], columns=['Predict cluster 1','Predict cluster 2'])
    print(conf_mat)
    print('Sensitivity (Successfully predict cluster 1): ',sensitivity)
    print('Specificity (Successfully predict cluster 2): ',specificity)

def gradient_data(data_X, weight):
    pred_y_val = pred_y(data_X, weight)
    gt0_x, gt0_y, gt1_x, gt1_y = [],[],[],[]
    for num, val in enumerate(pred_y_val):
        if val[0] == 0:
            gt0_x.append(data_X[num][1])
            gt0_y.append(data_X[num][2])
        else:
            gt1_x.append(data_X[num][1])
            gt1_y.append(data_X[num][2])
    return gt0_x,gt0_y,gt1_x,gt1_y

def netwon_data(data_X, weight):
    pred_y_val = pred_y(data_X, weight)
    nt0_x, nt0_y, nt1_x, nt1_y = [],[],[],[]
    for num, val in enumerate(pred_y_val):
        if val[0] == 0:
            nt0_x.append(data_X[num][1])
            nt0_y.append(data_X[num][2])
        else:
            nt1_x.append(data_X[num][1])
            nt1_y.append(data_X[num][2])
    return nt0_x,nt0_y,nt1_x,nt1_y

def draw_result(D1,D2,gt0_x,gt0_y,gt1_x,gt1_y,nt0_x,nt0_y,nt1_x,nt1_y):
    window, axes = plt.subplots(1,3)
    axes[0].set_title('Ground Truth')
    axes[1].set_title('Gradient Descent')
    axes[2].set_title('Netwon'+'s method')
    
    axes[0].scatter([point[0] for point in D1], [point[1] for point in D1], c='r')
    axes[0].scatter([point[0] for point in D2], [point[1] for point in D2], c='b')

    axes[1].scatter(gt0_x,gt0_y,c='r')
    axes[1].scatter(gt1_x,gt1_y,c='b')

    axes[2].scatter(nt0_x,nt0_y,c='r')
    axes[2].scatter(nt1_x,nt1_y,c='b')
    plt.show()
    plt.close()

if  __name__ == "__main__":
    N = int(input('N = '))
    mx1 = float(input('input mx1 = '))
    my1 = float(input('input my1 = '))
    mx2 = float(input('input mx2 = '))
    my2 = float(input('input my2 = '))    
    vx1 = float(input('input vx1 = '))
    vy1 = float(input('input vy1 = '))
    vx2 = float(input('input vx2 = '))
    vy2 = float(input('input vy2 = '))
    D1,D2 = [],[]
    for i in range(N):   # D1: target 0
        x = gaussian_data_generator(mx1,vx1)
        y = gaussian_data_generator(my1,vy1)
        D1.append((x,y,0))
    print(D1)
    for i in range(N):   # D2: target 1
        x = gaussian_data_generator(mx2,vx2)
        y = gaussian_data_generator(my2,vy2)
        D2.append((x,y,1))
    Data = D1+D2
    data_X=np.array([ [1, data[0], data[1]] for data in Data]).reshape((2*N,3))
    data_Y=np.array([data[2] for data in Data]).reshape((2*N,1))
    #W = np.zeros((3,1))
    cnt = 0
    weight = np.array([0.0 for i in range(3)]).reshape((3,1))
    # Gradient Descent
    print('Gradient Descent')
    while True:
        cnt+=1
        trans_X = data_X.transpose()
        d_weight = np.dot(trans_X, (data_Y-(1/(1+np.exp(np.dot(-1*data_X,weight))))))
        weight += d_weight
        # converge when d_weight->0
        if abs(d_weight[0][0])<0.001 and abs(d_weight[1][0])<0.001 and abs(d_weight[2][0])<0.001:
            print(d_weight)
            break
        elif cnt >= 1200:  # average for about cnt=10 will coverge
            break
    
    print('w:',weight)
    # Confusion matrix
    draw_confusion_matrix(data_X, data_Y, weight)
    gt0_x, gt0_y, gt1_x, gt1_y = gradient_data(data_X,weight)

    print()
    print('--------------------------------------------')


    print('Newton'+'s method:')
    cnt = 0
    weight = np.array([0.0 for i in range(3)]).reshape((3,1))
    while True:
        cnt+=1
        trans_X = data_X.transpose()
        d_weight = np.dot(trans_X, (data_Y-(1/(1+np.exp(np.dot(-1*data_X,weight))))))
        D = np.zeros((2*N,2*N))
        # D diagonal is (e^-Xi*w)/((1+e^-Xiw)^2)
        for i in range(2*N):
            num = np.exp(np.dot(-1*data_X[i],weight))[0]
            D[i,i] = num/((1+num)**2)
        # Hession = A^T*D*A not always can inverse
        H = np.dot(np.dot(trans_X,D), data_X)
        # check H singular or not, singular matrix cannot inverse
        if np.linalg.matrix_rank(H) == H.shape[0]:
            H_inv = np.linalg.inv(H)
            d_weight = np.dot(H_inv,d_weight)
            weight+=d_weight
        else: # singular
            weight+=d_weight
        # converge when d_weight->0
        if abs(d_weight[0][0])<0.001 and abs(d_weight[1][0])<0.001 and abs(d_weight[2][0])<0.001:
            print(d_weight)
            break
        elif cnt >= 12:  # average for about cnt=10 will coverge
            break 
    print('w:',weight)
    # Confusion matrix
    draw_confusion_matrix(data_X, data_Y, weight)
    nt0_x, nt0_y, nt1_x, nt1_y = netwon_data(data_X,weight)
    draw_result(D1,D2,gt0_x,gt0_y,gt1_x,gt1_y,nt0_x,nt0_y,nt1_x,nt1_y)