import gzip
import numpy as np
import random
def bytes_to_int(bytes):
    result = 0
    for b in bytes:
        result = result*256+int(b)
    return int(result)

def image_load(file):
    f = gzip.open(file,'rb')
    file_content = f.read()
    file_list = list(file_content)
    ##[0:4] magic number, [4:8]# of images, [8:12]# of rows, [12:16] # of cols
    num_img = bytes_to_int(file_list[4:8])
    num_row = bytes_to_int(file_list[8:12])
    num_col = bytes_to_int(file_list[12:16])
    img = np.array(file_list[16:]).reshape((num_img,num_row*num_col))
    return img

def label_load(file):
    f  = gzip.open(file,'rb')
    file_content = f.read()
    file_list = list(file_content)
    ##[0:4]magic number, [4:8]# of labels, [8:]one word one label
    num_label = bytes_to_int(file_list[4:8])
    label = np.array(file_list[8:]).reshape((num_label,1))
    return label

def invert_to_gray(tr_image):
    return tr_image//128

def E_step(class_pixel, class_prob, tr_data):
    # print(tr_data)
    W = np.zeros((1,10))
    for i in range(1):
        for j in range(10):
            tmp_class = class_prob[j]
            for k in range(784):
                tmp_class *= (class_pixel[j,k]**(tr_data[i,k]))*((1-class_pixel[j,k])**(1-tr_data[i,k]))
            W[i,j] = tmp_class
        total = np.sum(W[i])
        if total != 0:
            W[i]/=total
        else:
            W[i] = [0.1 for m in range(10)]
    print(W)
    return W

def M_step(W, tr_data):
    

if __name__ == "__main__":
    tr_image = image_load("train-images-idx3-ubyte.gz")
    tr_label = label_load("train-labels-idx1-ubyte.gz")
    # convert to gray level
    tr_data = invert_to_gray(tr_image)
    # print(tr_data.shape)
    class_pixel = np.array([[random.random() for j in range(784)]for i in range(10)]).reshape(10,784)
    class_prob = [0.1 for i in range(10)]
    cnt = 0
    
    while True:
        cnt+=1
        tmp_class_pixel = class_pixel
        tmp_class_prob = class_prob
        W = E_step(class_pixel, class_prob, tr_data)
        class_pixel, class_prob = M_step(W, tr_data)
        if cnt == 2:
            break