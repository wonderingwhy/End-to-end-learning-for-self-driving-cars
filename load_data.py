import scipy.misc
import random
import cv2

xs = []
ys = []

#points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0

#read data.txt
with open("E:/Download/ZJUCloud/Intro2AI_Assignments/Intro2AI_Assignments/Assignment1/driving_dataset/driving_dataset/data.txt") as f:
    for line in f:
        xs.append("E:/Download/ZJUCloud/Intro2AI_Assignments/Intro2AI_Assignments/Assignment1/driving_dataset/driving_dataset/" + line.split()[0])
        #the paper by Nvidia uses the inverse of the turning radius,
        #but steering wheel angle is proportional to the inverse of turning radius
        #so the steering wheel angle in radians is used as the output
        ys.append(float(line.split()[1]) * scipy.pi / 180)

#get number of images
num_images = len(xs)

#shuffle list of images
c = list(zip(xs, ys))
#random.shuffle(c)
xs, ys = zip(*c)

train_xs = xs[:int(len(xs) * 0.8)]
train_ys = ys[:int(len(xs) * 0.8)]

val_xs = xs[-int(len(xs) * 0.2):]
val_ys = ys[-int(len(xs) * 0.2):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)

def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        # x_out.append(scipy.misc.imresize(scipy.misc.imread(train_xs[(train_batch_pointer + i) % num_train_images])[-150:], [66, 200]) / 255.0)
        
        image = scipy.misc.imread(train_xs[(train_batch_pointer + i) % num_train_images])
        # Image Nomalization 直方图均衡化
        #for i in range(0, 3):
            #image[:,:,i] = cv2.equalizeHist(image[:,:,i])
        # 取图像下半部分（路面部分）并缩小scale
        image = scipy.misc.imresize(image[-150:], [66, 200])
        
        # RGB->YUV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        
        # 将pixel数值scale到到0-1之间
        x_out.append(image / 255.0)
        
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])
    train_batch_pointer += batch_size
    return x_out, y_out

def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        # x_out.append(scipy.misc.imresize(scipy.misc.imread(val_xs[(val_batch_pointer + i) % num_val_images])[-150:], [66, 200]) / 255.0)
        
        image = scipy.misc.imread(val_xs[(val_batch_pointer + i) % num_val_images])
        # Image Nomalization 直方图均衡化
        #for i in range(0, 3):
            #image[:,:,i] = cv2.equalizeHist(image[:,:,i])
        
        # 取图像下半部分（路面部分）并缩小scale
        image = scipy.misc.imresize(image[-150:], [66, 200])
        
        # RGB->YUV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        
        # 将pixel数值scale到到0-1之间
        x_out.append(image / 255.0)
        
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    return x_out, y_out
