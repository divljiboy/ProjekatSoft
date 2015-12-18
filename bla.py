
# coding: utf-8

# In[23]:

import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
from PIL import Image


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
def display_image(image, color= False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_bin(image_gs):
    ret,image_bin = cv2.threshold(image_gs, 100, 255, cv2.THRESH_BINARY)
    return image_bin
def image_bin_adaptive(image_gs):
    image_bin = cv2.adaptiveThreshold(image_gs, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 10)
    return image_bin
def dilate(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)
def erode(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)
def remove_noise(binary_image):
    ret_val = erode(dilate(binary_image))
    ret_val = invert(ret_val)
    return ret_val
def invert(image):
    return 255-image


# In[24]:

def draw_hand_rect(self, frame):  
    rows,cols,_ = frame.shape

    self.hand_row_nw = np.array([6*rows/20,6*rows/20,6*rows/20,10*rows/20,10*rows/20,10*rows/20,14*rows/20,14*rows/20,14*rows/20])

    self.hand_col_nw = np.array([9*cols/20,10*cols/20,11*cols/20,9*cols/20,10*cols/20,11*cols/20,9*cols/20,10*cols/20,11*cols/20])

    self.hand_row_se = self.hand_row_nw + 10
    self.hand_col_se = self.hand_col_nw + 10

    size = self.hand_row_nw.size
    for i in xrange(size):
        cv2.rectangle(frame,(self.hand_col_nw[i],self.hand_row_nw[i]),(self.hand_col_se[i],self.hand_row_se[i]),(0,255,0),1)
        
        

    return frame


# In[ ]:




# In[25]:

def getColorSquare(self, frame):
    size = self.hand_row_nw.size
    #for i in range(0,size):
    box =frame[self.hand_col_nw[0]:self.hand_col_se[0], self.hand_row_nw[0]:self.hand_row_se[0]]
    box1 =frame[self.hand_col_nw[1]:self.hand_col_se[1], self.hand_row_nw[1]:self.hand_row_se[1]]
    box2 =frame[self.hand_col_nw[2]:self.hand_col_se[2], self.hand_row_nw[2]:self.hand_row_se[2]]
    box3 =frame[self.hand_col_nw[3]:self.hand_col_se[3], self.hand_row_nw[3]:self.hand_row_se[3]]
    box4 =frame[self.hand_col_nw[4]:self.hand_col_se[4], self.hand_row_nw[4]:self.hand_row_se[4]]
    box5 =frame[self.hand_col_nw[5]:self.hand_col_se[5], self.hand_row_nw[5]:self.hand_row_se[5]]
    box6 =frame[self.hand_col_nw[6]:self.hand_col_se[6], self.hand_row_nw[6]:self.hand_row_se[6]]
    box7 =frame[self.hand_col_nw[7]:self.hand_col_se[7], self.hand_row_nw[7]:self.hand_row_se[7]]
    box8 =frame[self.hand_col_nw[8]:self.hand_col_se[8], self.hand_row_nw[8]:self.hand_row_se[8]]    
  
    return box,box1,box2,box3,box4,box5,box6,box7,box8


# In[26]:

def minmaxvalue(box):
    
    box1=np.array(box)
    minr=box1[..., 0].min()
    maxr=box1[..., 0].max()
    # green
    ming=box1[..., 1].min()
    maxg=box1[..., 1].max()
    # blue
    minb=box1[..., 2].min()
    maxb=box1[..., 2].max()
    print minr,ming,minb,maxr,maxg,maxb
    return  minr,ming,minb,maxr,maxg,maxb


# In[ ]:




# In[ ]:




# In[ ]:




# In[28]:

cap = cv2.VideoCapture(1)

class self(object):
    hand_row_nw = np.array((1,6))
    hand_col_nw = np.array((1,6))
    hand_row_se = np.array((1,6))
    hand_col_se = np.array((1,6))

lista = [None] * 9
kraj=False
while(1):
    
    ret, frame = cap.read()
    if kraj==False:
        frame=draw_hand_rect(self,frame)
        cv2.imshow('frame',frame)   
        
    else:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.imshow('frame23',hsv)
        lower_blue = np.array([minb0,ming0,minr0])
        upper_blue = np.array([maxb0,maxg0,maxr0])
        mask0 = cv2.inRange(hsv, lower_blue,upper_blue)
       
        lower_blue = np.array([minr1,ming1,minb1])
        upper_blue = np.array([maxr1,maxg1,maxb1])
        mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
        
        lower_blue = np.array([minr2,ming2,minb2])
        upper_blue = np.array([maxr2,maxg2,maxb2])
        mask2 = cv2.inRange(hsv, lower_blue, upper_blue)
        
        lower_blue = np.array([minr3,ming3,minb3])
        upper_blue = np.array([maxr3,maxg3,maxb3])
        mask3 = cv2.inRange(hsv, lower_blue, upper_blue)
        
        lower_blue = np.array([minr4,ming4,minb4])
        upper_blue = np.array([maxr4,maxg4,maxb4])
        mask4 = cv2.inRange(hsv, lower_blue, upper_blue)
        
        lower_blue = np.array([minr5,ming5,minb5])
        upper_blue = np.array([maxr5,maxg5,maxb5])
        mask5 = cv2.inRange(hsv, lower_blue, upper_blue)
        
        lower_blue = np.array([minr6,ming6,minb6])
        upper_blue = np.array([maxr6,maxg6,maxb6])
        mask6 = cv2.inRange(hsv, lower_blue, upper_blue)
        
        lower_blue = np.array([minr7,ming7,minb7])
        upper_blue = np.array([maxr7,maxg7,maxb7])
        mask7 = cv2.inRange(hsv, lower_blue, upper_blue)
        
        lower_blue = np.array([minr8,ming8,minb8])
        upper_blue = np.array([maxr8,maxg8,maxb8])
        mask8 = cv2.inRange(hsv, lower_blue, upper_blue)
        
        lower_blue = np.array([110,50,50])
        upper_blue = np.array([130,255,255])
        mask10 = cv2.inRange(hsv, lower_blue,upper_blue)
        
        final=mask10
        cv2.imshow('frame',frame)
        cv2.imshow('frameeee',final)   
        
        
   
    
    if cv2.waitKey(1) & 0xFF == ord('b'):
        kraj=True
        box,box1,box2,box3,box4,box5,box6,box7,box8=getColorSquare(self,frame)
        minr0,ming0,minb0,maxr0,maxg0,maxb0=minmaxvalue(box)
        minr1,ming1,minb1,maxr1,maxg1,maxb1=minmaxvalue(box1)
        minr2,ming2,minb2,maxr2,maxg2,maxb2=minmaxvalue(box2)
        minr3,ming3,minb3,maxr3,maxg3,maxb3=minmaxvalue(box3)
        minr4,ming4,minb4,maxr4,maxg4,maxb4=minmaxvalue(box4)
        minr5,ming5,minb5,maxr5,maxg5,maxb5=minmaxvalue(box5)
        minr6,ming6,minb6,maxr6,maxg6,maxb6=minmaxvalue(box6)
        minr7,ming7,minb7,maxr7,maxg7,maxb7=minmaxvalue(box7)
        minr8,ming8,minb8,maxr8,maxg8,maxb8=minmaxvalue(box8)
        
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
           break
            
    

#print self.hand_row_nw
#plt.hist(self.hand_hist.ravel(),256,[0,256]); plt.show()
#color = ('b','g','r')
#for i,col in enumerate(color):
#    histr = cv2.calcHist([frame],[i],None,[256],[0,256])
#    plt.plot(self.hand_hist,color = col)
#    plt.xlim([0,256])
#plt.show()   
    
    

#while(1):q
   # ret, frame = cap.read()

    #res=apply_hist_mask(frame, self.hand_hist) 
    #draw_final(self, frame,res) 
    #cv2.imshow('as',res)
    #cv2.imshow('frame',frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break
 
cap.release()
cv2.destroyAllWindows()


# In[ ]:




# In[ ]:




# In[ ]:



