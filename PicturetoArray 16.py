import numpy as np
from PIL import Image
import time

filename = '.jpg'
c = 1
k = 0

for i in range(1, 21):
   Testarr = np.ones((12288,10000))    
   for p in range(10000):
         
         image = Image.open(str(c) + filename,'r')
         L = list(image.getdata())

         for d in range(4096):
            for z in range(3):
                Testarr[k][p] = L[d][z]
                k += 1
         
                
         k = 0       
         c += 1

         
         
   np.save(str(i)+'TBilderY'+'.npy', Testarr)
   print(c)

