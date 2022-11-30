from PIL import Image
import numpy as np

Array = np.load('Test.npy')

def ImageCompile(Array):

    data = np.zeros((64, 64, 3),dtype=np.uint8)
    i = 0
    
    for h in range(64):
        for w in range(64):
            data[h,w] = [Array[i][1],Array[i+1][1],Array[i+2][1]]
            i += 3

    img = Image.fromarray(data, 'RGB')
    img.save('my2.2.jpg')
    img.show()
