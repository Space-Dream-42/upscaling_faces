from PIL import Image
import os





def Resizer():

    for i in range (100000,260600):
    
        imageFile = str(i)+'.jpg'
        celeb1 = Image.open(imageFile)

        width = 16
        height = 16

        celeb2 = celeb1.resize((16,16), Image.NEAREST)
        ext = '.jpg'
        celeb2.save(str(i) + ext)
    print('Fertig!')

def Delete():
    for i in range (10000,100000):
        os.remove('0' + str(i)+'.jpg')
    print('Fertig!')
    
