#Bibliotheken werden importiert            
import time
import numpy
import numpy as np

#Hyperparameter

hiddenLayer = []
alpha = 0.01
iterations = 150
cache = 0

w_out: np.array
z_out: np.array
a_out: np.array
dz_out: np.array
dw_out: np.array




#Klasse für die Hidden-Layer

class layer():

    neuronen  : int
    neuronen2 : int
    schicht   : int

    w  : np.array
    b  : np.array
    z  : np.array
    a  : np.array
    dw : np.array    
    dz : np.array 
    db : np.array
    
        
    def __init__(self, neuronen, schicht, neuronen2): 

        self.neuronen = neuronen
        self.schicht  = schicht
        self.neuronen2 = neuronen2                                 #neuronen2 ist die Anzahl der Neuronen aus der Vorherigen Schicht
        self.w = np.random.randn(neuronen2, neuronen) 
        self.b = np.random.randn(neuronen, 1)
        self.dw = np.array([])
        self.dz = np.array([])
        self.db = np.array([])

        print('Layer ',self.schicht,' wurde erstellt')
      

    #Forward-Propagation für die Hidden-Layer
    def forward(self,a2):                                               
    
        self.z = np.dot(self.w.T,a2) + self.b
        self.a = np.maximum(self.z, 0)


    #Back-Propagation für die Hidden-Layer    
    def backward(self,dw_2,dz_2):

        self.dw  = np.zeros((self.neuronen,self.neuronen2))
        self.db  = np.zeros((self.neuronen, 1))

        for j in range(1, 20 + 1):

            self.z  = np.load(str(self.schicht) + '-' + str(j) + ' Z.npy')

            if self.schicht - 1 == 0:

                a_2    = np.load(str(j) + 'TBilderX.npy')

            else:    
                a_2     = np.load(str(self.schicht - 1) + '-' + str(j) + ' A.npy')

            da_dz   = self.z >= 0 #self vergessen (z)
            da_dz   = da_dz.astype(np.int)

            self.dz = np.dot(dw_2.T, dz_2) * da_dz                  
            self.dw += np.dot(self.dz, a_2.T) * (1/10000)
            self.db += (np.sum(self.dz, axis = 1, keepdims = True)) * (1/10000)

            print('Layer >> ',self.schicht,' Back Propagation mit dem >> ',j,' Dataset')

        self.dw = self.dw * (1 / 20) 
        self.db = self.db * (1 / 20)

            
        #Gradienten Abstieg
        self.w = self.w - alpha * self.dw.T
        self.b = self.b - alpha * self.db

    
    #Get-Methoden der Klasse layer
    def getA(self):
        return self.a


    def getW(self):
        return self.w


    def getB(self):
        return self.b


    def getZ(self):
        return self.z


    def getdW(self):
        return self.dw


    def getdZ(self):
        return self.dz


    def getSchicht(self):
        return self.schicht


    def getNeuronen(self):
        return self.neuronen

        
#Main Programm , welches die Iterationen ausführt
def main():
    global dw_out,db_out

    cache = 0

    lay = int(input('Wie viele Hidden-Layer möchten Sie haben >> '))
    neuronen2 = 768

    for i in range(1,lay + 1):

       neuronen = int(input('Wie viele Neuronen moechten Sie im '+ str(i) +' Layer haben >> '))
       schicht = i
       hiddenLayer.append(layer(neuronen, schicht ,neuronen2))
       neuronen2 = neuronen

    n = hiddenLayer[lay - 1].getNeuronen()
    w_out = np.random.randn(n, 12288)
    b_out = np.random.randn(12288,1)

    print('Model wurde Initialisiert mit ',len(hiddenLayer),' Hidden Layern')

    for f in range(iterations):

        #Forward-Propagation für die Hidden-Layer
             
            start = time.time()

            for i in range(1, lay + 1):
                

                for j in range(1, 20 + 1):
                    
                    if i == 1:

                        x = np.load(str(j)+'TBilderX.npy')
                        hiddenLayer[i - 1].forward(x)
                        np.save(str(1) + '-' + str(j) + ' A.npy', hiddenLayer[i - 1].getA()) #Datei-Notation: n/m A.npy (n := layer (Laufindex: i), m := die m'te Examplegruppe(Laufindex: j))
                        np.save(str(1) + '-' + str(j) + ' Z.npy', hiddenLayer[i - 1].getZ())

                    else:

                        a = np.load(str(i - 1) + '-' + str(j) + ' A.npy')
                        hiddenLayer[i - 1].forward(a)
                        np.save(str(i) + '-' + str(j) + ' A.npy', hiddenLayer[i - 1].getA())
                        np.save(str(i) + '-' + str(j) + ' Z.npy', hiddenLayer[i - 1].getZ())


                    print('Layer >> ',i,' Forward Propagation mit dem >> ',j,' Dataset')

            
            stop = time.time()

            if stop-start >= 60:

                print('Die Forward Propagation hat >>',(round(stop-start)) / 60,' Minuten gedauert')   

            else:

                print('Die Forward Propagation hat >>', round(stop-start),' Sekunden gedauert')


            #Forward-Propagation für den Output-Layer

            for j in range(1, 20 + 1):

                a = np.load(str(lay) + '-' + str(j) + ' A.npy')
                
                z_out = np.dot(w_out.T, a) + b_out
                
                a_out = np.maximum(z_out,0)

                np.save('Output' + '-' + str(j) + ' A.npy', a_out)
                np.save('Output' + '-' + str(j) + ' Z.npy', z_out)

                print('Output Layer >> ',' Forward Propagation mit dem >> ',j,' Dataset')
                
                



            Loss = np.square(np.load('1TBilderY.npy') - np.load('Output-2 A.npy'))
            J = Loss.mean() 

            if J < cache:

                print('Kleiner')

            else:

                print('Größer')

            cache = J

            #Back-Propagation für den Output-Layer

            dw_out  = np.zeros((12288,hiddenLayer[lay - 1].getNeuronen()))
            db_out  = np.zeros((12288, 1))

            start = time.time()

            for j in range(1, 20 + 1):       
                
                 z_out     = np.load('Output' + '-' + str(j) + ' Z.npy')
                 
                 da_dz     = z_out >= 0 
                 da_dz     = da_dz.astype(np.int)
                
                 y         = np.load(str(j) + 'TBilderY.npy')

                 a         = np.load(str(lay-1) + '-' + str(j)+' A.npy')
                 a_out     = np.load('Output' + '-' + str(j) + ' A.npy')
                 dz_out    = -2 * (y - a_out) * da_dz
                 
                 
                 dw_out    += np.dot(dz_out, a.T) * (1/10000)
                 db_out    += np.sum(dz_out, axis = 1, keepdims = True) * (1/10000)
                 

                 print('Output Layer >> ',' Back Propagation mit dem >> ',j,' Dataset')


            
            stop = time.time()

            if stop-start >= 60:

                print('Die Back Propagation hat >>',(round(stop-start)) / 60,' Minuten gedauert')   

            else:

                print('Die Back Propagation hat >>', round(stop-start),' Sekunden gedauert')

            dw_out = dw_out * (1 / 20) 
            db_out = db_out * (1 / 20)

 
            #Gradientenabstieg
            w_out = w_out - alpha * dw_out.T
            b_out = b_out - alpha * db_out
           
   
            for i in range(1, lay +1): 

                if i == 1:
                    hiddenLayer[lay - i].backward(dw_out, dz_out)
                else:
                    hiddenLayer[lay - i].backward(hiddenLayer[lay - i + 1].getdW(), hiddenLayer[lay - i + 1].getdZ())

                

            print(f+1,' >> Iteration')

            if f % 10 == 0:

                if f == 0:
                    pass

                else:
                    np.save('Output' + '-' + str(j) + ' W.npy', w_out)
                    np.save('Output' + '-' + str(j) + ' B.npy', b_out)

                    for elements in range(1,lay+1):

                        np.save('Gewichte' + '-' + str(elements) + ' W.npy', hiddenLayer[elements -1].getW())
                        np.save('Bias' + '-' + str(elements) + ' B.npy', hiddenLayer[elements - 1].getB())
                       
main()
