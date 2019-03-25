import numpy as np

a = [[[0, 1, 2], [0, 1, 2]],[[ 3, 4, 5], [0, 1, 2]], [[6, 7, 8], [0, 1, 2]]]
b = [5, 5, 5]
a = np.array(a)
b = np.array(b)

#d = np.linalg.norm(a[:,:]-b)
c = np.zeros(((3,2,3)))
c[:,:] = b

d = (a-c)**2
e = d[:,:,0] + d[:,:,1] + d[:,:,2]
print (e)

