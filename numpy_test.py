import numpy as np 
a = np.random.rand(5,6)
f = np.random.rand(2,2)
print a
print
print f
print
c = a[0:2,0:2]
print c
print
o = f*c
print o
print
print o.sum()

print len(a[0])