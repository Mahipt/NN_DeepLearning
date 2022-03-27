import numpy as np 



array = np.zeros(10) 
for i in range(array.size):
    array[i] = 10 - i

print(array) 

array = np.sort(array)
print(array) 

