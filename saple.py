import numpy as np
total_array = []
array1 = [[1,2,3],[4,5,6],[7,8,9],[111,222,333]]
array2 = [[11,21,31],[41,51,61],[71,81,91],[444,444,444]]
array3 = [[12,22,32],[42,52,62],[72,82,92],[555,666,777]]
array4 = [[121,221,321],[421,521,612],[721,821,921],[5551,6661,7771]]
array5 = [[1211,2211,3211],[4211,5211,6121],[7211,8211,9211],[55511,66611,77711]]
total_array.append(array1)
total_array.append(array2)
total_array.append(array3)
total_array.append(array4)
total_array.append(array5)

total_array = np.array(total_array)
flatten_array = total_array.flatten()
print(total_array.shape)
print(flatten_array)
