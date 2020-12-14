import matplotlib.pyplot as plt
import math
import sys


def readDenseOutput(filename, index):
    t = []
    x = []
    header:bool = True
    with open(filename, 'r') as file:
        for row in file:
            if(row.strip() == "Time series:"):
                header = False
                continue
            
            if header == False:
                row = row.strip().split(',')
                t.append(float(row[0]))
                x.append(float(row[index+1]))
    return t, x
    
   
filename = sys.argv[1]
index    = int(sys.argv[2])
   
t, x = readDenseOutput(filename, index)

print([t[-1], x[-1]])

plt.figure(1)
plt.subplots_adjust(0.1, 0.1, 0.98, 0.95)
plt.plot(t, x, 'b-', linewidth=2)
#plt.plot(t, [abs(x[i]) for i in range(0, len(x))], 'b-', linewidth=2)
#plt.yscale("log")
plt.xlabel('time')
plt.ylabel('x'+sys.argv[2])
plt.grid('both')
plt.show()
