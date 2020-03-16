import matplotlib.pyplot as plt
import random


x_data = ['3750', '5000', '6000', '8000', '10000', '12000', '14000', '16000', '18000', '20000']
y_data = [32.0654, 32.0855, 32.0251, 32.1348, 32.0657, 32.1301, 32.1557, 32.1583, 32.1504, 32.1590]
y_data2 = [31.8468, 31.8927, 31.8264, 31.9103, 31.8745, 31.9524, 31.9897, 31.9925, 32.0013, 31.9906]
y_data3 = [32.38]*10
y_data4 = [31.68]*10

plt.plot(x_data,y_data, color='red',linewidth=2.0,linestyle=':')
plt.plot(x_data,y_data2, color='blue',linewidth=2.0,linestyle='-.')
plt.plot(x_data,y_data3, color='green',linewidth=2.0,linestyle='--')
plt.plot(x_data,y_data4, color='yellow',linewidth=2.0,linestyle='--')
plt.legend(["DBPN-R64-10-with-ensemble", "DBPN-R64-10", "DBPN-R64-10-in-paper", "DRRN-in-paper"], loc=2)
plt.xlabel("epoch")
plt.ylabel("PSNR")
plt.show()