import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('winequality-white.csv', header=0, sep=';')

df['quality'].plot(kind='hist', bins=7, color='grey')

plt.xlabel('Quality')
plt.ylabel('Count')
plt.title('White wines')
plt.axis([2, 10, 0, 2500])
plt.show()

df = pd.read_csv('winequality-redT.csv', header=0, sep=';')

df['quality'].plot(kind='hist', bins=6, color='red')

plt.xlabel('Quality')
plt.ylabel('Count')
plt.title('Red wines')
plt.axis([2, 9, 0, 800])
plt.show()