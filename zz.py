import numpy as np

nn = 1000
Kinlc = []

for i in range(nn):

    #kval = np.random.lognormal(mean=Kfactor[0], sigma=Kfactor[1])
    kval = np.random.normal(0., 1.)
    kval = mu*np.exp(kval*sigma))
    Kincl.append(kval)

Kincl = np.array(Kincl)

print(Kincl.mean())
print(Kincl.var())
import matplotlib.pyplot as plt

count, bins, ignored = plt.hist(np.log(Kincl), 20, normed=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
                   linewidth=2, color='r')
plt.axis('tight')
plt.show()
