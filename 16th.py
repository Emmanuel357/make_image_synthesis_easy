import numpy as np
from scipy import stats


#snp.random.seed(41)
data = np.random.normal(0,1,(64,512)) #sorted(stats.lognorm.rvs(s=0.5, loc=1, scale=1000, size=1000))
print(data)

print(np.array(data).shape)#; quit()

# fit normal distribution
mean, std = stats.norm.fit(data)
print(mean)
print(std)
pdf_norm = stats.norm.pdf(data, mean, std)
print(stats.norm.fit(pdf_norm))
print(pdf_norm)


