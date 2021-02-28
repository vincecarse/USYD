import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

#import/format data
prostata_table = pd.read_csv('https://web.stanford.edu/~hastie/ElemStatLearn/datasets/prostate.data', sep=None)
prostata_table.head()

#summary stats
prostata_table = prostata_table.drop('Unnamed: 0',axis=1)
prostata_table.corr(method='pearson')
prostata_table.cov()

#vis
sns.jointplot(x="lcavol", y="lweight", data=prostata_table)

#regression
X_one = prostata_table[['lcavol']]
X_one = sm.add_constant(X_one)
Y_one = prostata_table[['lpsa']]
reg1 = sm.OLS(Y_one, X_one)
result1 = reg1.fit()
print(result1.summary())

#find fitted values
Y_pred_one = result1.predict()
Y_pred_one.shape = (97,1)

#print summary
print('Parameters: ', result1.params)
print('Standard errors of regression coefficents: ', result1.bse)
print('R2: ', result1.rsquared)

#t-tests
print(result1.t_test(["const = 0"]))
print(result1.t_test(["lcavol = 0"]))

#more vis
fig1 = plt.figure()
plt.scatter(X_one['lcavol'], Y_one, label='data scatter', color='blue')
plt.plot(X_one['lcavol'], Y_pred_one, label='regression equation', color='red', linewidth=2.0,)
plt.xlabel('lcavol')
plt.ylabel('lpsa')
plt.title('Dummiest regression in the world')
plt.legend()
plt.xlim(-2,4.5)
plt.ylim(-1,6)
plt.show()

#reg 2
X = prostata_table[['lcavol','lweight','age','lbph','svi','lcp','gleason','pgg45']]
X = sm.add_constant(X)
Y = prostata_table[['lpsa']]
reg = sm.OLS(Y, X)
result = reg.fit()
print(result.summary())

#key info
print('Parameters: ', result.params)
print('Standard errors: ', result.bse)
print('R2: ', result.rsquared)


#more tests (inc F)
print(result.t_test("lcavol = 1"))
print(result.f_test(["lcavol = 1"]))
print(result.t_test(["lcavol = 100","lweight = 500"]))
print(result.f_test(["lcavol = 1", "lweight = 3"]))
