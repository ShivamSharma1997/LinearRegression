import pandas as pd
import numpy as np

from LinearRegression import LinearRegression

########################## IMPORTING AND PROCESSING TRAINING DATASET ##########################

train = pd.read_csv('../data/train.csv')

train_keys = train.keys()

y = np.array(train[train_keys[1]])
pclass = np.array(train[train_keys[2]])
sex = np.array(train[train_keys[4]])
age = np.array(train[train_keys[5]])
sibsp = np.array(train[train_keys[6]])
parch = np.array(train[train_keys[7]])
fare = np.array(train[train_keys[9]])

age = np.array(pd.DataFrame(age).fillna(method='pad'))

for i in range(len(sex)):
	if(sex[i] == 'male'):
		sex[i] = 1
	else:
		sex[i]= 0

X = [list(pclass), list(sex), list(sibsp), list(parch), list(age)]
X = np.transpose(X)

########################## TRAINING REGRESSION MODEL ##########################

regr = LinearRegression(max_epoch=30,al=0.5)
regr.fit(X, y)

######################### IMPORTING AND PROCESSING TEST DATASET ##########################

test = pd.read_csv('../data/test.csv')

test_keys = test.keys()

p_id = np.array(test[test_keys[0]])
pclass = np.array(test[test_keys[1]])
sex = np.array(test[test_keys[3]])
age = np.array(test[test_keys[4]])
sibsp = np.array(test[test_keys[5]])
parch = np.array(test[test_keys[6]])
fare = np.array(test[test_keys[8]])

age = np.array(pd.DataFrame(age).fillna(method='pad'))

for i in range(len(sex)):
	if(sex[i] == 'male'):
		sex[i] = 1
	else:
		sex[i]= 0

X = [list(pclass), list(sex), list(sibsp), list(parch), list(age)]
X = np.transpose(X)
########################## PREDICTIONS ON TESTING DATASET ##########################

pred = regr.predict(X)

Y_new = []
th = np.mean(pred)

for p in pred:
    if p > th:
        Y_new.append(1)
    else:
        Y_new.append(0)

########################## SAVING PREDICTIONS ##########################

res = pd.DataFrame()

res['PassengerId'] = p_id
res['Survived'] = Y_new

res.to_csv('../data/my_solution.csv',index=0)