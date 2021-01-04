import pandas as pd


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

data=pd.read_csv("model_data.csv",index_col="Date")
data.index=pd.to_datetime(data.index)


train_threshold=int(len(data)*0.8)

train_x=data.iloc[:train_threshold,:-1]
test_x=data.iloc[train_threshold:,:-1]

train_y=data.iloc[:train_threshold,-1]
test_y=data.iloc[train_threshold:,-1]



params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }



xgb_model=XGBClassifier(min_child_weight=10,gamma=1,subsample=1,colsample_bytree=0.8,max_depth=3)
xgb_model.fit(train_x, train_y)

# make predictions for test data
y_pred_with_prob = pd.DataFrame(xgb_model.predict_proba(test_x))
y_pred_with_prob.columns = xgb_model.classes_

y_pred_with_prob.index = test_y.index

y_pred = xgb_model.predict(test_x)

# evaluate predictions
accuracy = accuracy_score(test_y, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

print("-" * 100)



my_values=test_x.iloc[-1:,:]

xgb_model.predict_proba(my_values)



import pickle


pickle.dump(xgb_model,open("ml_model.pkl","wb"))



# from sklearn.metrics import plot_confusion_matrix
# import matplotlib.pyplot as plt
#
# plot_confusion_matrix(xgb_model,test_x,test_y)
# plt.show()