from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,f1_score
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.metrics import confusion_matrix,f1_score, precision_score, recall_score, accuracy_score

def logistic_regression(X,X_test,y,y_test,start_time,gv_scroll_size_total,non_gv_scroll_size_total,trends_df,summary_df):

    lr = LogisticRegression()
    lr.fit(X.iloc[:, 6:], y)

    # print('logistic regression')
    # print('f1 score ----------',f1_score(y_test,lr.predict(X_test.iloc[:,6:])))
    # print('accuracy score--------',lr.score(X_test.iloc[:,6:],y_test))

    gv_20 = np.argsort(lr.coef_)[0][-50:]

    features = X.iloc[:, 6:].columns

    cm = confusion_matrix(y_test, lr.predict(X_test.iloc[:, 6:]))
    prediction = lr.predict(X_test.iloc[:, 6:])
    prediction = pd.DataFrame({'Predicted_value': prediction[0:,]})

    tf_idf_size = np.shape(X.iloc[:, 6:])[1]
    f1 = f1_score(y_test, lr.predict(X_test.iloc[:, 6:]))
    accuracy = accuracy_score(y_test, lr.predict(X_test.iloc[:, 6:]))
    precision = precision_score(y_test, lr.predict(X_test.iloc[:, 6:]))
    recall = recall_score(y_test, lr.predict(X_test.iloc[:, 6:]))

    trends_new_row = pd.Series(data=features[gv_20], name=start_time)
    summary_new_row = pd.Series(
        data=[start_time, gv_scroll_size_total, non_gv_scroll_size_total, tf_idf_size, f1, accuracy, precision, recall,
              len(X), len(X_test), cm[0][0], cm[0][1], cm[1][0], cm[1][1]])

    trends_df = trends_df.append(trends_new_row, ignore_index=True)
    summary_df = summary_df.append(summary_new_row, ignore_index=True)
    return trends_df, summary_df,prediction
