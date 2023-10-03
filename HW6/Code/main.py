from utils import *
import json


if __name__== '__main__':
    X_train, y_train = load_data('C:/Users/admin/OneDrive - National Economics University/ML/ML NEU/ML1/HW solution/Logistic Regression/Data/ds1_train.csv')
    X_test, y_test = load_data('C:/Users/admin/OneDrive - National Economics University/ML/ML NEU/ML1/HW solution/Logistic Regression/Data/ds1_valid.csv')
    logistic_reg = LogisticRegression(alpha=0.1, epoch=10000000)
    logistic_reg.fit(X_train, y_train)
    y_pred = logistic_reg.predict(X_test)

    logistic_reg.plot('C:/Users/admin/OneDrive - National Economics University/ML/ML NEU/ML1/HW solution/Logistic Regression/Output/output_graph.png')
    
    acc_score = accuracy_score(y_test, y_pred)
    result = {'accuracy_score': acc_score}
    
    output_path = r'C:/Users/admin/OneDrive - National Economics University/ML/ML NEU/ML1/HW solution/Logistic Regression/Output/output.json'
    with open(output_path, 'w') as json_file:
        json.dump(result, json_file)  
