# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from google.colab import files
data = files.upload()

# Loading data
data = pd.read_csv('creditcard.csv')

# Normalizing the 'Amount' and 'Time' columns
data['NormalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data['NormalizedTime'] = StandardScaler().fit_transform(data['Time'].values.reshape(-1,1))

# Dropping the original 'Time' and 'Amount' columns
data = data.drop(['Time', 'Amount'], axis=1)

# Assigning feature and target variables
X = data.drop('Class', axis=1)
y = data['Class']

# Using the SMOTE function for dealing with class imbalance
X_resampled, y_resampled = SMOTE().fit_resample(X, y)

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Model Building and predicting on test data
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Print classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Calculate Area Under the Precision-Recall Curve
y_score = model.decision_function(X_test)
average_precision = np.average(precision_recall_curve(y_test, y_score)[0])
print('Area Under Precision-Recall Curve: {0:0.2f}'.format(average_precision))

# Plot Precision-Recall curve for the model
precision, recall, _ = precision_recall_curve(y_test, y_score)

plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
plt.show()