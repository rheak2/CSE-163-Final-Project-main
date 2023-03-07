'''
docstring
'''

from utils import extinction_level_numerical
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

sns.set()

def manipulate_data(data:pd.DataFrame) -> pd.DataFrame:
    data = extinction_level_numerical(data)
    

# Read in data
data = pd.read_csv('/home/filename.csv')

# Separate data into features and labels
features = data.loc[:, data.columns != 'target_col']
labels = data['target_col']

# Create an untrained model
model = DecisionTreeClassifier() # or DecisionTreeRegressor()
# Train it on our training data
model.fit(features, labels)

# Make predictions on the data
predictions = model.predict(features)
# Assess the accuracy of the model
accuracy_score(labels, predictions) # or mean_squared_error(labels, predictions)