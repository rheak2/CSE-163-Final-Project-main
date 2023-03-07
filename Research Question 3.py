'''
docstring
'''

import utils
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import species_data_processing_file

sns.set()

def manipulate_data(data:pd.DataFrame) -> pd.DataFrame:
    data = utils.extinction_level_numerical(data)
    data = utils.tl_change_between_multiple_yrs(2007, 2021, data)
    return data

def create_and_train_model(data: pd.DataFrame) -> Any:
    # Separate data into features and labels
    features = data.loc[:, ['Common name', 'location']]
    features = pd.get_dummies(features)
    labels = data['target_col']

# Create an untrained model
model = DecisionTreeClassifier() # or DecisionTreeRegressor()
# Train it on our training data
model.fit(features, labels)

# Make predictions on the data
predictions = model.predict(features)
# Assess the accuracy of the model
accuracy_score(labels, predictions) # or mean_squared_error(labels, predictions)


def main():
    df = species_data_df
    df = manipulate_data(df)



if __name__ == "__main__":
   main()