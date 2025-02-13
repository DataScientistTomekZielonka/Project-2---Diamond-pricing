import numpy as np
import pandas as pd
import joblib

# clusters_scaler = joblib.load('../models/clusters/clusters_scaler.joblib')

# Changing categorical values into numerical
def transform_diamond_features(row):
    """
    Transform categorical features (cut, color, clarity) into numerical values for a single diamond row.
    
    Parameters:
    row (pd.Series): A single row from the diamonds dataset
    
    Returns:
    pd.Series: Transformed row with numerical values
    """
    # Define the mappings
    cut_class = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    cut_numeric = list(range(1, 6))
    color_class = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
    color_numeric = list(range(8, 1, -1))
    clarity_class = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
    clarity_numeric = list(range(1, 9))
    
    # Create mapping dictionaries
    cut_map = dict(zip(cut_class, cut_numeric))
    color_map = dict(zip(color_class, color_numeric))
    clarity_map = dict(zip(clarity_class, clarity_numeric))
    
    # Create a copy of the row to avoid modifying the original
    transformed_row = row.copy()
    
    # Transform each feature
    transformed_row['cut'] = cut_map.get(row['cut'], row['cut'])
    transformed_row['color'] = color_map.get(row['color'], row['color'])
    transformed_row['clarity'] = clarity_map.get(row['clarity'], row['clarity'])
    
    return transformed_row


# Scaling whole dataset for classification
def scale_for_clusters(scaler, input_row):

    """
    Scale features for clustering.
    
    Parameters:
    scaler: sklearn scaler object
    input_row: pd.DataFrame or pd.Series
    
    Returns:
    np.array: Scaled values
    """
    
    if isinstance(input_row, pd.Series):
        input_row = pd.DataFrame([input_row])
    scaled_row = scaler.transform(input_row)
    #return scaled_row.flatten()  # Flatten back to 1D for assignment
    return scaled_row


    # Ensure the row is a 2D array (reshape for scaler)
    #scaled_row = scaler.transform(input_row.values.reshape(1, -1))  # Reshape to 2D
    #return scaled_row.flatten()  # Flatten back to 1D for assignment


# Soft voting for the proper class
def soft_voting_classifier(models, input_row):
    """
    Computes a soft voting prediction from multiple models.

    Args:
        models (list): A list of trained models.
        input_row (pd.DataFrame or np.ndarray): A single row of predictors (2D format).

    Returns:
        int: The predicted class label from the soft voting classifier.
    """
    
    # Ensure input_row is 2D
    #input_row_2d = input_row.values.reshape(1, -1)
    input_row = pd.DataFrame(input_row)

    # Collect probabilities from all models
    #probabilities = [model.predict_proba(input_row_2d) for model in models]
    
    probabilities = [model.predict_proba(input_row) for model in models]
    
    # Average the probabilities across models
    avg_probabilities = np.mean(probabilities, axis = 0)
    
    # Return the class with the highest average probability
    predicted_class = np.argmax(avg_probabilities, axis = 1)
    
    return predicted_class[0]

def predict(x : pd.Series, scaler, model) -> float :
    x_scaled = scaler.transform(x.values.reshape(1, -1))
    prediction = model.predict(x_scaled)
    return prediction

