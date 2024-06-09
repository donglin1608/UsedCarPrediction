import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def plot_milage_price_histogram_with_regression(file_prefix):
    # Read the CSV file
    df = pd.read_csv(f'{file_prefix}_train.csv')

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist2d(df['milage'], df['price'], bins=30, cmap='viridis')
    plt.colorbar(label='Count')
    plt.xlabel('Milage')
    plt.ylabel('Price')
    plt.title('Histogram of Milage against Price')

    # Prepare data for regression
    X = df['milage'].values.reshape(-1, 1)
    y = df['price'].values

    # Fit the linear regression model
    reg = LinearRegression().fit(X, y)

    # Predict using the model
    X_pred = np.linspace(df['milage'].min(), df['milage'].max(), 100).reshape(-1, 1)
    y_pred = reg.predict(X_pred)

    # Plot the regression line
    plt.plot(X_pred, y_pred, color='red', linewidth=2, label='Regression Line')
    plt.legend()

    # Show the plot
    plt.show()

    # Print the model coefficients and intercept
    print(f"Coefficient (slope): {reg.coef_[0]}")
    print(f"Intercept: {reg.intercept_}")

    return reg

# Call the function with 'Acura_1' and get the regression model
acura_1_regression_model = plot_milage_price_histogram_with_regression('Acura_1')
acura_2_regression_model = plot_milage_price_histogram_with_regression('Acura_2')
acura_3_regression_model = plot_milage_price_histogram_with_regression('Acura_3')
acura_4_regression_model = plot_milage_price_histogram_with_regression('Acura_4')
acura_5_regression_model = plot_milage_price_histogram_with_regression('Acura_5')

