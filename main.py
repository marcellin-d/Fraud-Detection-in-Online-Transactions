import os
from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.model import train_model
from src.evaluate import evaluate_model
from src.utils import split_data

def main():
    """
    Main function to load, preprocess, train, evaluate and print results.
    """
    try:
        # Define the path for processed data
        data_path = './data/raw/creditcard.csv'
        
        # Check if the data file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}. Please check the file path.")
        
        # Load data
        df = load_data(data_path)
        
        # Preprocess the training and testing data
        df_processed = preprocess_data(df=df, output_path='./data/processed/creditcard_processed.csv')
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = split_data(df_processed, 'Class')
        
        
        # Train the model
        model = train_model(X_train, y_train, model_type='logistic', solver='liblinear', max_iter=1000)
        
        # Evaluate the model
        evaluation_results = evaluate_model(model, X_test, y_test, output_path='evaluation_report.txt')
        
        # Output the evaluation results
        print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
        print(f"Classification Report:\n{evaluation_results['classification_report']}")
        print(f"Confusion Matrix:\n{evaluation_results['confusion_matrix']}")
        
    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except KeyError as key_error:
        print(f"Error: {key_error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
if __name__ == "__main__":
    main()
