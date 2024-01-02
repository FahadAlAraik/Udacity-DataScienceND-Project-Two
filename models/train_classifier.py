# import libraries
import pandas as pd
import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
import sys


def load_data(database_filepath):
    """
    Function to load cleaned data from SQLite database
    -
    Parameters:
        - database_filepath: path to the database
    -
    Returns:
        - X: features
        - Y: target
        - category_names: categories names
    """
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('messages',engine)
    X = df['message']
    Y = df.drop(columns=['id','message','original','genre'])
    category_names = Y.columns
    return X,Y,category_names


def tokenize(text):
    """
    tokenize(text) function to process text: lower, tokenize, remove stop words and lemmatize 
    -
    Parameters
        -text (str)
    Returns: 
        -tokens (list(str))
    """
    # Make the text lowercase
    text = text.lower()

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens



def build_model():
    """
    Function to build model using scikit-learn pipeline
    -
    Parameters: N/A
    -
    Returns:
        - pipeline: pipelined model
    """
    
    # Define Pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)), #CountVectorizer
    ('tfidf', TfidfTransformer()),
    ('rfc', MultiOutputClassifier(RandomForestClassifier()))])
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
    }

    # Instantiate GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    return grid_search    


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function to evaluate the model using classification report function
    -
    Parameters: 
        - model: model(classifier)
        - X_test: test features
        - Y_test: test target
        - category_names: category names
    -
    Return: N/A
    """
    y_pred = model.predict(X_test)
    for i, column in enumerate(Y_test.columns):
        print(column, classification_report(Y_test[column], y_pred[:, i]))


def save_model(model, model_filepath):

    """
    Function to save model as a pickle file
    -
    Parameters:
        - model: machine learning model
        - model_filepath: the path that the model will be saved at
    -
    Returns: N/A
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        Y_train = Y_train.apply(pd.to_numeric, errors='coerce')
        Y_test = Y_test.apply(pd.to_numeric, errors='coerce')

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
