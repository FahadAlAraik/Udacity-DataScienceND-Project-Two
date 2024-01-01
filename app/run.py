import json
import plotly
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """
    tokenize(text) function to process text: lower, tokenize, remove stop words, remove special characters, and lemmatize 
    -
    Parameters
        -text (str)
    Output: 
        -tokens (list(str))
    """
    # Make the text lowercase
    text = text.lower()

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Remove special characters using regex
    tokens = [re.sub(r'[^a-zA-Z0-9]', '', token) for token in tokens]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Count the most 15 occurred words using CountVectorizer
    from sklearn.feature_extraction.text import CountVectorizer

    # Instantiate CountVectorizer
    count_vectorizer = CountVectorizer(tokenizer=tokenize)

    # Fit and transform the messages column
    messages_count = count_vectorizer.fit_transform(df['message'])

    # Sum the occurrences of each word
    word_counts = messages_count.sum(axis=0)

    # Get the feature names (words)
    words = count_vectorizer.get_feature_names()

    # Create a DataFrame with words and their counts
    word_counts_df = pd.DataFrame({'word': words, 'count': word_counts.tolist()[0]})

    # Sort the DataFrame by count in descending order and select the top 15
    top_words =word_counts_df.sort_values(by='count', ascending=False)
    top_words = top_words.iloc[1:].head(15)
    top_words = top_words[top_words['word'].apply(lambda x: len(x) > 2)].head(15)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
        # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_words['word'],
                    y=top_words['count']
                )
            ],
            'layout': {
                'title': 'Top 15 Most Occurred Words',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Word"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()