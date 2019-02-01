# importing essential packages
# pandas is preferred than dask, spark [due to few training samples]
import pandas as pd

# scikit learn is used to apply scientific operation on dataset
from sklearn.model_selection import train_test_split

# choosing CountVectorization Algorithm because of high frequency in words like who,what etc.
from sklearn.feature_extraction.text import CountVectorizer

# Logistic Regression model [Multinomial] is preferred
# Due to more than one possible discrete output and
# Vectorization gives a numeric output
from sklearn.linear_model import LogisticRegression

filepath = 'data/nlpdata.txt'

# reading data as a dataframe from txt file and colnames given
df = pd.read_csv(filepath, names=['sentence', 'label'], sep=' ,,, ', engine='python')
sentences = df['sentence'].values
labels = df['label'].values

# spliting the data for test purpose [25%]
sentences_test, sentences_train, y_test, y_train = train_test_split( sentences, labels, test_size = 0.25, random_state = 1000)

# initilize and fit the text samples available
vectorizer = CountVectorizer()
vectorizer.fit(sentences)

# transforming samples into document-term matrix [train and test data]
X_train = vectorizer.transform(sentences)
X_test  = vectorizer.transform(sentences_test)

# initilizing classifier and training the dataset
classifier = LogisticRegression()
classifier.fit(X_train, labels)

# predicting class of a unknown question
question = [input("Enter your statement: ")]
predicted_class = classifier.predict(vectorizer.transform(question))
print("Predicted Class: ", predicted_class[0])
