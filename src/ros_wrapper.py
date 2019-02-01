# importing essential packages
import rospy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# initilizing node
rospy.init_node("krithi_systems")

# rate with 2Hz
rate = rospy.Rate(2)

filepath = 'data/nlpdata.txt'
df = pd.read_csv(filepath, names=['sentence', 'label'], sep=' ,,, ', engine='python')
sentences = df['sentence'].values
labels = df['label'].values

sentences_test, sentences_train, y_test, y_train = train_test_split( sentences, labels, test_size = 0.25, random_state = 1000)
vectorizer = CountVectorizer()
vectorizer.fit(sentences)

X_train = vectorizer.transform(sentences)
X_test  = vectorizer.transform(sentences_test)

classifier = LogisticRegression()
classifier.fit(X_train, labels)

question = [input("Enter your statement: ")]
predicted_class = classifier.predict(vectorizer.transform(question))

# continuous loop
while(not rospy.is_shutdown()):
    print("Predicted Class: ", predicted_class[0])
    rate.sleep()
