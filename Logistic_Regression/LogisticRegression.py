# importing libraries
import nltk
import pandas as pd
import numpy as np 
import re
from gensim.models import Word2Vec
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn import metrics

# importing the stop words
stop_words = set(nltk.corpus.stopwords.words('english')) 

# importing the sample dataset from Emotion_Phrases.csv
data = pd.read_csv("datasets/Emotion_Phrases.csv", header=None)

# adding the coloumns names
data.columns = ['emotions', 'text']

# getting the informations of data
data.info()

# converting the emotions to the ids
data['emo_id'] = data['emotions'].factorize()[0]

# print(data.head(10))
# print(data['emo_id'])

# dropped the duplicates and sorted it
emo_id_df = data[['emotions', 'emo_id']].drop_duplicates().sort_values('emo_id')

# converted it into a dictionary 
emo_to_id = dict(emo_id_df.values)

# representing the dataset with a plot
fig = plt.figure()
data.groupby('emotions').text.count().plot.bar(ylim=0)
plt.show()

# converted the data into the dictionary
data_dict = data.to_dict()

sw_rem = []
txt = []

# preprocessed the database:
#       *   Removed the words other than A-Z and a-z and space
#       *   Tokenize the words
#       *   Convert the tokenized words into a list
for k in range(len(data_dict['text'])):
    data_dict['text'][k] = re.sub('([^A-Za-z ])', '', data_dict['text'][k])
    txt.append(nltk.tokenize.word_tokenize(data_dict['text'][k]))
    for j in range(len(txt[k])):
        if txt[k][j] not in stop_words: 
           sw_rem.append(txt[k][j])
    data_dict['text'][k] = " ".join(sw_rem)
    sw_rem = [] 
data = pd.DataFrame.from_dict(data_dict)

# using tfidf vectorizer to convert the words into vectors as words that occur more frequently in one document and less frequently in other documents should be given more importance as they are more useful for classification.
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=1, norm='l2', encoding='utf-8', ngram_range=(1,2), stop_words='english')
l_data = data.text.tolist()
fitted_vectorizer=tfidf.fit(l_data)
features=fitted_vectorizer.transform(l_data)
labels = data.emo_id

# split train and test data
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, data.index, test_size=0.33, random_state=0)

# instantiate the model 
model = LinearSVC()

# train the model
model.fit(X_train, y_train)

# used the train model to predict the dataset
y_pred = model.predict(X_test)

# confusion matrix
conf_mat = metrics.confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=emo_id_df.emotions.values, yticklabels=emo_id_df.emotions.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# generated an accuracy report based on the tests
print(metrics.classification_report(y_test, y_pred, target_names=data['emotions'].unique()))

# print(clf.predict(count_vect.transform(["Angry Angry Angry Angry"])))