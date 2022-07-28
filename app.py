from numpy import positive
import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
model = pickle.load(open('./models/RandomForest.pkl', 'rb'))
CountVectorizer = pickle.load(open('./models/CountVectorizer.pkl', 'rb'))

def data_cleaning(text):
    corpus=[]
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    corpus.append(text)
    return countVectorizer(corpus)


def prediction(vector):
    return model.predict(vector)[0]


def countVectorizer(text):
    vector = CountVectorizer.transform(text).toarray()
    return prediction(vector)


def main():

    st.title("What's The Sentiment")
    text = st.text_area('Text to analyze',"")
    if st.button('Run Analysis'):
        # 1-> positive
        # 0-> negative
        sentiment=data_cleaning(text)
        if sentiment==1:
            st.success('The sentiment is Positive ')
        else :
            st.error('The sentiment is negative')

if __name__ == "__main__":
    main()
