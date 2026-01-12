import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK data (only first time)
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load trained model and vectorizer
import pickle

model = pickle.load(open(r"C:\\Users\\ASUS\\Desktop\\SMS Spam Classifier\\model.pkl","rb"))
tfidf = pickle.load(open(r"C:\\Users\\ASUS\\Desktop\\SMS Spam Classifier\\vectorizer.pkl","rb"))


# Streamlit UI
st.title("Email / SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):

    if input_sms.strip() == "":
        st.warning("Please enter a message.")
    else:
        # 1. preprocess
        transformed_sms = transform_text(input_sms)

        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])

        # 3. predict
        result = model.predict(vector_input)[0]

        # 4. Display
        if result == 1:
            st.error(" Spam Message")
        else:
            st.success("Not Spam")
