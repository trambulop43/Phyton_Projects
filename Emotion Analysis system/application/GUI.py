
from tkinter import *
import numpy as np
import pandas as pd
import tkinter as tk, os, random
import pickle
from sklearn import preprocessing
from nltk.corpus import stopwords
from textblob import Word
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

lbl_enc = preprocessing.LabelEncoder()
data = pd.read_csv('text_emotion.csv')
data = data.drop('author', axis=1)
data = data.drop(data[data.sentiment == 'anger'].index)
data = data.drop(data[data.sentiment == 'boredom'].index)
data = data.drop(data[data.sentiment == 'enthusiasm'].index)
data = data.drop(data[data.sentiment == 'empty'].index)
data = data.drop(data[data.sentiment == 'fun'].index)
data = data.drop(data[data.sentiment == 'relief'].index)
data = data.drop(data[data.sentiment == 'hate'].index)
y = lbl_enc.fit_transform(data.sentiment.values)

def de_repeat(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)
data['content'] = data['content'].apply(lambda x: " ".join(x.lower() for x in x.split()))
data['content'] = data['content'].str.replace('[^\w\s]',' ')
stop = stopwords.words('english')
data['content'] = data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
data['content'] = data['content'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
data['content'] = data['content'].apply(lambda x: " ".join(de_repeat(x) for x in x.split()))
freq = pd.Series(' '.join(data['content']).split()).value_counts()[-10000:]
freq = list(freq.index)
data['content'] = data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
X_train, X_val, y_train, y_val = train_test_split(data.content.values, y, stratify=y, random_state=42, test_size=0.3, shuffle=True)

count_vect = CountVectorizer(analyzer='word')
count_vect.fit(data['content'])
X_train_count =  count_vect.transform(X_train)
X_val_count =  count_vect.transform(X_val)

root = Tk()
topFrame = Frame(root)
topFrame.pack()
bottomFrame = Frame(root)
bottomFrame.pack(side = BOTTOM)
root.title("Emotion Identifier")
#root.tk.call('wm', 'iconphoto', root._w, tk.PhotoImage(file="EASTT.jpg"))
root.geometry("500x450")

##########################################################################################
def submit():
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    filename1 = 'countV_model.sav'
    loaded_model1 = pickle.load(open(filename1, 'rb'))
    X_train_count =  loaded_model1.transform(X_train)
    X_val_count =  loaded_model1.transform(X_val)
    
    sentence = []
    sentence = Input.get("1.0",END)
    test = pd.DataFrame([sentence])
    test[0] = test[0].apply(lambda x: " ".join(x.lower() for x in x.split()))
    test[0] = test[0].str.replace('[^\w\s]',' ')
    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    test[0] = test[0].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    from textblob import Word
    test[0] = test[0].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    test[0] = test[0].apply(lambda x: " ".join(de_repeat(x) for x in x.split()))
    freq1 = pd.Series(' '.join(test[0]).split()).value_counts()[-10000:]
    freq1 = list(freq1.index)
    test[0] = test[0].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
        
    test1 = loaded_model1.transform(test[0])
    se = loaded_model.predict(test1)
    lse = lbl_enc.inverse_transform(se)
    label2.config(text=lse)
##########################################################################################

##########################################################################################
def clear():
    Input.delete(1.0,END)
    label2.config(text="Emotion will be identified here")
##########################################################################################
  
Label1= Label(root, text="Enter a Text below", font = ("Helvetica",20,"bold"))
Input = Text(root,height = 50,width = 400,font = ("Helvetica",16),background="lemon chiffon")
button_frame = Frame(root)
Button1 = Button(button_frame, text= "analyze emotion", command = submit)
Button1.grid(row = 0,column = 0)
Button2 = Button(button_frame, text= "Clear", command = clear)
Button2.grid(row = 0,column = 1,padx = 20)

label2= Label(root,text="Emotion will be identified here",font = ("Helvetica",16))



button_frame.pack()
label2.pack()
Label1.pack()
Input.pack(pady = 20)


root.mainloop()