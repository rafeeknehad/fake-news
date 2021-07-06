from imports import *
# Split dataset into training set and test set
#list_data,list_label = embedding.doc2vec_Fun()
# list_data_embedded=embedding.get_data2();
# list_labels=embedding.get_data();
# X_train, X_test, y_train, y_test = train_test_split(list_data_embedded,list_labels, test_size=0.20)
# #Create a svm Classifier
# clf = svm.SVC(kernel='linear') # Linear Kernel
# #Train the model using the training sets
# clf.fit(X_train, y_train)
# #Predict the response for test dataset
# y_pred = clf.predict(X_test)
# # Model Accuracy: how often is the classifier correct?
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# #predict the value from the user input 
# # save the model to disk
# #test the input of the user 
def Train_DATA():
    list_data_embedded=embedding.get_data2();
    list_labels=embedding.get_data();
    X_train, X_test, y_train, y_test = train_test_split(list_data_embedded,list_labels, test_size=0.20)
    #Create a svm Classifier
    clf = svm.SVC(kernel='linear') # Linear Kernel
    #Train the model using the training sets
    clf.fit(X_train, y_train)
    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    #predict the value from the user input 
    # save the model to disk
    #test the input of the user
    return clf 



def PRE(clf):
    doc =""" 
   "Two separate reports cited a senior intelligence official as confirming that most of the 17 US intelligence agencies believe that the coronavirus originated in the Wuhan Institute of virology.
   
The Daily Caller spoke to one senior official, who anonymously related that “the majority view among the US intelligence community agencies is that COVID-19 is natural and accidentally leaked out of a laboratory in Wuhan.”

The official told the news site that not all agencies are “fully behind the idea” that the spread was due to an accidental laboratory leak, but “most believe that to be the case.”

The official also stated that the intelligence agencies are unanimously in agreement that the outbreak “was not the result of an intentional act.”

The report dovetails with details contained in a Fox News investigation, which also cited a senior official as saying that most of the intel community is decided on the lab leak theory.

White House reporter John Roberts reported Saturday “there is agreement among most of the 17 Intelligence agencies that COVID-19 originated in the Wuhan lab. The source stressed that the release is believed to be a MISTAKE, and was not intentional.”  The agencies that have yet to commit to the theory as confirmed are awaiting a more solid “smoking gun,” according to the official.

 The findings come after a ‘five eyes’ report was leaked, detailing how the intelligence agencies of the US, UK, Australia, New Zealand and Canada all believe China engaged in a cover up of the real severity of the outbreak.

The document, obtained by Australia’s Saturday Telegraph newspaper,  finds that China’s secrecy amounted to an “assault on international transparency.”

The confirmation from the senior US official contradicts a report in the New York Times last week that suggested the White House has unduly pressured intelligence agencies to link the Wuhan bio-lab to the coronavirus outbreak.

The story was repeatedly cited by mainstream sources and leftist websites, to suggest that US intelligence agencies have all but dismissed the notion, when that clearly appears to be the opposite of what is happening.

Prominent scientists are on record dating back years, with warnings that Chinese virus research is unsafe, with many saying that the Wuhan institute has to be considered the number one suspect as the origin for the current global pandemic."
"""
    doc=p.text_preprocessing(doc)
    # Tokenization of each document
    tokenized_doc = []
    for d in doc:
        tokenized_doc.append(word_tokenize(d.lower()))
    #print(tokenized_doc)
    # Convert tokenized document into gensim formated tagged data
    tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_doc)]
    #print(tagged_data)
    ## Train doc2vec model
    model = Doc2Vec(tagged_data, vector_size=100, window=3, min_count=2, workers=5, epochs = 50)
    # Save trained doc2vec model
    model.save("test_doc2vec.model")
    ## Load saved doc2vec model
    model= Doc2Vec.load("test_doc2vec.model")
    output = clf.predict([model.dv[0]])
    print(output)
    return output

def COMBO():
    res1 = Train_DATA()
    res2 = PRE(res1)
    return res2
#######################FLASK#############################################################
pickle.dump(COMBO(),open('model.pkl','wb'))

app=Flask(__name__)
mod = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])

def predict():
    if request.method == 'POST':
        #message = request.form['message']
        pred = COMBO()
        print(pred)
        return render_template('home.html',info=pred)
    else: 
        return render_template('home.html',info="Something went wrong!")

if __name__ == '__main__':
    app.run(debug=True)
