from flask import Flask, render_template , request
import pandas as pd



def getListOfKeyWord(keyword):
    df=pd.read_excel('finaltrue.xlsx')
    corpus=[]
    for i in range(len(df)):
        if keyword in df["text"][i]:
            corpus.append(df["text"][i])
    return corpus
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('advance_search.html')

@app.route('/advance_search',methods=['POST'])
def advance_search():
    if request.method=='POST':
        word=""
        word = request.form['kword']
        result=[]
        result=getListOfKeyWord(word)
        return render_template('advance_search.html',key=result)
    else:
        return render_template('advance_search.html',key="Something went wrong!")
    

if __name__ == '__main__':
    app.run(debug=True)

