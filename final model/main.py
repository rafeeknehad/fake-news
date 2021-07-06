from imports import *

app=Flask(__name__)
#train_CNN() #just to train the model once
#train_LSTM()
#model3 = pickle.load(open('model3.pkl','rb'))
loaded_model = keras.models.load_model('tokLSTM.h5')
with open('tokenizer.pickle', 'rb') as handle:
    loaded_tk = pickle.load(handle)
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])

def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = PRE_LSTM(loaded_model,loaded_tk,message)
        print(pred)
        return render_template('home.html',info=pred)
    else: 
        return render_template('home.html',info="Something went wrong!")

@app.route('/download',methods=['POST'])
def download():
    if request.method=='POST':
        getListOfTruth()
        return render_template('home.html')
    else: 
        return render_template('home.html',info="Something went wrong!")

@app.route('/advance_search',methods=['GET','POST'])

def advance_search():

    if request.method=='POST':
        word = request.form['kword']
        result=getListOfKeyWord(word)
        table=listing(result)
        return render_template('advance_search.html',key=table)
    return render_template('advance_search.html')

   
        

if __name__ == '__main__':
    app.run(debug=True)
