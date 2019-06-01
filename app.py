from flask import Flask
from flask import request
from flask import redirect
from flask import render_template
from flask import Response
from flask import url_for
from werkzeug import secure_filename

# init the flask web app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

# load the fast.ai model
from fastai.vision import *
defaults.device = torch.device('cpu')
learn = load_learner('.')

# init globals
filename = 'example.jpg'
pred_class = ''

# routes
@app.route("/", methods=["GET"])
def home():
    imagefilename = url_for('static', filename=filename)
    return render_template('index.html', user_image=imagefilename, prediction=pred_class)

@app.route('/predict', methods=['GET', "POST"])
def predict():
    global pred_class
    print("button pressed, predicting...")
    img = open_image(Path('static/'+filename).absolute())
    pred_class,pred_idx,outputs = learn.predict(img)
    print("predicted", pred_class)
    print(outputs)
    imagefilename = url_for('static', filename=filename)
    return redirect('/')
 
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    f = request.files['file']
    uploaded_filename = secure_filename(f.filename)
    f.save('static/'+uploaded_filename)
    global filename, pred_class
    filename=uploaded_filename
    pred_class = ''
    return redirect('/')


if __name__ == "__main__":
    app.run("0.0.0.0", debug=True)

