#from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request, send_from_directory
import subprocess
import os
import numpy as np
import pandas as pd
from PIL import Image
import base64
import io



app = Flask(__name__)
PEOPLE_FOLDER = os.path.join('static','eg')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
#run_with_ngrok(app)

@app.route('/')
def index():
	mimg = os.path.join(app.config['UPLOAD_FOLDER'],'1.png')
	return render_template("index.html", msg="hey", my_image= mimg)


@app.route("/prediction", methods=["POST"])
def prediction():
  img1 = request.files['img1']

  img1.save("I0_0.png")
  img2 = request.files['img2']

  img2.save("I0_1.png")
  cmd1 =['python3', 'inference_img.py', '--img', 'I0_0.png', 'I0_1.png']
  #make1 = subprocess.call("python3 inference_img.py --img I0_0.png I0_1.png",shell=True)
  make1=subprocess.Popen(cmd1).wait()
  
  make2 = subprocess.call("ffmpeg -y -r 10 -f image2 -i output/img%d.png -s 448x256 -vf \"split[s0][s1];[s0]palettegen=stats_mode=single[p];[s1][p]paletteuse=new=1\" static/eg/slomo.gif",shell=True)

  
  gifval = os.path.join(app.config['UPLOAD_FOLDER'],'slomo.gif')
  
  return render_template("prediction.html",user_image=gifval,msg=make2)
 
if __name__ == "__main__":
	app.run(debug=True)
