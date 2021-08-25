import os
import cv2
import torch
from torch.nn import functional as F
import warnings
#from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request, send_from_directory
import subprocess
import json
import numpy as np
import pandas as pd
from PIL import Image
import base64
import io

app = Flask(__name__)
app.config.from_file("config.json",load=json.load)
app.app_context().push()
#UPLOAD_FOLDER = os.path.join('static','eg')
#TEMP_FOLDER = os.path.join('static')

#app = Flask(__name__)
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#app.config['TEMP_FOLDER'] = TEMP_FOLDER
#run_with_ngrok(app)

@app.route('/')

def index():
    mimg = os.path.join(app.config['UPLOAD_FOLDER'],'1.png')
    return render_template("index.html", msg="hey", my_image= mimg)


@app.route("/prediction", methods=["POST"])
def prediction():
    i1 = request.files['img1']

    i1.save(os.path.join('static','I0_0.png'))
  
    i2 = request.files['img2']

    # i2.save(os.path.join('static','I0_1.png'))
    # cmd1 =['python3', 'inference_img.py', '--img', 'static/I0_0.png', 'static/I0_1.png']
    # #make1 = subprocess.call("python3 inference_img.py --img I0_0.png I0_1.png",shell=True)
    # make1=subprocess.Popen(cmd1).wait()

    warnings.filterwarnings("ignore")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    # parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
    # parser.add_argument('--img', dest='img', nargs=2, required=True)
    # parser.add_argument('--exp', default=4, type=int)
    # parser.add_argument('--ratio', default=0, type=float, help='inference ratio between two images with 0 - 1 range')
    # parser.add_argument('--rthreshold', default=0.02, type=float, help='returns image when actual ratio falls in given range threshold')
    # parser.add_argument('--rmaxcycles', default=8, type=int, help='limit max number of bisectional cycles')
    # parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')

    # args = parser.parse_args()

    # #args.img= []

    img=['static/I0_0.png', 'static/I0_1.png']
    exp = 4
    ratio = 0
    rthreshold = 0.02
    rmaxcycles = 8
    modelDir = '/app/model'#os.path.join(app.config['MODEL_FOLDER'])


    try:
        try:
            from model.RIFE_HDv2 import Model
            model = Model()
            model.load_model(modelDir, -1)
            print("Loaded v2.x HD model.")
        except:
            from model.RIFE_HDv3 import Model
            model = Model()
            model.load_model(modelDir, -1)
            print("Loaded v3.x HD model.")
    except:
        from model.RIFE_HD import Model
        model = Model()
        model.load_model(modelDir, -1)
        print("Loaded v1.x HD model")
    model.eval()
    model.device()
    print("lansknfa hahaha")
    if img[0].endswith('.exr') and img[1].endswith('.exr'):
        img0 = cv2.imread(img[0], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
        img1 = cv2.imread(img[1], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
        img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device)).unsqueeze(0)
        img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device)).unsqueeze(0)

    else:
        img0 = cv2.imread(img[0], cv2.IMREAD_UNCHANGED)
        img1 = cv2.imread(img[1], cv2.IMREAD_UNCHANGED)
        img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

    n, c, h, w = img0.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    img0 = F.pad(img0, padding)
    img1 = F.pad(img1, padding)


    if ratio:
        img_list = [img0]
        img0_ratio = 0.0
        img1_ratio = 1.0
        if ratio <= img0_ratio + rthreshold / 2:
            middle = img0
        elif ratio >= img1_ratio - rthreshold / 2:
            middle = img1
        else:
            tmp_img0 = img0
            tmp_img1 = img1
            for inference_cycle in range(rmaxcycles):
                middle = model.inference(tmp_img0, tmp_img1)
                middle_ratio = ( img0_ratio + img1_ratio ) / 2
                if ratio - (rthreshold / 2) <= middle_ratio <= ratio + (rthreshold / 2):
                    break
                if ratio > middle_ratio:
                    tmp_img0 = middle
                    img0_ratio = middle_ratio
                else:
                    tmp_img1 = middle
                    img1_ratio = middle_ratio
        img_list.append(middle)
        img_list.append(img1)
    else:
        img_list = [img0, img1]
        for i in range(exp):
            tmp = []
            for j in range(len(img_list) - 1):
                mid = model.inference(img_list[j], img_list[j + 1])
                tmp.append(img_list[j])
                tmp.append(mid)
            tmp.append(img1)
            img_list = tmp

    print("next top ")
    #if not os.path.exists('{TEMP_FOLDER}'):
        #os.mkdir('{TEMP_FOLDER}')
    for i in range(len(img_list)):
        if img[0].endswith('.exr') and img[1].endswith('.exr'):
            cv2.imwrite('static/img{}.exr'.format(i), (img_list[i][0]).cpu().numpy().transpose(1, 2, 0)[:h, :w], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
            print("next2")
        else:
            cv2.imwrite('static/img{}.png'.format(i), (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
            print("next ")
  
    frames = subprocess.call("ffmpeg -y -r 10 -f image2 -i static/img%d.png -s 448x256 -vf \"split[s0][s1];[s0]palettegen=stats_mode=single[p];[s1][p]paletteuse=new=1\" static/eg/slomo.gif",shell=True)
  
    gifval = os.path.join(app.config['UPLOAD_FOLDER'],'slomo.gif')
  
    return render_template("prediction.html",user_image=gifval)
if __name__ == "__main__":
    app.run(debug=True)



