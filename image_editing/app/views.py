"""
Definition of views.
"""
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, logout


from datetime import datetime
from django.shortcuts import render,redirect
from django.http import HttpRequest
from django.core.files.storage import default_storage
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import matplotlib.pyplot as plt
import cv2
from django.conf import settings
from django.conf.urls.static import static
import os

from django.shortcuts import render
from django.http import HttpResponse

from pickle import load
from numpy import argmax
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow as tf


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
tokenizer = load(open('./image_captioning/tokenizer.pkl', 'rb'))
model = load_model('./image_captioning/model_19.h5')

from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from .forms import UserRegisterForm
from django.core.mail import send_mail
from django.core.mail import EmailMultiAlternatives
from django.template.loader import get_template
from django.template import Context

def home(request):
    """Renders the home page."""
    assert isinstance(request, HttpRequest)
    img_colorization = 'media/static/image_colorization.jfif'
    img_restoration = 'media/static/photo_restoration.jpg'
    img_bg_remove = 'media/static/photo_remove_background.jpg'
    print(img_colorization)
    context = {'image_colorization' : img_colorization,'img_restoration' : img_restoration,'img_bg_remove' : img_bg_remove}
    return render(
        request,
        'app/index.html',
        context
    )

def upload_photo(request):
    assert isinstance(request, HttpRequest)
    if request.method=='POST':
        f=request.FILES['image']
        response = {}
        file_name = "pic.jpg"
        file_name_2 = default_storage.save(file_name, f)
        file_url = default_storage.url(file_name_2)
        image = file_url


        prototxt = "./image_colorization_model_training/model/colorization_deploy_v2.prototxt"
        model = "./image_colorization_model_training/model/colorization_release_v2.caffemodel"
        points = "./image_colorization_model_training/model/pts_in_hull.npy"
        net = cv2.dnn.readNetFromCaffe(prototxt, model)
        pts = np.load(points)
        class8 = net.getLayerId("class8_ab")
        conv8 = net.getLayerId("conv8_313_rh")
        pts = pts.transpose().reshape(2, 313, 1, 1)
        net.getLayer(class8).blobs = [pts.astype("float32")]
        net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        scaled = image.astype("float32") / 255
        lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50

        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

        L = cv2.split(lab)[0]
        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
        colorized = np.clip(colorized, 0, 1)
        colorized = (colorized*255).astype("uint8")
        cv2.imwrite(file_url, cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR))
        response['output'] = file_url
        print(file_url)
        print(response['output'])
        return render(
            request,
            'app/Page-1.html',
            context=response
        )
    else:
        return render(
            request,
            'app/Page-1.html',
            {
                'title':'upload_photo',
                'message':'Upload photo page',
                'year':datetime.now().year,
            }
        )

   
max_length = 32
tokenizer = load(open("./image_captioning/new/tokenizer.p","rb"))
model = load_model('./image_captioning/new/models/model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")


def image_captioning(request):
    assert isinstance(request, HttpRequest)
    if request.method=='POST':
        f=request.FILES['image']
        response = {}
        file_name = "pic.jpg"
        file_name_2 = default_storage.save(file_name, f)
        file_url = default_storage.url(file_name_2)
        img_path = file_url
        photo = extract_features(img_path, xception_model)
        description = generate_desc(model, tokenizer, photo, max_length)
        response['desc'] = description[6:-3]
        response['output'] = file_url
        return render(
            request,
            'app/image_captioning.html',
            context=response
        )
    else:
        return render(
            request,
            'app/image_captioning.html',
            
        )

def extract_features(filename, model):
        try:
            image = Image.open(filename)
            
        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature

def word_for_id(integer, tokenizer):
 for word, index in tokenizer.word_index.items():
     if index == integer:
         return word
 return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text


def Login(request):
    if request.method == 'POST':
   
        # AuthenticationForm_can_also_be_used__
   
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username = username, password = password)
        if user is not None:
            form = login(request, user)
            messages.success(request, f' wecome {username} !!')
            return redirect('home')
        else:
            messages.info(request, f'account done not exit plz sign in')
    form = AuthenticationForm()
    return render(request, 'app/login.html', {'form':form, 'title':'log in'})


def register(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            email = form.cleaned_data.get('email')
            
            return redirect('login')
    else:
        form = UserRegisterForm()
    return render(request, 'app/Signup.html', {'form': form, 'title':'reqister here'})



def bg_remove(request):
    assert isinstance(request, HttpRequest)
    if request.method=='POST':
        f=request.FILES['image']
        response = {}
        file_name = "pic.jpg"
        file_name_2 = default_storage.save(file_name, f)
        file_url = default_storage.url(file_name_2)
        image = file_url
        import numpy as np
        deeplab_model = load_model()
        foreground, bin_mask = remove_background(deeplab_model, '/content/profile.jpg')
        response['foreground'] = file_url

        return render(
            request,
            'app/bg_remove.html',
            context=response
        )
    else:
        return render(
            request,
            'app/bg_remove.html',
            {
                'title':'upload_photo',
                'message':'Upload photo page',
                'year':datetime.now().year,
            }
        )
