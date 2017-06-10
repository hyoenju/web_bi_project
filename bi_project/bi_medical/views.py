from django.http import HttpResponse
from django.shortcuts import render
from predict import predict_data
from django.views.decorators.csrf import csrf_exempt
from preprocessing import preprocessing_input_data
import zipfile
import os
import shutil


def home(request):
    return render(request, "home.html")

def predict(request):

    predict_result = predict_data()
    print(predict_result)
    return HttpResponse(predict_result)


UPLOAD_DIR = "./"

@csrf_exempt
def upload(request):

    UPLOAD_DIR = './'
    INPUT_DIR = './image/'

    if request.method == 'POST':
        if 'input-file-preview' in request.FILES:

            file = request.FILES['input-file-preview']
            filename = file._name
            fp = open('%s/%s' % (UPLOAD_DIR, filename), 'wb')
            for chunk in file.chunks():
                fp.write(chunk)
            fp.close()

        try:
            jungle_zip = zipfile.ZipFile(filename)
            jungle_zip.extractall(INPUT_DIR)
            jungle_zip.close()

        finally:
            os.remove(filename)


        folder_name = os.listdir(INPUT_DIR)[0]

        shutil.move(os.path.join(INPUT_DIR, folder_name), os.path.join(INPUT_DIR, "patient"))

        preprocessing_input_data()

        predict_result = predict_data()

        os.remove('patient.npy')
        shutil.rmtree('./image/patient')

        return HttpResponse(predict_result)