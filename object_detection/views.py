from django.shortcuts import render
from .utils import detect_objects_in_image

def index(request):
    return render(request, 'index.html')

def detect_objects(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        processed_image = detect_objects_in_image(image)
        return render(request, 'result.html', {'processed_image': processed_image})
    else:
        return render(request, 'index.html', {'error_message': 'Please upload an image.'})
