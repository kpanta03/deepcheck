# audio_detection/views.py

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
import os
from .utils import predict_deepfake

@csrf_exempt
def detect_audio(request):
    if request.method == "POST" and request.FILES.get('audio_file'):
        audio_file = request.FILES['audio_file']
        fs = FileSystemStorage()
        file_path = fs.save(audio_file.name, audio_file)
        file_path = fs.url(file_path)

        # Full file path on the server
        full_file_path = os.path.join(os.getcwd(), file_path[1:])

        # Run the prediction
        result = predict_deepfake(full_file_path)

        if result is not None:
            return JsonResponse({"status": "success", "result": result})
        else:
            return JsonResponse({"status": "error", "message": "Error in audio detection"})
    else:
        return JsonResponse({"status": "error", "message": "No file uploaded"})

