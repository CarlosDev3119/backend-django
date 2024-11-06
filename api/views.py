# Import de librerias
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
import asyncio  # Import para procesamiento asincrónico
import json
from asgiref.sync import sync_to_async

from .models import Register,Student

classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
prototxtPath = r"deploy.prototxt"
weightsPath = r"res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
emotionModel = load_model("modelTuning.keras")

# Función de predicción
def predict_emotion(frame, faceNet, emotionModel):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []
    
    for i in range(0, detections.shape[2]):
        if detections[0, 0, i, 2] > 0.4:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (Xi, Yi, Xf, Yf) = box.astype("int")
            if Xi < 0: Xi = 0
            if Yi < 0: Yi = 0
            
            face = frame[Yi:Yf, Xi:Xf]
            if face.size == 0:
                continue
            
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (299, 299))
            face2 = img_to_array(face)
            face2 = np.expand_dims(face2, axis=0)

            faces.append(face2)
            locs.append((Xi, Yi, Xf, Yf))
            pred = emotionModel.predict(face2)
            preds.append(pred[0])

    return (locs, preds)

# Procesar cada imagen de forma secuencial
async def process_image(student, image_base64):
    image_data = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_data))
    image = np.array(image)

    # Convierte la imagen al formato RGB si es necesario
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    (locs, preds) = predict_emotion(image, faceNet, emotionModel)

    response = []
    for (box, pred) in zip(locs, preds):
        label = classes[np.argmax(pred)]
        accuracy = float(np.max(pred) * 100)
        response.append({
            'emotion': label,
            'accuracy': accuracy,
            'box': {
                'Xi': int(box[0].item()),  
                'Yi': int(box[1].item()),  
                'Xf': int(box[2].item()), 
                'Yf': int(box[3].item())
            }
        })

  
    if response:
        main_result = response[0]  
        
        
        emotionResp = main_result['emotion']
        accuracyResp = main_result['accuracy']
        register = Register(
            student =student,
            emotion_type = emotionResp,
            accuracy = accuracyResp
        )

        await sync_to_async(register.save)()
        
        # Aquí podrías guardar `main_result` en la base de datos
        print(emotionResp)
        return main_result

    
    return {'status': 'No face detected'}

@csrf_exempt
def emotion_detection(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            item_id = data.get('id')
            images = data.get('images')
            student = Student.objects.get(pk=int(item_id))
            
            
            if not item_id or not images:
                return JsonResponse({'status': 'error', 'message': 'ID or images not provided'})

            responses = []
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            for image_base64 in images:
                response = loop.run_until_complete(process_image(student, image_base64))
                responses.append(response)

            loop.close()
            return JsonResponse({'status': 'success', 'predictions': responses})

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})

    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})
