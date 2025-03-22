import os
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
from pdf2image import convert_from_path
from PIL import Image
import torch.nn.functional as F
import shutil
import logging
import csv
logging.basicConfig(level=logging.INFO, format="%(asctime)s, %(levelname)s: %(message)s")
import warnings

warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = './model'
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=True)
model = LayoutLMv3ForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()
print()

id2label = {0: 'class_a', 1: 'class_b', 2: 'class_c', 3: 'class_d', 4: 'class_e', 5: 'class_f', 6: 'class_g'}


def layout_V3_prediction(file_path):
    # Convert the PDF to images (first page)
    if file_path.endswith('.pdf'):
        images = convert_from_path(file_path)
        image = images[0]
    else:
        image = Image.open(file_path).convert("RGB")
        
    # Preprocess the image for LayoutLMv3
    encoding = processor(images=image, return_tensors="pt", truncation=True, padding=True)
    encoding = {key: val.to(device) for key, val in encoding.items()}
    
    # Run inference
    with torch.no_grad():
        outputs = model(**encoding)
    
    # Get logits and apply softmax for class probabilities
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
    
    # Get predicted class and confidence
    predicted_class_idx = torch.argmax(probs, dim=-1).item()
    predicted_label = id2label[predicted_class_idx]
    confidence = round(probs[0, predicted_class_idx].item() * 100, 2)

    return predicted_label, confidence
    
    

csv_dir = './output/'
os.makedirs(csv_dir, exist_ok=True)

csv_path = os.path.join(csv_dir, f'prediction.csv')
file_exists = os.path.isfile(csv_path)
if not file_exists:
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["Filename", "Class","Predicted", "Confidence"]
        writer.writerow(header)

root_dir = "../data/"
 
for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        file_name = os.path.basename(file_path)
        try: 
            predicted_class, confidence = layout_V3_prediction(file_path)
     
            predicted_dir = os.path.join(root_dir, predicted_class)
           
            # if not os.path.exists(predicted_dir):
            #     os.mkdir(predicted_dir)
            
            # # Move the PDF to the corresponding predicted directory
            # shutil.move(file_path, predicted_dir)
            # Log the prediction and confidence
            # ---------------
            with open(csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([file_name, folder,predicted_class,confidence])
            logging.info(f'Predicted Class: {predicted_class}, Confidence: {confidence}%')
    
        except Exception as e:
            # If there's an error, log it and move to the next file
            logging.error(f"Error processing file {file}: {str(e)}")
            pass
