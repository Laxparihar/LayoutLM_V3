import os
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import (
    LayoutLMv3FeatureExtractor, LayoutLMv3Tokenizer, LayoutLMv3Processor,
    LayoutLMv3ForSequenceClassification, TrainingArguments, Trainer
)
from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array2D, Array3D
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image

Image.warnings.simplefilter('ignore', Image.DecompressionBombWarning)


import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
 
# ----------------------------------------------------------------------------------------------------
# Paths to the training and testing datasets
train_dataset_path = "data/train"
test_dataset_path = "data/test"

labels = sorted(os.listdir(train_dataset_path))  # Sorting ensures consistent label order
idx2label = {v: k for v, k in enumerate(labels)}
label2idx = {k: v for v, k in enumerate(labels)}

# ----------------------------------------------------------------------------------------------------
# Load training data
train_images, train_labels = [], []
for label in labels:
    image_files = os.listdir(f"{train_dataset_path}/{label}")
    train_images.extend([f"{train_dataset_path}/{label}/{img}" for img in image_files])
    train_labels.extend([label2idx[label]] * len(image_files))  # Convert labels to integers

train_data = pd.DataFrame({'image_path': train_images, 'label': train_labels})

# Load testing data
test_images, test_labels = [], []
for label in labels:
    image_files = os.listdir(f"{test_dataset_path}/{label}")
    test_images.extend([f"{test_dataset_path}/{label}/{img}" for img in image_files])
    test_labels.extend([label2idx[label]] * len(image_files))  # Convert labels to integers

test_data = pd.DataFrame({'image_path': test_images, 'label': test_labels})

# ----------------------------------------------------------------------------------------------------
# Manually create a validation set (20% of training data)
validation_data = train_data.sample(frac=0.2, random_state=42)
train_data = train_data.drop(validation_data.index).reset_index(drop=True)
validation_data = validation_data.reset_index(drop=True)

print(f"{len(train_data)} training examples, {len(validation_data)} validation examples")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------------------------------
# Initialize LayoutLMv3 Processor
feature_extractor = LayoutLMv3FeatureExtractor()
tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")
processor = LayoutLMv3Processor(feature_extractor, tokenizer)

# ----------------------------------------------------------------------------------------------------
# Encoding function
def encode_example(examples):
    images = [Image.open(path).convert("RGB") for path in examples['image_path']]
    
    # Ensure proper padding and truncation
    encoded_inputs = processor(
        images,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Convert labels to tensor of `int64`
    encoded_inputs["labels"] = torch.tensor(examples["label"], dtype=torch.long)

    return encoded_inputs

# ----------------------------------------------------------------------------------------------------
# Convert Pandas DataFrame to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_data)
valid_dataset = Dataset.from_pandas(validation_data)

# Apply encoding function (batched=True for efficiency)
train_dataset = train_dataset.map(encode_example, batched=True)
valid_dataset = valid_dataset.map(encode_example, batched=True)

# ----------------------------------------------------------------------------------------------------
# Initialize Model
model = LayoutLMv3ForSequenceClassification.from_pretrained(
    "microsoft/layoutlmv3-base", num_labels=len(label2idx)
)
model.to(device)

# ----------------------------------------------------------------------------------------------------
# TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    save_total_limit=1,
    lr_scheduler_type="linear",
)

# ----------------------------------------------------------------------------------------------------
# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
)
trainer.train()
trainer.save_model("./saved_model/")

# ----------------------------------------------------------------------------------------------------
# Inference on test set
def run_inference(test_data):
    image_paths, predictions, true_labels = [], [], []
    
    model.eval()
    
    for image_path, true_label in tqdm(zip(test_images, test_labels), total=len(test_images), desc="Processing images", unit="image"):
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt", padding="max_length", truncation=True)
        inputs = {key: value for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs.to(model.device))
       
        logits = outputs.logits
        predicted_class_idx = torch.argmax(logits, dim=-1).item()
        predicted_class = idx2label[predicted_class_idx]
        
        image_paths.append(image_path)
        predictions.append(predicted_class)
        true_labels.append(idx2label[true_label])  # Convert int to label

    results_df = pd.DataFrame({
        'image_path': image_paths,
        'true_label': true_labels,
        'predicted_label': predictions
    })
    results_df.to_csv("predictions.csv", index=False)

    accuracy = accuracy_score(true_labels, predictions)
    print(f"Accuracy: {accuracy:.4f}")

    report = classification_report(true_labels, predictions, target_names=list(idx2label.values()))
    print("Classification Report:")
    print(report)

    with open("classification_report.txt", "w") as f:
        f.write(report)

run_inference(test_data)
