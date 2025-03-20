# LayoutLMv3 Document Classification

This project trains a LayoutLMv3 model for document classification using Hugging Face's Transformers and PyTorch.

## Project Structure

```
.
├── data/
│   ├── train/
│   │   ├── class_1/
│   │   ├── class_2/
│   │   ├── ...
│   ├── test/
│   │   ├── class_1/
│   │   ├── class_2/
│   │   ├── ...
│
├── saved_model/
├── results/
├── logs/
├── predictions.csv
├── classification_report.txt
├── train.py
├── README.md
└── requirements.txt
```

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/Laxparihar/LayoutLMV3_Fine_Tuning.git
   cd LayoutLM_V3
   ```
2. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Install detectron2
   ```sh
   pip install --upgrade pip setuptools wheel ; pip install cython
   ```
   ```sh
   pip install layoutparser torchvision && pip install "detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2"
   ```
## Dataset Structure

The dataset should be structured as follows:
```
data/
├── train/
│   ├── class_1/ (contains images for class 1)
│   ├── class_2/ (contains images for class 2)
│   ├── ...
│
├── test/
│   ├── class_1/ (contains images for class 1)
│   ├── class_2/ (contains images for class 2)
│   ├── ...
```

## Training the Model

Run the training script:
```sh
python train.py
```

This will:
- Load and preprocess the dataset.
- Train the LayoutLMv3 model.
- Save the trained model to `saved_model/`.
- Generate predictions and a classification report.

## Running Inference on Test Data

After training, run inference:
```sh
python train.py --inference
```
This will generate `predictions.csv` and `classification_report.txt`.

## Model Evaluation

- **Accuracy** and **Classification Report** are printed in the console and saved to files.
- The best model (based on validation loss) is saved automatically.

## Saving and Loading the Model

The trained model is saved in `saved_model/`. To load it later:
```python
from transformers import LayoutLMv3ForSequenceClassification
import torch

model = LayoutLMv3ForSequenceClassification.from_pretrained("saved_model/")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- Hugging Face Transformers
- PyTorch
- scikit-learn
- TQDM

---
Feel free to modify this `README.md` as needed!

