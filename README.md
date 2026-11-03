# 🌿 Plant Disease Detection Using Deep Learning

An AI-powered plant disease detection system that uses a **Convolutional Neural Network (CNN)** built with PyTorch to classify plant leaf images into 39 different categories. The project combines deep learning with a Flask web application to help users identify potential plant diseases from leaf images.

## 📌 Project Overview

Plant diseases can affect crop quality and agricultural productivity. This project explores image-based disease classification using a CNN trained on the PlantVillage dataset.

Users can upload a leaf image to the web application and receive a predicted disease category. The application also includes pages for disease information, fertilizer or supplement recommendations, and user contact.

### ✨ Key Features

- **39-class classification:** Predicts among 39 plant and disease categories.
- **CNN-based image recognition:** Uses a custom deep learning architecture implemented in PyTorch.
- **Image upload and prediction:** Analyze leaf images through a Flask web interface.
- **Disease information:** Displays disease-related information alongside predictions.
- **Fertilizer and supplement store:** Provides a section for related product recommendations.
- **Sample testing images:** Includes example images for testing the prediction workflow.
- **Pretrained model support:** Load trained model weights instead of training the network from scratch.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python 3.8+ |
| Deep learning framework | PyTorch |
| Neural network | Convolutional Neural Network (CNN) |
| Image processing | Pillow, Torchvision |
| Data processing | NumPy, Pandas |
| Visualization | Matplotlib |
| Web framework | Flask |
| Production server | Gunicorn |
| Development environment | Jupyter Notebook |

## 🧠 Model Architecture

The project uses a custom CNN consisting of convolutional blocks followed by fully connected layers.

### Architecture Details

- **Input:** RGB leaf image resized and cropped to 224 × 224 pixels.
- **Convolutional layers:** Six convolutional layers organized into three feature-channel groups initially and two deeper feature groups, with channel counts of 32, 64, 128, and 256.
- **Activation:** ReLU.
- **Normalization:** Batch Normalization.
- **Downsampling:** Max Pooling.
- **Regularization:** Dropout with a probability of 0.4.
- **Fully connected layers:** 50,176 input features → 1,024 hidden units → 39 output units.
- **Loss function:** Cross-Entropy Loss.
- **Optimizer:** Adam.

### Model Summary

| Metric | Value |
|---|---:|
| Input dimensions | 3 × 224 × 224 |
| Output classes | 39 |
| Total parameters | 52,595,399 |
| Trainable parameters | 52,595,399 |
| Non-trainable parameters | 0 |
| Batch size used in the notebook | 64 |
| Training epochs shown | 5 |

The notebook also contains code for plotting training and validation losses and evaluating accuracy on the training, validation, and test subsets.

## 📊 Dataset

The model notebook uses the PlantVillage dataset, loaded using PyTorch's `ImageFolder` utility.

- **Dataset size in the notebook:** 61,486 images.
- **Number of categories:** 39.
- **Image preprocessing:** Resize to 255 pixels, center crop to 224 × 224, and convert to a tensor.
- **Data partitioning:** Randomly shuffled indices are used to create training, validation, and test subsets.

### Dataset Links

- [PlantVillage dataset on Mendeley Data](https://data.mendeley.com/datasets/tywbtsjrjv/1)
- [Project tutorial and dataset information](https://medium.com/analytics-vidhya/plant-disease-detection-using-convolutional-neural-networks-and-pytorch-87c00c54c88f)

Refer to the dataset's licensing and usage conditions before redistributing the data.

## 🚀 Run the Project Locally

### Prerequisites

Install the following before running the project:

- Python 3.8 or a compatible Python version
- pip
- Git
- A pretrained model checkpoint
- The project dependencies

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Plant-Disease-Detection
```

Replace `<YOUR_REPOSITORY_URL>` with the actual repository URL.

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

For more information, refer to the [Python virtual environment documentation](https://docs.python.org/3/tutorial/venv.html).

### 3. Install Dependencies

If `requirements.txt` is present in the repository root:

```bash
pip install -r requirements.txt
```

If the project has a separate dependency file inside `Flask Deployed App`, navigate to that directory and install the dependencies listed there.

If no dependency file is available, install the packages required by the application and its model. The exact versions should be taken from the project's working environment.

### 4. Download the Pretrained Model

Download the pretrained model checkpoint from the project's Google Drive folder:

[Download Plant Disease Detection Model](https://drive.google.com/drive/folders/1ewJWAiduGuld_9oGSrTuLumg9y62qS6A?usp=share_link)

Place the downloaded checkpoint in the directory expected by the application.

**Important:** The original setup instructions refer to `plant_disease_model_1.pt`, while the model notebook references `plant_disease_model_1_latest.pt` in its loading example. Use the filename expected by the relevant application or notebook, and do not rename the file unless the corresponding loading code is updated.

### 5. Run the Flask Application

Navigate to the Flask application directory:

```bash
cd "Flask Deployed App"
```

Run the application:

```bash
python app.py
```

Open the local URL printed in the terminal by Flask, commonly `http://127.0.0.1:5000`, if that is the configured address.

The exact startup command and URL depend on the application's configuration.

## 🧪 Testing the Model

The repository includes a `test_images/` directory containing sample leaf images that can be used to test the prediction workflow.

1. Start the Flask application.
2. Open the AI Engine or image prediction page.
3. Upload a sample image from `test_images/`.
4. Submit the image for analysis.
5. Review the predicted plant disease category and any associated information displayed by the application.

Use different samples to evaluate how the model behaves across plant types and disease categories.

**Note:** Correct predictions are not guaranteed for every image. Lighting, image quality, backgrounds, and similarities between disease symptoms can affect classification performance.

## 📓 Explore the Jupyter Notebook

The `Model/` section contains the model documentation and notebook.

To work with the model in Jupyter Notebook:

```bash
pip install notebook
jupyter notebook
```

Open the project notebook and follow its cells in sequence. Ensure that the dataset directory and pretrained checkpoint are available at the paths referenced in the notebook.

The notebook covers:

- Dataset loading and preprocessing
- CNN architecture definition
- Model training with Adam and Cross-Entropy Loss
- Training and validation loss visualization
- Accuracy evaluation
- Loading pretrained model weights
- Single-image prediction

## 📁 Project Structure

The following is a conceptual overview of the main project components. Actual filenames and directories may vary slightly.

```text
Plant-Disease-Detection/
│
├── Flask Deployed App/
│   ├── app.py
│   ├── CNN.py
│   ├── templates/
│   ├── static/
│   └── ...
│
├── Model/
│   ├── Plant Disease Detection Code.ipynb
│   ├── Plant Disease Detection Code.md
│   └── ...
│
├── test_images/
│   └── Sample leaf images
│
├── demo_images/
│   ├── 1.png
│   ├── 2.png
│   ├── 3.png
│   ├── 4.JPG
│   └── 5.png
│
├── requirements.txt
└── Readme.md
```

The model checkpoint may be downloaded separately rather than stored in the repository.

## 🖥️ Web Application Screenshots

The `demo_images/` directory contains screenshots of the application interface.

### 1. Main Page

![Main Page](demo_images/1.png)

### 2. AI Engine

![AI Engine](demo_images/2.png)

### 3. Prediction Results

![Results Page](demo_images/3.png)

### 4. Supplements and Fertilizer Store

![Supplements and Fertilizer Store](demo_images/4.JPG)

### 5. Contact Us

![Contact Us](demo_images/5.png)

If an image does not appear on GitHub, verify that its filename, capitalization, extension, and relative path match the actual file in `demo_images/`.

## 🤝 Contributions

This project is open to contributions that improve its usability, model performance, documentation, and functionality.

Potential areas for improvement include:

- Improving the user interface and accessibility.
- Experimenting with alternative CNN architectures.
- Improving preprocessing and data augmentation.
- Evaluating model performance across individual classes.
- Improving prediction reliability on real-world leaf photographs.
- Enhancing disease information and recommendations.
- Improving application deployment and error handling.
- Updating documentation and adding reproducible experiments.

### Contribution Workflow

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Test the updated functionality.
5. Update relevant documentation and model artifacts when necessary.
6. Commit and push your changes.
7. Open a pull request describing the changes.

For guidance, read [How to Create a Pull Request on GitHub](https://opensource.com/article/19/7/create-pull-request-github).

Please ensure that code changes work correctly and that any new dependencies, model checkpoints, or dataset requirements are documented.

## ⚠️ Limitations

- The model predicts only among its 39 learned categories.
- Predictions depend on the quality and representativeness of the training data.
- The reported evaluation values in the notebook should be independently verified before being presented as final benchmark results.
- The model's predictions are not a substitute for expert agricultural diagnosis.
- Fertilizer or supplement recommendations should be independently validated for the specific crop and disease.

## 🔮 Future Improvements

- Evaluate additional CNN architectures and transfer-learning approaches.
- Improve generalization to field photographs with complex backgrounds.
- Add confidence scores and better handling of uncertain predictions.
- Provide clearer disease descriptions and validated management guidance.
- Expand the dataset to cover more crops, diseases, and real-world conditions.
- Add automated testing and improve deployment reliability.

## 📜 License

Add the repository's applicable open-source license here. Ensure that the license is compatible with the code, pretrained weights, and datasets being redistributed.

## 🙌 Acknowledgments

- The PlantVillage dataset and its contributors.
- The PyTorch and Torchvision communities.
- The Flask community.
- Contributors who improve the project's code, documentation, and user experience.

---

**Plant Disease Detection — Deep learning for image-based plant disease classification.**
