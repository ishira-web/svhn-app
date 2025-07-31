import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

class SVHN_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SVHN_CNN, self).__init__()
        # Convolutional Block 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: 64x16x16
        )
        # Convolutional Block 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: 128x8x8
        )
        # Convolutional Block 3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: 256x4x4
        )
        # Fully Connected Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x

# --- Step 2: Load the Trained Model ---

@st.cache_resource
def load_model():
    """Loads the trained PyTorch model."""
    model = SVHN_CNN(num_classes=10)
    # Load the saved weights. Ensure the .pth file is in the same directory.
    # map_location='cpu' allows the model to run on machines without a GPU.
    model.load_state_dict(torch.load('svhn_model_final.pth', map_location=torch.device('cpu')))
    model.eval() # Set the model to evaluation mode
    return model

model = load_model()

# --- Step 3: Define the Image Transformation Pipeline ---
# This must be the same normalization as used for the validation/test data.
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
])

# --- Step 4: Create the Streamlit User Interface ---
st.set_page_config(page_title="SVHN Classifier", page_icon="🔢")
st.title("🔢 Street View House Number Classifier")
st.write(
    "This app uses a Convolutional Neural Network (CNN) to classify images of "
    "house numbers. Upload an image of a single digit to see the prediction."
)

# File uploader allows user to add their own image
uploaded_file = st.file_uploader("Choose an image of a digit...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file).convert('RGB')

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # --- Step 5: Preprocess Image and Make Prediction ---
    # Preprocess the image and add a batch dimension
    img_tensor = transform(image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(img_tensor)
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(output, dim=1)
        # Get the top prediction
        confidence, predicted_class = torch.max(probabilities, 1)
        predicted_digit = predicted_class.item()

    # Display the prediction result
    st.success(f"**Predicted Digit:** `{predicted_digit}`")
    st.info(f"**Confidence:** `{confidence.item():.2%}`")
