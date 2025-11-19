# SmartDx — AI-Powered Malaria Microscopy Detection

smartDx is an AI-based diagnostic tool designed to assist laboratory technicians and clinicians in detecting malaria parasites from microscopic blood smear images.  
It uses a trained convolutional neural network (CNN) to classify images as **Parasitized** or **Uninfected**, providing a fast, low-cost screening method suitable for low-resource settings.

---

## 🚀 Features

- Upload a microscopic blood smear image and receive an instant prediction.
- Real-time malaria parasite detection using a CNN model.
- Clean and simple Streamlit web interface.
- Works on desktops and mobile browsers.
- Expandable to support additional pathogens in the future.

---

## 🛠 Tech Stack

- **Python 3.x**
- **TensorFlow / Keras**
- **Streamlit** (for web UI)
- **OpenCV** (basic image handling)
- **NumPy**, **Matplotlib**

---

## 📦 Installation

### 1. Clone the repository

git clone https://github.com/stam100/SmartDx_project.git
cd SmartDx_project

### 2. Create and activate a virtual environment

python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

### 3. Install dependencies

pip install -r requirements.txt

## Running the App

In the project folder, run:

streamlit run app.py
The app will open in your browser on:

http://localhost:8501


## 🤖 Model Details
The CNN model was trained on the NIH Malaria Dataset, consisting of:

Parasitized cell images

Uninfected cell images

Training involved an 80/10/10 split (train/validation/test), normalization, and 10 epochs of training.

The final saved model is located in:

"/models/malaria_model.keras"
and is loaded inside the Streamlit app using:

MODEL_PATH = os.path.join("models", "malaria_model.keras")

## 📚 Dataset Citation
This project uses the publicly available NIH malaria dataset:

Dataset:
"Malaria Cell Images Dataset"
National Institutes of Health (NIH).

### Citation in publications:
S. K. Laughlin, D. Dembele, and C. Quinn. (2018). Malaria Cell Images Dataset. 
Available at: https://lhncbc.nlm.nih.gov/publication/pub9932


## 📂 Folder Structure

SmartDx_project/
│
├── streamlit_app.py
├── utils.py
├── README.md
├── requirements.txt
│
├── models/
    └── malaria_model.keras


## 🔮 Future Improvements
1. Support for P. vivax, P. malariae, and other species.

2. Smartphone microscope integration.

3. Auto-capture microscope imaging.

4. On-device inference using TensorFlow Lite.

5. Dashboard for lab technicians.

6. Integration with hospital EMRs.

## 📄 License
This project is open-source under the MIT License.

## 🙌 Acknowledgements
Special thanks to the NIH for providing the malaria dataset and the open-source community for the tools that made this project possible.
