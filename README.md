# ML/CV Learning Process 

My journey to learning how machines learn, interact and upate. Project first learning with progress reports and checks. constantly improving methodology of documentation and coding convention. 

## Note
There will be use of AI in this for the learning process and minor changes such as path change according to file placement. I am not yet fully fluent in everything related to files, git etc so I will be using AI chatbots to help. NOT VIBECODING

## Roadmap
- [x] Numpy and linear algebra fundamentals
- [x] Pandas, scikit-learn, OpenCV, Dlib
- [x] Regression Models + Neural networks theory
- [x] Kaggle Pandas Course
- [x] Kaggle Intro to ML
- [x] Kaggle Intermediate ML
- [ ] Gesture recognition
- [ ] Auth control using gestures
- [ ] Andrew Ng ML Specialization
- [ ] Andrew Ng Deep Learning Specialization

## Progress Log
| Project | Type | Dataset | Status | Date |
|---|---|---|---|---|
| Prediction Models | Classification + Regression | Kaggle | Done | Feb-March 2026
| Face Detection | CV | Webcam | Done | April 2026|
| Face Recognition | CV | Webcam && Video | Done | April 2026 |
| Object Tracking | CV | Webcam && Video | Done | May 2026 |
| Gesture Admin Control | CV + Auth | Custom/Webcam | In progress | TBD| 

## Projects

###  Computer Vision (Main Focus)
- Live webcam face detection
- Face detection and recognition
- Image processing fundamentals
- Object detection and tracking

###  Data Visualization
Exploratory analysis and plotting across multiple real-world datasets using matplotlib:
- Insurance, Housing, BMW Automotive Sales, Breast Cancer Risk & Survival datasets, Jordan market regression (Real world data)
- Scatter, bar, and line plots with API-imported data via KaggleHub

###  Prediction Models
Supervised learning models built with scikit-learn for practice and performance comparison:
- Diabetes prediction
- Housing price regression
- Obesity classification
- Cancer Survival rate regression
- Traffic prediction

## Target Project
### Admin project with gesture control
The idea comes from Tony Stark who uses holograms and his hands to control the holograms. <br> <br>
My idea is to make a facial recognition model to recgonise when its me in the frame, and then a set of commands will be unlocked for just me to use. 
<br> <br>
This project will take some time and I'm willing to take the long way to reach it by perfecting my courses normally and eventually completing it. 

## Stack
- Python  
- Pandas 
- Numpy 
- Scikit-Learn
- Matplotlib 
- OpenCV 
- Dlib 
- DeepFace  

## Structure
models             → sklearn prediction models <br>
computer_vision    → OpenCV projects <br>
data_visualization → matplotlib plots <br>
competitions       → Kaggle competition entries <br>
experiments        → failed/exploratory attempts <br>
data               → gitignored, populated by download_data.py <br>

## Sources
- Udemy: Computer Vision masterclass
- Andrew Ng Specializations (Coursera) (NEXT)
- Kaggle Learn courses
- AI assistance (non-vibecoding)

## Setup

### 1. Environment Setup
Clone the repository and install the required core dependencies:

```bash
git clone [https://github.com/Yassinini/ML-Learning-Journey.git](https://github.com/Yassinini/ML-Learning-Journey.git)
cd ML-Learning-Journey
pip install -r requirements.txt

```

### 2. Kaggle API Configuration
This project pulls datasets dynamically via the KaggleHub API. 

1. Download your `kaggle.json` key from your **[Kaggle Account Settings](https://www.kaggle.com/settings)** (API -> Create New Token).
2. Move the file to your system's default directory:
   * **Linux/macOS:** `~/.kaggle/kaggle.json`
   * **Windows:** `%USERPROFILE%\.kaggle\kaggle.json`

### 3. Initialize Data
The `data/` folder is gitignored. Run the automation script to download all required datasets:

```bash
python download_data.py
```
