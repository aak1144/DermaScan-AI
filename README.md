# 🔬 DermaScan AI - Skin Disease Detection & Care Assistant

A cutting-edge web application that leverages advanced deep learning models to analyze skin lesions and provide comprehensive dermatological assistance. DermaScan AI combines computer vision with conversational AI to deliver dual-model analysis and personalized skin care guidance in multiple languages.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Integration](#api-integration)
- [Model Specifications](#model-specifications)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Performance Optimization](#performance-optimization)
- [Security Best Practices](#security-best-practices)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

DermaScan AI is an intelligent dermatological assessment platform designed to:

1. **Analyze skin lesions** using two state-of-the-art deep learning models
2. **Compare predictions** from a Custom CNN and VGG16 transfer learning model
3. **Provide skincare guidance** through an AI-powered conversational assistant
4. **Support multilingual interaction** (English, Hindi, Kannada)
5. **Enable voice input/output** for accessibility and user convenience

The application runs on **Streamlit**, providing a responsive, real-time web interface without requiring frontend expertise.

---

## ✨ Features

### Core Features

| Feature | Description |
|---------|-------------|
| **Dual Model Analysis** | Compare predictions from Custom CNN and VGG16 models for better accuracy |
| **Image Upload & Processing** | Support for JPG and PNG formats with automatic preprocessing |
| **Confidence Scoring** | Display probability scores for each prediction class |
| **Binary & Multiclass Support** | Handle both binary (benign/malignant) and multiclass classification |
| **Comparison View** | Side-by-side analysis results for both models |

### AI Assistant Features

| Feature | Description |
|---------|-------------|
| **Conversational AI** | Powered by Google Generative AI (Gemini models) |
| **Multilingual Support** | English, Hindi, and Kannada language support |
| **Voice Input** | Speech-to-text capability with automatic language detection |
| **Voice Output** | Text-to-speech synthesis in multiple languages |
| **Chat History** | Maintains conversation context across multiple turns |
| **Domain-Specific** | Restricted to skin care and dermatology topics |

### UI/UX Features

| Feature | Description |
|---------|-------------|
| **Modern Dark Theme** | Eye-friendly gradient background with glassmorphism design |
| **Responsive Layout** | Two-column design for image and results comparison |
| **Status Indicators** | Real-time model loading status and error messages |
| **Expandable Details** | Raw output inspection for technical users |
| **Custom CSS** | Professional styling with smooth animations |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Interface (Streamlit)                   │
├─────────────────────────────────────────────────────────────────┤
│  Image Upload │ Model Selection │ Results Display │ Chat Panel   │
├─────────────────────────────────────────────────────────────────┤
│                      Image Preprocessing                         │
│              (Resize, Normalize, Batch Expansion)                │
├──────────────────────┬──────────────────────────────────────────┤
│  Custom CNN Model    │    VGG16 Transfer Learning Model          │
│  (custom_cnn.h5)     │    (vgg16_skin_cancer.h5)                │
├──────────────────────┴──────────────────────────────────────────┤
│              Prediction & Confidence Calculation                 │
├─────────────────────────────────────────────────────────────────┤
│                   Google Generative AI (Gemini)                  │
│         (Conversational Assistant & Multilingual Support)        │
├─────────────────────────────────────────────────────────────────┤
│    Speech Recognition (STT) │ Text-to-Speech Synthesis (TTS)     │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Image Input** → User uploads skin lesion image (JPG/PNG)
2. **Preprocessing** → Image resized and normalized to model input specifications
3. **Dual Inference** → Concurrent predictions from both models
4. **Post-processing** → Probability normalization and confidence calculation
5. **Output Display** → Results rendered in comparison view with confidence scores
6. **Chat Processing** → User query sent to Gemini with system prompt and history
7. **Voice Processing** → Optional voice input converted to text, response converted to audio

---

## 🚀 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **pip**: Package installer for Python
- **Virtual Environment**: Recommended for dependency isolation

### Step 1: Clone/Download Project

```bash
# Navigate to your desired directory
cd /path/to/projects

# If using Git
git clone <repository-url>
cd DermaScan\ AI
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
# Test imports
python test_imports.py

# Check model availability
python check_models.py
```

### Requirements

The project uses the following Python packages:

```
streamlit            # Web framework
numpy               # Numerical computing
tensorflow          # Deep learning framework
Pillow              # Image processing
google-generativeai # Gemini API
python-dotenv       # Environment variable management
streamlit-mic-recorder # Voice input
SpeechRecognition   # Speech processing
gTTS                # Google Text-to-Speech
```

---

## ⚙️ Configuration

### 1. Environment Variables

Create a `.env` file in the project root:

```bash
# .env file
GEMINI_API_KEY=your_actual_api_key_here
```

**Do NOT commit .env files to version control!**

### 2. Model Files Setup

Ensure the following model files are in the project root directory:

```
DermaScan AI/
├── custom_cnn_model.h5      (Custom CNN model)
├── vgg16_skin_cancer.h5     (VGG16 transfer learning model)
└── app.py
```

**To obtain model files:**
- Train your own models or download pre-trained versions
- Place them in the same directory as `app.py`
- Verify model loading with: `python check_models.py`

### 3. Input Size Configuration

Edit `app.py` to match your model's expected input dimensions:

```python
# Line 37-39
CUSTOM_CNN_SIZE = (128, 128)    # Adjust to your model's input
VGG16_SIZE = (128, 128)         # Adjust to your model's input
```

### 4. Classification Labels

Update the class labels in `app.py` (Line 41-45):

```python
LABELS = [
    "Class 0: Negative / Benign Lesion", 
    "Class 1: Positive / Malignant Tumor", 
    "Class 2: Unknown / Other"
]
```

### 5. Gemini API Setup

1. **Get API Key:**
   - Visit [Google AI Studio](https://ai.google.dev/)
   - Click "Get API Key"
   - Create a new API key (free tier available)

2. **Add to Environment:**
   - Add to `.env` file (recommended)
   - Or paste in the Streamlit sidebar when prompted

3. **Rate Limits:**
   - Free tier: 60 requests per minute
   - Paid tier: Higher limits available

---

## 📖 Usage

### Starting the Application

```bash
# Activate virtual environment first
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Run the Streamlit app
streamlit run app.py
```

The application will open at `http://localhost:8501` in your default browser.

### Using the Image Analysis Feature

1. **Upload Image:**
   - Click "Select an image file (JPG or PNG) for analysis"
   - Choose a JPG or PNG image of a skin lesion

2. **Review Results:**
   - Left panel: Original uploaded image
   - Right panel (top): Custom CNN model predictions
   - Right panel (bottom): VGG16 model predictions

3. **Interpret Output:**
   - **Predicted Diagnosis:** Classification result (e.g., "Benign Lesion")
   - **Confidence:** Probability percentage of the predicted class
   - **Raw Output:** Expandable section showing probabilities for all classes

### Using the Skin Care Assistant

1. **Text Input:**
   - Type your question about skin care in the chat box
   - Press Enter or click send

2. **Voice Input:**
   - Select language from sidebar (English, Hindi, Kannada)
   - Click "Start Recording"
   - Speak your question
   - Click "Stop Recording"

3. **Voice Output:**
   - Audio player appears with assistant's response
   - Listen or read the text response

4. **Chat History:**
   - All messages are maintained in the session
   - Context is preserved across multiple turns

---

## 🔌 API Integration

### Google Generative AI (Gemini)

**API Endpoint:** `https://generativeai.googleapis.com/`

**Supported Models** (in order of preference):

1. `models/gemini-flash-latest` - Fastest, optimized
2. `models/gemini-pro-latest` - Balanced performance
3. `models/gemini-2.5-flash` - Alternative fast model
4. `models/gemini-2.5-pro` - Fallback option

**Features:**
- Automatic model fallback on quota exceeded
- System prompts for domain-specific responses
- Multilingual support
- Conversation history integration

**Error Handling:**
- Quota exceeded (429) → Try next model
- API errors → Display user-friendly message
- Invalid API key → Prompt for re-entry

---

## 🧠 Model Specifications

### Custom CNN Model

| Parameter | Value |
|-----------|-------|
| **Model Type** | Convolutional Neural Network |
| **Input Size** | 128×128 pixels (RGB) |
| **Output Type** | Binary or Multiclass Classification |
| **File** | `custom_cnn_model.h5` |
| **Training Method** | Custom training |
| **Inference Speed** | Real-time (<1s) |

**Preprocessing Pipeline:**
```
Input Image → Convert to RGB → Resize (128×128) → 
Normalize (0-1) → Add Batch Dimension → Model Input
```

### VGG16 Transfer Learning Model

| Parameter | Value |
|-----------|-------|
| **Model Type** | Transfer Learning (VGG16 base) |
| **Input Size** | 128×128 pixels (RGB) |
| **Output Type** | Binary or Multiclass Classification |
| **File** | `vgg16_skin_cancer.h5` |
| **Training Method** | Transfer learning on VGG16 |
| **Inference Speed** | Real-time (<1s) |
| **Base Model** | VGG16 (pre-trained on ImageNet) |

**Preprocessing Pipeline:**
```
Input Image → Convert to RGB → Resize (128×128) → 
Normalize (0-1) → Add Batch Dimension → Model Input
```

### Output Interpretation

**Binary Classification (1 Output Neuron):**
- Sigmoid activation
- Single probability value for Class 1 (Malignant)
- Threshold: 0.5
- Classes: [Benign, Malignant]

**Multiclass Classification (N Output Neurons):**
- Softmax activation
- Probability distribution across all classes
- Highest probability = predicted class
- Sum of probabilities = 1.0

---

## 🛠️ Troubleshooting

### Common Issues & Solutions

#### 1. Model Files Not Found

**Error:** `❌ Error: Model file 'custom_cnn_model.h5' not found.`

**Solution:**
- Verify model files exist in the project directory
- Check file names match exactly (case-sensitive on Linux/Mac)
- Run: `python check_models.py` to diagnose
- Ensure .h5 files are not in subdirectories

#### 2. Import Errors

**Error:** `ModuleNotFoundError: No module named 'streamlit'`

**Solution:**
```bash
# Verify virtual environment is activated
pip install -r requirements.txt
python test_imports.py
```

#### 3. CSS File Missing

**Warning:** `CS file not found. Please ensure 'assets/style.css' exists.`

**Solution:**
- Create `assets/` directory if missing
- Place `style.css` in the `assets/` folder
- Reload the Streamlit app (press 'R' or refresh browser)

#### 4. Gemini API Key Issues

**Error:** `API quota exceeded` or `Invalid API key`

**Solutions:**
- Verify `.env` file has correct API key
- Check API key is active at [Google AI Console](https://ai.google.dev/)
- Wait 24 hours for free tier quota reset
- Upgrade to paid plan for higher limits
- Alternatively, paste key in sidebar temporarily

#### 5. File Pointer Error

**Error:** `Error reading file after first model prediction`

**Solution:**
- Already handled in code with `uploaded_file.seek(0)`
- Ensure you're using the latest app.py version
- Clear browser cache: Ctrl+Shift+Delete (Windows/Linux) or Cmd+Shift+Delete (Mac)

#### 6. Out of Memory Error

**Error:** `RuntimeError: CUDA out of memory` or system becomes slow

**Solutions:**
```bash
# Option 1: Set CPU-only mode (slower but uses less RAM)
set TF_CPP_LOGGING_LEVEL=3  # Windows
export TF_CPP_LOGGING_LEVEL=3  # macOS/Linux

# Option 2: Use smaller batch sizes (already batch_size=1)

# Option 3: Restart Streamlit (clears cache)
streamlit run app.py --logger.level=error
```

#### 7. Microphone Not Working

**Error:** `Error: Microphone not available` or no input detected

**Solutions:**
- Check browser permissions (Allow Microphone)
- Test microphone in other apps first
- Try different browser (Chrome recommended)
- Use text input as alternative

#### 8. Text-to-Speech (TTS) Issues

**Error:** `Error generating speech: ...`

**Solutions:**
- Check internet connection (gTTS requires online)
- Verify language code is valid
- Try smaller text first
- Check for special characters that TTS can't process

---

## 📁 Project Structure

```
DermaScan AI/
│
├── 📄 app.py                          # Main Streamlit application
├── 📄 check_models.py                 # Model verification utility
├── 📄 test_imports.py                 # Dependency testing script
├── 📄 requirements.txt                # Python dependencies
├── 📄 README.md                       # This file
├── 📄 .env.example                    # Example environment variables
│
├── 🧠 custom_cnn_model.h5            # Custom CNN model (binary/multiclass)
├── 🧠 vgg16_skin_cancer.h5           # VGG16 transfer learning model
│
├── 📁 assets/
│   └── 📄 style.css                   # Custom Streamlit styling
│
├── 📁 __pycache__/                    # Python cache (auto-generated)
│
└── 📄 JupyterFile.ipynb              # Jupyter notebook (analysis/training)
```

---

## ⚡ Performance Optimization

### Model Loading Optimization

The application uses **Streamlit's caching** (`@st.cache_resource`):

```python
@st.cache_resource
def load_and_verify_model(path, model_name):
    # Models loaded only once during app startup
    # Cached across all user sessions
```

**Benefits:**
- Models load only once per app startup
- Subsequent predictions are instant
- Memory efficient
- Eliminates repeated I/O operations

### Image Preprocessing Optimization

```python
def preprocess_image(image_file, target_size):
    # Efficient batch processing (batch_size=1)
    # Minimal memory footprint
    # Fast resizing and normalization
```

**Optimization Techniques:**
- Direct NumPy operations (faster than loops)
- In-place normalization
- Minimal data type conversions
- Vectorized operations

### Prediction Optimization

```python
# Verbose=0 suppresses unnecessary logging
predictions = model.predict(processed_input, verbose=0)

# Compile=False for inference-only models
model = load_model(path, compile=False)
```

### UI Rendering Optimization

- Lazy loading of components
- Minimal re-renders through caching
- Efficient column layouts
- Asset compression through CSS

---

## 🔒 Security Best Practices

### 1. API Key Management

```python
# ✅ GOOD: Use environment variables
api_key = os.getenv("GEMINI_API_KEY")

# ❌ BAD: Never hardcode API keys
api_key = "AIzaSyDxxx..."
```

**Implementation:**
- Store API key in `.env` file
- Add `.env` to `.gitignore`
- Never commit secrets to version control
- Rotate keys periodically

### 2. Input Validation

```python
# File type validation
if uploaded_file is not None:
    if uploaded_file.type in ["image/jpeg", "image/png"]:
        # Process file
```

**Security Checks:**
- File type validation (MIME types)
- File size limits (prevent DoS)
- Image format verification
- Malicious file detection

### 3. Model Security

```python
# Load models with security
model = load_model(path, compile=False)
```

**Considerations:**
- Verify model file integrity
- Use trusted model sources
- Regular model updates
- Monitor for adversarial inputs

### 4. User Data Privacy

- No data storage (stateless architecture)
- Chat history stored only in session (cleared on app restart)
- Uploaded images processed in-memory only
- No data sent to third parties (except Gemini API)

### 5. Rate Limiting & Quotas

```python
# Handle API quota exceeded
if "429" in str(error) or "quota" in str(error).lower():
    error_msg = "API quota exceeded. Please try again later."
```

**Strategies:**
- Implement request throttling
- Set appropriate rate limits
- Monitor API usage
- Upgrade plan if needed

### 6. Dependency Management

```bash
# Regular security updates
pip install --upgrade pip
pip install -r requirements.txt --upgrade
pip audit  # Check for vulnerabilities
```

---

## 🤝 Contributing

### How to Contribute

1. **Fork the Repository**
   ```bash
   git clone https://github.com/yourusername/DermaScan-AI.git
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Follow PEP 8 style guide
   - Add comments for complex logic
   - Test thoroughly

4. **Commit Changes**
   ```bash
   git commit -m "Add: Brief description of changes"
   ```

5. **Push to Branch**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Submit Pull Request**
   - Describe changes clearly
   - Reference any related issues

### Development Guidelines

- **Code Style:** PEP 8 compliant
- **Comments:** Comprehensive documentation
- **Testing:** Test on multiple environments
- **Error Handling:** Graceful fallbacks
- **Performance:** Optimize for speed and memory

---

## 📝 License

This project is licensed under the **MIT License** - see LICENSE file for details.

**MIT License Summary:**
- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ❌ Warranty not provided
- ⚠️ Attribution required

---

## 📚 Additional Resources

### Documentation
- [Streamlit Documentation](https://docs.streamlit.io/)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs)
- [Google Generative AI Docs](https://ai.google.dev/docs)
- [Python Official Docs](https://docs.python.org/3/)

### Tutorials
- [Building Streamlit Apps](https://docs.streamlit.io/library/get-started)
- [Transfer Learning with TensorFlow](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Medical Image Analysis with Python](https://pytorch.org/vision/stable/index.html)

### Model Training
- [Training Custom CNN Models](https://www.tensorflow.org/tutorials/images/cnn)
- [VGG16 Transfer Learning](https://towardsdatascience.com/transfer-learning-with-vgg16-645a0dfb32d3)
- [Skin Lesion Datasets](https://dataverse.harvard.edu/dataverse/SkinLesionAnalysisFramework)

---

## 🐛 Issue Reporting

Found a bug? Have a suggestion?

1. **Check existing issues** to avoid duplicates
2. **Describe the problem** clearly with steps to reproduce
3. **Include system information:**
   - OS (Windows/macOS/Linux)
   - Python version
   - Streamlit version
   - Error messages/logs

4. **Provide expected vs. actual behavior**

---

## 👥 Support

For questions, issues, or suggestions:

- **GitHub Issues:** Submit issue on GitHub
- **Email:** [your-email@example.com]
- **Documentation:** Refer to this README
- **Community:** Reach out to community forums

---

## 🎉 Acknowledgments

- Google Generative AI team for Gemini API
- Streamlit team for the amazing web framework
- TensorFlow/Keras community for deep learning tools
- Medical imaging researchers and datasets

---

## 🚦 Project Status

- **Current Version:** 1.0.0
- **Status:** Production Ready ✅
- **Last Updated:** January 2026
- **Maintenance:** Active

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Average Prediction Time** | <1 second (per model) |
| **Memory Usage** | ~500MB (with both models loaded) |
| **Supported Image Formats** | JPG, JPEG, PNG |
| **Concurrent Users** | Depends on server capacity |
| **API Response Time** | 1-5 seconds (Gemini) |
| **Uptime SLA** | 99% (when properly deployed) |

---

## 📞 Contact & Feedback

We'd love to hear from you! Your feedback helps us improve DermaScan AI.

- **Bug Reports:** [GitHub Issues](https://github.com/yourusername/DermaScan-AI/issues)
- **Feature Requests:** [GitHub Discussions](https://github.com/yourusername/DermaScan-AI/discussions)
- **Email:** support@dermascan-ai.example.com

---

**Made with ❤️ for better skin health diagnosis**

*Disclaimer: DermaScan AI is a diagnostic assistance tool and should not be used as a substitute for professional medical advice. Always consult with a qualified dermatologist for accurate diagnosis and treatment.*

---

**Happy scanning! 🔬**
