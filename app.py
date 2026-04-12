import streamlit as st
import numpy as np
# We must import Keras components from tensorflow
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os
from dotenv import load_dotenv
import io

load_dotenv() # Load environment variables from .env file

# --- Streamlit Page Configuration (MUST BE FIRST) ---
st.set_page_config(
    page_title="Skin Care & Cancer Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔬"
)

import google.generativeai as genai
from streamlit_mic_recorder import speech_to_text # New Import
from gtts import gTTS # New Import for Text-to-Speech

# --- Custom CSS Injection ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Try to load CSS if exists
try:
    local_css("assets/style.css")
except FileNotFoundError:
    st.warning("CS file not found. Please ensure 'assets/style.css' exists.")

# --- Configuration and Constants ---
# Define model file names (must match your local file names)
CUSTOM_CNN_PATH = "custom_cnn_model.h5"
VGG16_PATH = "vgg16_skin_cancer.h5"

# Define expected input sizes for preprocessing (adjust these if your models were trained on different sizes)
CUSTOM_CNN_SIZE = (128, 128) # Assuming a typical custom size
VGG16_SIZE = (128, 128)     # Standard VGG16 input size

# Define generic class labels (UPDATE THESE with your actual skin cancer diagnosis labels)
# Note: For binary classification (output size 1), only the first two labels (Index 0 and 1) are used.
LABELS = [
    "Class 0: Negative / Benign Lesion", 
    "Class 1: Positive / Malignant Tumor", 
    "Class 2: Unknown / Other"
]


# --- 1. Model Loading with Caching ---
# @st.cache_resource ensures the heavy models are loaded only once when the app starts
@st.cache_resource
def load_and_verify_model(path, model_name):
    """Loads a Keras model from the specified path."""
    try:
        # We use compile=False because we only need inference, not training
        model = load_model(path, compile=False)
        st.sidebar.success(f"✅ Successfully loaded {model_name}.")
        return model
    except FileNotFoundError:
        st.sidebar.error(f"❌ Error: Model file '{path}' not found.")
        st.sidebar.warning("Place this file in the same directory as the script.")
        return None
    except Exception as e:
        st.sidebar.error(f"❌ Error loading {model_name}: {e}")
        return None

# Load models
custom_cnn_model = load_and_verify_model(CUSTOM_CNN_PATH, "Custom CNN")
vgg16_model = load_and_verify_model(VGG16_PATH, "VGG16 Skin Cancer")


# --- 2. Image Preprocessing Logic ---
def preprocess_image(image_file, target_size):
    """Loads, resizes, and normalizes the image for model input."""
    # Convert uploaded file to PIL Image
    image = Image.open(image_file).convert('RGB')
    
    # Resize the image to the model's expected input shape
    image = image.resize(target_size)
    
    # Convert to NumPy array
    img_array = np.array(image).astype('float32')
    
    # Normalize: Scale pixel values to the [0, 1] range
    img_array /= 255.0
    
    # Add a batch dimension (1, H, W, C)
    return np.expand_dims(img_array, axis=0)


# --- 3. Prediction Function ---
def predict_image(image_file, model, input_size):
    """Preprocesses the image and makes a prediction."""
    if model is None:
        return {"Predicted Class": "Model Not Found", "Confidence": "N/A", "Raw Output": []}

    try:
        # Preprocess the input image
        processed_input = preprocess_image(image_file, input_size)

        # Predict (verbose=0 suppresses Keras output during prediction)
        predictions = model.predict(processed_input, verbose=0)[0]
        
        # Determine the output dimensionality
        if predictions.size == 1:
            # --- FIX: Explicitly handle Binary Classification (Sigmoid output, 1 neuron) ---
            prob_class_1 = predictions.item() # Get the single scalar value (Prob of Class 1: Malignant)
            
            # Determine prediction based on 0.5 threshold
            if prob_class_1 >= 0.5:
                predicted_index = 1
                confidence = prob_class_1
            else:
                predicted_index = 0
                confidence = 1.0 - prob_class_1 # Confidence in Class 0: Benign

            # For raw output display, show both probabilities
            raw_output = [1.0 - prob_class_1, prob_class_1]

        else:
            # Case 2: Multiclass Classification (Softmax output, N neurons)
            predicted_index = np.argmax(predictions)
            confidence = predictions[predicted_index]
            raw_output = predictions.tolist()


        # Determine the label based on the index
        predicted_label = LABELS[predicted_index] if predicted_index < len(LABELS) else f"Index {predicted_index} (Label Not Defined)"

        return {
            "Predicted Class": predicted_label,
            "Confidence": f"{confidence * 100:.2f}%",
            "Raw Output": raw_output
        }

    except Exception as e:
        # CRUCIAL FIX: Return the actual error message (str(e)) to the UI
        return {"Predicted Class": "Prediction Error", "Confidence": str(e), "Raw Output": []}


# --- 4. Streamlit UI Layout ---

st.title("🔬 DermaScan AI")
st.markdown("A comparison predictor using a custom model and a VGG16 transfer-learned model.")
st.markdown("---")

st.sidebar.header("Model Status & Specs")
st.sidebar.info("Ensure the model files are in the same folder as this script.")

# Display model specifications in the sidebar
st.sidebar.markdown(f"**Custom CNN:** Input Size `{CUSTOM_CNN_SIZE[0]}x{CUSTOM_CNN_SIZE[1]}`")
st.sidebar.markdown(f"**VGG16 Model:** Input Size `{VGG16_SIZE[0]}x{VGG16_SIZE[1]}`")

st.sidebar.markdown("---")
st.sidebar.header("Chatbot Configuration")

# Try to get key from environment variable first
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    api_key = st.sidebar.text_input("Gemini API Key", type="password")
    if not api_key:
        st.sidebar.warning("API Key not found in .env or input.")

if api_key:
    genai.configure(api_key=api_key)

st.header("Upload Image")
uploaded_file = st.file_uploader(
    "Select an image file (JPG or PNG) for analysis:", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Use a two-column layout for the image and the results section
    col_image, col_results = st.columns([1, 2])

    with col_image:
        st.subheader("Input Image")
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    with col_results:
        st.subheader("Comparison Analysis")
        st.info("Processing image with both models...")

        # Create two sub-columns for model results display
        res_col1, res_col2 = st.columns(2)
        
        # Important: File objects (like uploaded_file) can only be read once. 
        # We need to seek back to the start or clone the object before the second model reads it.

        # ---------------------
        # Custom CNN Prediction
        # ---------------------
        with res_col1:
            st.markdown("<h4 style='color: #4CAF50;'>Custom CNN Model</h4>", unsafe_allow_html=True)
            custom_cnn_result = predict_image(uploaded_file, custom_cnn_model, CUSTOM_CNN_SIZE)
            
            st.metric(
                label="Predicted Diagnosis",
                value=custom_cnn_result["Predicted Class"]
            )
            st.metric(
                label="Confidence (of predicted class)",
                value=custom_cnn_result["Confidence"]
            )
            
            with st.expander("Raw Model Output (Probabilities)"):
                # For binary models, this will now show [Prob Class 0, Prob Class 1]
                st.json(custom_cnn_result["Raw Output"])

        # ---------------------
        # VGG16 Prediction
        # ---------------------
        with res_col2:
            st.markdown("<h4 style='color: #2196F3;'>VGG16 Model (Transfer Learning)</h4>", unsafe_allow_html=True)
            
            # Reset file pointer for the second model
            uploaded_file.seek(0) 
            vgg16_result = predict_image(uploaded_file, vgg16_model, VGG16_SIZE)
            
            st.metric(
                label="Predicted Diagnosis",
                value=vgg16_result["Predicted Class"]
            )
            st.metric(
                label="Confidence (of predicted class)",
                value=vgg16_result["Confidence"]
            )

            with st.expander("Raw Model Output (Probabilities)"):
                st.json(vgg16_result["Raw Output"])

        st.markdown("---")
        st.success("Analysis complete! Review the combined output above.")
        st.markdown("The models may produce different results due to distinct architectures, training focus, and preprocessing steps. The error message for the VGG16 model should now be much more specific.")

else:
    st.info("Please upload an image file (JPG/PNG) to start the dual model prediction process.")

# --- 5. Skin Care Chatbot ---
st.markdown("---")
st.header("💬 Skin Care Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
input_text = st.chat_input("Ask about skin care...")

# Voice Input in Sidebar
with st.sidebar:
    st.markdown("### 🗣️ Language & Voice")
    
    # 1. Language Selector
    language_options = {
        "English": "en",
        "Hindi": "hi",
        "Kannada": "kn"
    }
    selected_language_name = st.selectbox(
        "Select Language / भाषा / ಭಾಷೆ",
        list(language_options.keys())
    )
    selected_language_code = language_options[selected_language_name]

    st.markdown("### 🎤 Voice Input")
    st.info(f"Speak in {selected_language_name}...")
    
    # 2. Update Voice Input Language
    voice_token = speech_to_text(
        language=selected_language_code, 
        start_prompt="Start Recording", 
        stop_prompt="Stop Recording", 
        just_once=False, 
        key='STT'
    )

# Determine prompt source
prompt = None
if input_text:
    prompt = input_text
elif voice_token:
    prompt = voice_token

if prompt:
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    if not api_key:
        st.error("Please provide a Gemini API Key in .env file or sidebar.")
    else:
        try:
            # System prompt to restrict domain
            base_system_prompt = """
            You are a specialized skin care assistant. You must respond ONLY to queries related to skincare, skin diseases, symptoms, treatments, remedies, products, dermatology advice, and general skin health.
            If the user asks anything that is NOT related to skincare or skin diseases, you must politely decline and say you can answer only skin-related queries.
            """
            
            # 3. Update System Prompt for Language
            if selected_language_code != 'en':
                lang_instruction = f"\nIMPORTANT: You must provide your response in the {selected_language_name} language (ISO code: {selected_language_code})."
                system_prompt = base_system_prompt + lang_instruction
            else:
                system_prompt = base_system_prompt
            
            # Try different Gemini models (in order of preference, with correct naming)
            model_names = [
                'models/gemini-flash-latest',
                'models/gemini-pro-latest',
                'models/gemini-2.5-flash',
                'models/gemini-2.5-pro'
            ]
            response = None
            
            for model_name in model_names:
                try:
                    model = genai.GenerativeModel(model_name, system_instruction=system_prompt)
                    
                    # Prepare the conversation history for the model
                    chat = model.start_chat(history=[
                        {"role": m["role"] if m["role"] == "user" else "model", "parts": [m["content"]]}
                        for m in st.session_state.messages[:-1]
                    ])
                    
                    # Send the message
                    response = chat.send_message(prompt)
                    break  # Success, exit loop
                    
                except Exception as model_error:
                    if "429" in str(model_error) or "quota" in str(model_error).lower():
                        continue  # Try next model
                    else:
                        raise  # Re-raise if it's not a quota error
            
            if response:
                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(response.text)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response.text})

                # --- NEW: Text-to-Speech Integration ---
                try:
                    # Generate audio from the response text
                    # 4. Update TTS Language
                    tts = gTTS(text=response.text, lang=selected_language_code)
                    
                    # Save to a bytes buffer to avoid creation of temporary files
                    audio_bytes_io = io.BytesIO()
                    tts.write_to_fp(audio_bytes_io)
                    audio_bytes_io.seek(0)
                    
                    # Display audio player
                    with st.chat_message("assistant"):
                        st.audio(audio_bytes_io, format='audio/mp3')
                        
                except Exception as tts_error:
                    st.error(f"Error generating speech: {tts_error}")
                # ---------------------------------------

            else:
                # All models failed due to quota
                error_msg = "⚠️ API quota exceeded. Please try again later or check your billing at https://ai.google.dev/gemini-api/docs/rate-limits"
                with st.chat_message("assistant"):
                    st.warning(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            if "429" in str(e) or "quota" in str(e).lower():
                error_msg = "⚠️ API quota exceeded. Your free tier limit has been reached. Please wait for the quota to reset (usually 24 hours) or upgrade your plan."
            st.error(error_msg)
            st.info("💡 Tip: Check your usage at https://ai.dev/rate-limit")

