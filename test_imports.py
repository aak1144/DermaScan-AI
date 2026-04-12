try:
    import google.protobuf
    print(f"Protobuf version: {google.protobuf.__version__}")
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    print("Imports successful!")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
