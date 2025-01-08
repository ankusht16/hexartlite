import tensorflow as tf

# Load the .h5 model
model = tf.keras.models.load_model('chatbot_model.h5')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: Apply optimization to reduce model size
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert and save the model
tflite_model = converter.convert()

with open('chatbot_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model has been successfully converted to TensorFlow Lite format.")
