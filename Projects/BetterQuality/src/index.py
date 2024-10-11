import tensorflow as tf
import tensorflow_hub as hub
import cv2

# Cargar el modelo ESRGAN
model = hub.load('https://tfhub.dev/captain-pool/esrgan-tf2/1')

# Cargar y preparar la imagen
image = cv2.imread("Profile.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = tf.convert_to_tensor(image, dtype=tf.float32)
image = image[tf.newaxis, ...]

# Mejora la imagen
enhanced_image = model(image)
enhanced_image = tf.clip_by_value(enhanced_image[0], 0, 255)
enhanced_image = tf.cast(enhanced_image, tf.uint8)

# Guardar la imagen mejorada
cv2.imwrite("imagen_mejorada3.jpg", cv2.cvtColor(enhanced_image.numpy(), cv2.COLOR_RGB2BGR))
