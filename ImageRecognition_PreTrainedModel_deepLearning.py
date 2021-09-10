import numpy as np
from keras.preprocessing import image
from keras.applications import vgg16
from pathlib import Path

# Load Keras' VGG16 model that was pre-trained against the ImageNet database
model = vgg16.VGG16()

# Load the image file, resizing it to 224x224 pixels (required by this model)
script_location = Path(__file__).absolute().parent
img_file = script_location / "bay.jpg"
img = image.load_img(img_file, target_size=(224, 224))

# Convert the image to a numpy array
x = image.img_to_array(img)

# Add a fourth dimension (since Keras expects a list of images)
x = np.expand_dims(x, axis=0)

# Normalize the input image's pixel values to the range used when training the neural network
x = vgg16.preprocess_input(x)

# Run the image through the deep neural network to make a prediction
predictions = model.predict(x)

# Look up the names of the predicted classes. Index zero is the results for the first image.
predicted_classes = vgg16.decode_predictions(predictions, top = 9)

print("Top predictions for this image:")

for imagenet_id, name, likelihood in predicted_classes[0]:
    print("Prediction: {} - {:2f}".format(name, likelihood))



#print image
import matplotlib.pyplot as plt
img = image.load_img(img_file, target_size=(224, 224))
sample_image = img
# Draw the image as a plot
plt.imshow(sample_image)
# Show the plot on the screen
plt.show()