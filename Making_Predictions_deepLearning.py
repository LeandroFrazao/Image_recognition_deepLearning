from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np

# These are the CIFAR10 class labels from the training data (in order from 0 to 9)
class_labels = [
    "Plane",
    "Car",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Boat",
    "Truck"
]

# Load the json file that contains the model's structure
script_location = Path(__file__).absolute().parent
model_structure_file = script_location / "model_structure_fromColab.json"
#model_structure_file = Path("model_structure.json").read_text()
model_structure = model_structure_file.read_text()

# Recreate the Keras model object from the json data
model = model_from_json(model_structure)

# Re-load the model's trained weights
weight_file= script_location / "model_weights_fromColab.h5"
model.load_weights(weight_file)

# Load an image file to test, resizing it to 32x32 pixels (as required by this model)
img_file = script_location / "test3_F.png"
img = image.load_img(img_file, target_size=(32, 32))

# Convert the image to a numpy array
image_to_test = image.img_to_array(img) / 255

# Add a fourth dimension to the image (since Keras expects a list of images, not a single image)
list_of_images = np.expand_dims(image_to_test, axis=0)

# Make a prediction using the model
results = model.predict(list_of_images)

# Since we are only testing one image, we only need to check the first result
single_result = results[0]

# We will get a likelihood score for all 10 possible classes. Find out which class had the highest score.
most_likely_class_index = int(np.argmax(single_result))
class_likelihood = single_result[most_likely_class_index]

# Get the name of the most likely class
class_label = class_labels[most_likely_class_index]

# Print the result
print("This is image is a {} - Likelihood: {:2f}".format(class_label, class_likelihood))


###Print image
import matplotlib.pyplot as plt
sample_image = image.load_img(img_file, target_size=(32, 32))
# Draw the image as a plot
plt.imshow(sample_image)
# Label the image
plt.title(class_label +" ?")
# Show the plot on the screen
plt.show()
