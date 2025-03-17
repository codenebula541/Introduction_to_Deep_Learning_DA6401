import numpy as np
import wandb
from keras.datasets import fashion_mnist

# Class labels in Fashion-MNIST
class_labels = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def log_sample_images():
    # Initialize a new W&B run
    wandb.init(project="DA6401-Assignment1", name="experiment_1")

    # Load the Fashion-MNIST dataset
    (x_train, y_train), (_,_) = fashion_mnist.load_data()

    # Select one sample per class
    sample_images = []      #list that will conatin the sample images from each class_lable
    for class_label in np.unique(y_train):
        index = np.where(y_train == class_label)[0][0]  # np.where(y_train == class_label) gives all ocuurencing index of class_label in tuple form. [0][0] extarct first array from tuple, first element ie first occurence of class_label
        image = x_train[index]
        caption = class_labels[class_label]
        sample_images.append(wandb.Image(image, caption=caption))

    # Log images to W&B
    wandb.log({"Sample Images": sample_images})

    # Finish the W&B run
    wandb.finish()

if __name__ == "__main__":
    log_sample_images()
