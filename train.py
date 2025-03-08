import numpy as np
import wandb
from keras.datasets import fashion_mnist

class_labels = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def load_images():
    wandb.init(project="gd_witg_backpropagation", name="experiment_1")
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    image_list = []
    for label in np.unique(y_train):
        idx = np.where(y_train == label)[0][0]
        image = x_train[idx]
        image_caption = class_labels[label]
        image_list.append(wandb.Image(image, caption=image_caption))

    wandb.log({"Images": image_list})
    wandb.finish()

if __name__ == "__main__":
    load_images()
