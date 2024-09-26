import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2


class ResNet50:
    def __init__(self, weights="imagenet"):
        self.model = tf.keras.applications.ResNet50(weights=weights)

    def get_resized_images(self, images):
        images_resized = []
        for image in images:
            image_resized = tf.keras.layers.Resizing(height=224, width=224, crop_to_aspect_ratio=True)(image)
            images_resized.append(image_resized)
        return np.array(images_resized)

    def predict(self, images=None, images_resized=None):
        # if np.array(images).ndim == 3: images = np.array([images])
        if images and type(images[0]) == str:
            images_arr = []
            for path in images:
                images_arr.append(cv2.imread(path)[..., ::-1])
            images = images_arr
        self.images_resized = images_resized or self.get_resized_images(images)
        inputs = tf.keras.applications.resnet50.preprocess_input(self.images_resized.copy())
        # TYPE CASTING DUE TO UNINTENDED CLIPPING DURING PLOTTING
        self.images_resized = tf.cast(self.images_resized, tf.uint8)
        preds = tf.keras.applications.resnet50.decode_predictions(self.model.predict(inputs, verbose=0))
        self.labels = []
        for pred in preds: self.labels.append(pred[0][1])
        # return self.labels
    
    def plot(self, rows="auto", show_axis=False):
        show_axis = "on" if show_axis else "off"
        len_labels = len(self.labels)
        if rows == "auto": rows = int(len_labels ** .5)
        cols = (len_labels+rows-1)//rows
        fctr = 2.5
        fig, axes = plt.subplots(rows, cols, figsize=(cols*fctr, rows*fctr))
        if rows == cols == 1: axes = [[axes]]
        elif 1 in (rows, cols): axes = [[axis for axis in axes]]
        for i, (pred, img) in enumerate(zip(self.labels, self.images_resized)):
            axis = axes[i//cols][i%cols]
            axis.set_title(pred)
            axis.imshow(img)
            axis.axis(show_axis)
        plt.tight_layout()

    def predict_plot(self, images, images_resized=None, rows="auto"):
        self.predict(images, images_resized)
        self.plot(rows)