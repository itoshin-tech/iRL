import tensorflow as tf
import numpy as np
from PIL import Image

print(tf.__version__)
# 2.1.0

tf.random.set_seed(0)

"""
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28), name='flatten_layer'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
], name='my_model')
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_categorical_accuracy],
)

model.save('agt_data/test.h5')
"""

newmodel = tf.keras.models.load_model('agt_data/test.h5')

newmodel.summary()
