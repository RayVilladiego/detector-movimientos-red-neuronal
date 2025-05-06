import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 16
IMG_SIZE = (224, 224)

datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
train = datagen.flow_from_directory("data", target_size=IMG_SIZE, batch_size=BATCH_SIZE, subset='training')
val = datagen.flow_from_directory("data", target_size=IMG_SIZE, batch_size=BATCH_SIZE, subset='validation')

model = tf.keras.applications.MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
model.trainable = False

model = tf.keras.Sequential([
    model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(train.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train, epochs=10, validation_data=val)
model.save("model/detector_mov.h5")
