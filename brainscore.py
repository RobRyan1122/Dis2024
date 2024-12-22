import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from brainscore_vision.benchmarks.majajhong2015 import MajajHongV4PublicBenchmark

def my_model():
    # Load the ResNet50 model, excluding the top (output) layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Optionally, add your own layers on top of the ResNet50 base
    # For example, you could add a GlobalAveragePooling layer followed by a Dense layer
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(86, activation='softmax')(x)  # Assuming 86 classes
    model = Model(inputs=base_model.input, outputs=x)

    # Compile the model (optional: you can adjust the loss function and optimizer as needed)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Now you can use your model function in the benchmarking code
benchmark = MajajHongV4PublicBenchmark()  # Choose your benchmark
model = my_model()  # Create your model
score = benchmark(model)  # Score the model using the benchmark
print(score)
