from keras.api.applications import ResNet50
from keras.api.models import Model
from keras.api.layers import Dense, Flatten


def my_resnet50():
    base_model = ResNet50(include_top=False, pooling='avg', weights='imagenet')
    x = Flatten()(base_model.output)
    output = Dense(6, activation='sigmoid')(x)  # Multi-label classification
    model = Model(inputs=base_model.input, outputs=output)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Use the data generator
    train_gen = DataGenerator(folder_path='path/to/dataset', batch_size=16, img_size=(224, 224), augment=True)

    # Train the model
    model.fit(train_gen, epochs=10)
