from keras.api.applications import ResNet50
from keras.api.layers import Dense, Flatten
from keras.api.models import Model
from keras.api.optimizers import Adam
from src.preprocessing import DataGenerator


# Define paths dynamically before training
annotations_path = r'D:\0-Code\PG\2_sem\0_Dyplom\ai-capstone-proj\kaggle\input\codebrim-original\original_dataset\annotations'
images_path = r'D:\0-Code\PG\2_sem\0_Dyplom\ai-capstone-proj\kaggle\input\codebrim-original\original_dataset\images'

# Define the Keras model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = Flatten()(base_model.output)
output = Dense(6, activation='sigmoid')(x)  # Multi-label classification (6 classes)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Initialize Data Generator
train_gen = DataGenerator(
    annotations_path=annotations_path,
    images_path=images_path,
    batch_size=16,
    img_size=(224, 224),
    augment=True
)

model.fit(train_gen, epochs=10)