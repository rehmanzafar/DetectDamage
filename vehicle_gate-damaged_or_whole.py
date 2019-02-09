import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras import optimizers
from keras.applications import VGG16
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

sns.set_style('whitegrid')

# Set dataset path
location = 'car-damage-dataset/data1a'
train_data_dir = location+'/training'
validation_data_dir = location+'/validation'

# Set dimensions of your input data
img_width, img_height = 224, 224
epochs = 50
batch_size = 20

# Compute number of samples
train_samples = [len(os.listdir(train_data_dir+'/'+i)) for i in sorted(os.listdir(train_data_dir))]
nb_train_samples = sum(train_samples)
train_labels = np.array([0] * train_samples[0] + [1] * train_samples[1])
validation_samples = [len(os.listdir(validation_data_dir+'/'+i)) for i in sorted(os.listdir(validation_data_dir))]
nb_validation_samples = sum(validation_samples)
validation_labels = np.array([0] * validation_samples[0] + [1] * validation_samples[1])

# Define your VGG16 Model
def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape='th'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2)))

    # model.add(Flatten())
    # model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1000, activation='softmax'))

    # model.add(Flatten(input_shape=model.output_shape[1:]))
    # model.add(Dense(4096, activation='relu', W_regularizer=l2(0.01)))
    # model.add(Dropout(0.5))
    # model.add(Dense(4096, activation='relu', W_regularizer=l2(0.01)))
    # model.add(Dropout(0.5))
    # model.add(Dense(1000, activation='sigmoid'))

    if weights_path:
        model.load_weights(weights_path)
    return model
def train_model(weights='imagenet'):
    prev_model = VGG16(weights=weights, include_top=False, input_shape = (3,img_height,img_width))
    top_model = Sequential()

    # Set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in prev_model.layers[:25]:
        layer.trainable = False

    top_model.add(prev_model)
    top_model.add(Flatten(input_shape=prev_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    top_model.compile(optimizers.SGD(lr=0.0001, momentum=0.9),
              loss='binary_crossentropy', metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    print(top_model.summary())

    # Fine-tune the model
    top_model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=epochs,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)

    # Save fine-tuned model
    top_model.save('fine_tunned_model.h5')

    return top_model, None
def testModel():
    model = load_model('fine_tunned_model.h5')
    im_original = cv2.resize(cv2.imread('/media/rehman/5280BFF380BFDBA3/Dr Talal/car-damage-detective/car-damage-dataset/data1a/training/00-damage/0013.JPEG'), (img_height, img_width))
    im = im_original.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    im_converted = cv2.cvtColor(im_original, cv2.COLOR_BGR2RGB)
    plt.imshow(im_converted)

    pred = model.predict(im)
    plt.plot(pred.ravel())

    if pred[0][0] <=.5:

        print ("Validation complete - proceed to location and severity determination")
    else:
        print ("Are you sure that your car is damaged? Please submit another picture of the damage.")
        print ("Hint: Try zooming in/out, using a different angle or different lighting")
def evaluate_binary_model(directory, labels):
    model = load_model('fine_tunned_model.h5')
    datagen = ImageDataGenerator(rescale=1./255)

    generator = datagen.flow_from_directory(directory,
                                target_size=(img_height, img_width),
                                batch_size=10,
                                class_mode='binary', # categorical for multiclass
                                shuffle=False)

    predictions = model.predict_generator(generator, len(labels))

    # For multiclass use: pred_labels = np.argmax(predictions, axis=1)

    pred_labels = [0 if i <0.5 else 1 for i in predictions]

    print('===================================================')
    print(classification_report(validation_labels, pred_labels))
    print('===================================================')
    cm = confusion_matrix(validation_labels, pred_labels)
    sns.heatmap(cm, annot=True, fmt='g');
    plt.show()

# Utility methods for performance plotting
def print_best_model_results(model_hist):
    best_epoch = np.argmax(model_hist['val_acc'])
    print('epoch:', best_epoch+1,     ', val_acc:', model_hist['val_acc'][best_epoch],     ', val_loss:', model_hist['val_loss'][best_epoch])
def plot_metrics(hist, stop=50):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

    axes = axes.flatten()

    axes[0].plot(range(stop), hist['acc'], label='Training', color='#FF533D')
    axes[0].plot(range(stop), hist['val_acc'], label='Validation', color='#03507E')
    axes[0].set_title('Accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(loc='lower right')

    axes[1].plot(range(stop), hist['loss'], label='Training', color='#FF533D')
    axes[1].plot(range(stop), hist['val_loss'], label='Validation', color='#03507E')
    axes[1].set_title('Loss')
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(loc='upper right')

    plt.tight_layout();

    print("Best Model:")
    print_best_model_results(hist)
def plot_acc_metrics(hist1, hist2, stop=50):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4.25,6))

    axes = axes.flatten()

    axes[0].plot(range(stop), hist1['acc'], label='Training', color='#FF533D')
    axes[0].plot(range(stop), hist1['val_acc'], label='Validation', color='#03507E')
    axes[0].set_title('Training')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(loc='lower right')

    axes[1].plot(range(stop), hist2['acc'], label='Training', color='#FF533D')
    axes[1].plot(range(stop), hist2['val_acc'], label='Validation', color='#03507E')
    axes[1].set_title('Fine-tuning')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(loc='lower right')

    plt.tight_layout();

#heatmap_labels = ['Damaged', 'Whole']
# sns.heatmap(cm, annot=True, annot_kws={"size": 16},
#             fmt='g', cmap='OrRd', xticklabels=heatmap_labels, yticklabels=heatmap_labels);

# sns.heatmap(cm, annot=True, annot_kws={"size": 16},
#             fmt='g', cmap='Blues', xticklabels=heatmap_labels, yticklabels=heatmap_labels);


if __name__ == '__main__':
    train = train_model()
    test = testModel()
    evaluate_binary_model(validation_data_dir, validation_labels)