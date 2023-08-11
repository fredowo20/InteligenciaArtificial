import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split

import tensorflow as tf

from sklearn.metrics import r2_score

## defining images's path dataframe
image_dir1 = Path('./iris-test/iris-setosa')
filepaths1 = pd.Series(list(image_dir1.glob(r'**/*.jpg')), name='Filepath').astype(str)
df_filepaths1 = pd.DataFrame(filepaths1)

image_dir2 = Path('./iris-test/iris-versicolour')
filepaths2 = pd.Series(list(image_dir2.glob(r'**/*.jpg')), name='Filepath').astype(str)
df_filepaths2 = pd.DataFrame(filepaths2)

image_dir3 = Path('./iris-test/iris-virginica')
filepaths3 = pd.Series(list(image_dir3.glob(r'**/*.jpg')), name='Filepath').astype(str)
df_filepaths3 = pd.DataFrame(filepaths3)

df_images = pd.concat([df_filepaths1, df_filepaths2, df_filepaths3], ignore_index=True)

## defining csv data dataframe
datos = pd.read_csv('./iris.csv')
df = datos["PetalLengthCm"]
df_datos = pd.DataFrame(datos)
df_datos.drop('Id', inplace=True, axis=1)
df_datos.drop('Species', inplace=True, axis=1)

## concatenate both dataframes
df_final = pd.concat([df_datos,df_images], axis=1)

print(df_final)

# X_train,X_test,Y_train,Y_test = train_test_split(df_images,df_datos, train_size=0.7)

# train_df = pd.concat([X_train,Y_train], axis=1)
# test_df = pd.concat([X_test,Y_test], axis=1)

## defining training and test dataframes
train_df, test_df = train_test_split(df_final, train_size=0.7, shuffle=False)

#train_df = np.asarray(train_df).astype(np.float32)

print(train_df)

## defining generation functions
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

## training, validation and testing parameters
columns = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']

train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col=columns,
    #y_col='PetalLengthCm',
    color_mode='rgb',
    class_mode='raw',
    batch_size=10,
    shuffle=True,
    seed=42,
    subset='training'
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col=columns,
    #y_col='PetalLengthCm',
    color_mode='rgb',
    class_mode='raw',
    batch_size=10,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col=columns,
    #y_col='PetalLengthCm',
    color_mode='rgb',
    class_mode='raw',
    batch_size=10,
    shuffle=False
)

## training process
inputs = tf.keras.Input(shape=(120, 120, 3))
x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(inputs)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(4, activation='linear')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='mse'
)

history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=5,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
)

## error metrics
predicted_parameters = np.squeeze(model.predict(test_images))
true_parameters = test_images.labels

rmse = np.sqrt(model.evaluate(test_images, verbose=0))
print("     Test RMSE: {:.5f}".format(rmse))

r2 = r2_score(true_parameters, predicted_parameters)
print("Test R^2 Score: {:.5f}".format(r2))

null_rmse = np.sqrt(np.sum((true_parameters - np.mean(true_parameters))**2) / len(true_parameters))
print("Null/Baseline Model Test RMSE: {:.5f}".format(null_rmse))


## prediction with test data
model.predict(test_images)

## prediction with test data with respective image
test_prediction = pd.DataFrame(model.predict(test_images), columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])

test_prediction.reset_index(drop=True, inplace=True)
test_df['Filepath'].reset_index(drop=True, inplace=True)

prediction_with_img = pd.concat([test_prediction, test_df['Filepath']], axis=1)

print(prediction_with_img)
