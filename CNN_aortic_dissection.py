#Programmed by Tobias Spindelboeck
#All rights reserved
#CNN ver 1.10



""" --- Importing Libraries --- """

from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras.layers.merge import concatenate
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras import backend as K
import time


#seed(1)
tf.random.set_seed(1) #1

#Begin of time measuring.
start = time.time()



""" --- Define Variables --- """

#CNN wil be trained.
train 	 = False

#Data will be evaluated.
evaluate = False

#Name of saved model.
save_path: str = r"CNN_1-10.h5"

#Path of the trainigs and evaluation data.
sensor_data_path: str = r"data"

#Saves an image of the model architecture. May cause an error, if the dedicated library is not installed on windows.
plot_model_structure = True

#Defines how much is taken for the training and for the testing from the incoming data.
split_ratio = 0.5   #Default 50%

#Maximum number of epochs.
epoch_size  = 200   #Default 200

#Determines the size of the mini batches.
size_batch  = 34    #Default 34 (sligtly better results than 32).

#The learning rate is very important.
learn_rate  = 0.09  #Default 0.09 - 0.03

#Standard deviation for noise which is added to the input data.
add_noise   = 0.5   #Default 0.5.

#How many channels are used for the convolution.
sensors     = 5     #Default 5.    

#The order of mixing the input data befor splitting. 
rand_state  = 37    #Default 37.

#During training a small portion is taken for a validation, but here the test data is used.
vald_split  = 0.11 #Default 11%. 

#Number of epochs with no improvement for early stopping.
var_patience = 3    #Default 3.



""" --- Data Splitting --- """

def data(x_1, x_2, ratio, noise, sensor, random):

    #The data is importet
    x_healthy = np.loadtxt(x_1, delimiter=',')
    x_ill     = np.loadtxt(x_2, delimiter=',')

    #The labels for the data are created. Zero for healthy and one for diseased.
    y_healthy = np.zeros(int(len(x_healthy[1])/sensor))
    y_ill     = np.ones(int(len(x_ill[1])/sensor))
    
    #Both data and labels are combined to one.
    x_data = np.concatenate((x_healthy,x_ill), axis=1)
    y_data = np.concatenate((y_healthy,y_ill), axis=0)

    #Some noise is added to the data. Noise defines the value for the std.
    noise = np.random.normal(0,noise, x_data.shape) 
    x_data = x_data + noise

    #The input data is reshaped in order to have the right dimensions for the CNN.
    x_data = np.reshape(x_data.transpose(),(int(len(x_data.transpose())/sensor),x_healthy.shape[0],sensor))
	
    #The data set is split into training and test data.
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=ratio, random_state=random)

    return X_train, X_test, y_train, y_test



""" --- CNN Model --- """

def CNN_model(var_shape):

    #How many filters/feature maps should be used.
    var_filters     = 9     #Default 9.

    #Defines the length of the convolution window.
    var_kernel_size = 3     #Default 3.

    #Activation function for the convolution.
    var_activation  = r'relu'   #Default ReLu.

    #Percentage of dropped out neurons.
    var_dropout     = 0.2 #Default 0.2.

    #Length of the pooling window.
    var_pool_size   = 2 #Default 2.


    #Injector 1
    #The data of the first injector is imported.
    input_inj_1 = Input(shape=var_shape)

    #1d convolution is performed.
    conv1_1 = Conv1D(filters=var_filters, kernel_size=var_kernel_size, activation=var_activation)(input_inj_1)

    #Some neurons are dropped out.
    drop1_1 = Dropout(var_dropout)(conv1_1)

    #Pooling layer with MaxPooling is applied.
    pool1_1 = MaxPooling1D(pool_size=var_pool_size)(drop1_1)

    #Feature maps are flattened.
    flat1_1 = Flatten()(pool1_1)


    #Sensor 2
    input_inj_2 = Input(shape=var_shape)
    conv1_2 = Conv1D(filters=var_filters, kernel_size=var_kernel_size, activation=var_activation)(input_inj_2)
    drop1_2 = Dropout(var_dropout)(conv1_2)
    pool1_2 = MaxPooling1D(pool_size=var_pool_size)(drop1_2)
    flat1_2 = Flatten()(pool1_2)


    #Sensor 3
    input_inj_3 = Input(shape=var_shape)
    conv1_3 = Conv1D(filters=var_filters, kernel_size=var_kernel_size, activation=var_activation)(input_inj_3)
    drop1_3 = Dropout(var_dropout)(conv1_3)
    pool1_3 = MaxPooling1D(pool_size=var_pool_size)(drop1_3)
    flat1_3 = Flatten()(pool1_3)


    #The three injectors are combined into one.
    merged = concatenate([flat1_1, flat1_2, flat1_3])
	
    #The size of the fully connected neurons is reduced.
    dense1 = Dense(20, activation='relu')(merged)

    #The layer is reduced to only two classes - healthy and dissected
    #with softmax as output activation
    output_layer = Dense(2, activation='softmax')(dense1)

    #Groups the layers into an object.
    model = Model(inputs=[input_inj_1, input_inj_2, input_inj_3], outputs=output_layer)

    #Model is compiled.
    model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	
    #For a nice output of the progress.
    print(model.summary())

    #Generates an image of the network structure.
    if plot_model_structure:
        plot_model(model, show_shapes=True, to_file='fancy_image-test.png')


    return model



""" --- Load Data --- """

#The file path for the input data is assigned.
sensor_h_1, sensor_i_1 = r'data\data_health_1.txt', r'data\data_ill_1.txt'
sensor_h_2, sensor_i_2 = r'data\data_health_2.txt', r'data\data_ill_2.txt'
sensor_h_3, sensor_i_3 = r'data\data_health_3.txt', r'data\data_ill_3.txt'


#The function "data" is called and the data is imported, labeled and splitted into test and training.
X_train_1, X_test_1, y_train_1, y_test_1 = data(sensor_h_1, sensor_i_1, split_ratio, add_noise, sensors, rand_state)
X_train_2, X_test_2, y_train_2, y_test_2 = data(sensor_h_2, sensor_i_2, split_ratio, add_noise, sensors, rand_state)
X_train_3, X_test_3, y_train_3, y_test_3 = data(sensor_h_3, sensor_i_3, split_ratio, add_noise, sensors, rand_state)



""" --- Train Model --- """

if train:

    #Evaluates the input shape.
    shape_of_input = X_train_1.shape[1:]

    #Calls the CNN function.
    model = CNN_model(shape_of_input)

    #The learning rate and the Nesterov momentum is set.
    optimizers.SGD(lr=learn_rate, momentum=0.0, decay=0.0, nesterov=True)

    #Test data is validation data.
    #history = model.fit([X_train_1, X_train_2, X_train_3], y_train_1, validation_split=vald_split, epochs=epoch_size, batch_size=size_batch)

    #Default Keras EarlyStopping.
    early_stopping_monitor = EarlyStopping(patience=var_patience)

    #The network is trained with the trainigs data and test data for evaluation.
    history = model.fit([X_train_1, X_train_2, X_train_3], y_train_1, validation_data=([X_test_1, X_test_2, X_test_3], y_test_1), epochs=epoch_size, batch_size=size_batch, callbacks=[early_stopping_monitor])
	
    #The trained network is saved.
    model.save(save_path)

    #End of time measuring and output.
    end = time.time()
    print("Time to train:",end - start)

    
    #Plot of Accuracy and Loss per epoch
    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Test'], loc='lower right')
    plt.grid(True)

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Test'], loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    

    #print(model.count_params())



""" --- Evaluate Model --- """

if evaluate:

    #Evaluates the input shape.
    shape_of_input_2 = X_train_1.shape[1:]

    #Calls the CNN function.
    model = CNN_model(shape_of_input_2)

    #The learning rate and the Nesterov momentum is set.
    optimizers.SGD(lr=learn_rate, momentum=0.0, decay=0.0, nesterov=True)

    #The network is loaded.
    model = load_model(save_path)

    #The overall accuracy and loss is evaluated and printed.
    loss, accuracy = model.evaluate([X_test_1, X_test_2, X_test_3], y_test_1, verbose=2)

    print("Accuracy of test-data:", accuracy*100)