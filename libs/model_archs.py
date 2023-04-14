import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, regularizers

###########################################
# Basic ResNet Building Block
def resnet_layer(inputs,
				num_filters=16,
				kernel_size=3,
				strides=1,
				activation='relu',
				batch_normalization=True,
                conv_first=True):
    
    conv=layers.Conv2D(num_filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(1e-4))
    x=inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
    else:
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
        x = conv(x)
    return x

# ResNet V1 architecture
def resnet_v1(input_shape, depth, num_classes):
	if (depth - 2) % 6 != 0:
		raise ValueError('depth should be 6n + 2 (eg 20, 32, 44 in [a])')
	# Start model definition.
	num_filters = 16
	num_res_blocks = int((depth - 2) / 6)

	inputs = layers.Input(shape=input_shape)
	x = resnet_layer(inputs=inputs)
	# Instantiate the stack of residual units
	for stack in range(3):
		for res_block in range(num_res_blocks):
			strides = 1
			if stack > 0 and res_block == 0: # first layer but not first stack
				strides = 2 # downsample
			y = resnet_layer(inputs=x,
							num_filters=num_filters,
							strides=strides)
			y = resnet_layer(inputs=y,
							num_filters=num_filters,
							activation=None)
			if stack > 0 and res_block == 0: # first layer but not first stack
				# linear projection residual shortcut connection to match
				# changed dims
				x = resnet_layer(inputs=x,
								num_filters=num_filters,
								kernel_size=1,
								strides=strides,
								activation=None,
								batch_normalization=False)
			x = keras.layers.add([x, y])
			x = layers.Activation('relu')(x)
		num_filters *= 2

	# Add classifier on top.
	# v1 does not use BN after last shortcut connection-ReLU
	x = layers.AveragePooling2D(pool_size=8)(x)
	y = layers.Flatten()(x)
	outputs = layers.Dense(num_classes,
                            activation='softmax',
                            kernel_initializer='he_normal')(y)

	# Instantiate model.
	model = models.Model(inputs=inputs, outputs=outputs, name='resnet_v1')
	return model


                 
# ResNet V2 architecture
def resnet_v2(input_shape, depth, num_classes):
	if (depth - 2) % 9 != 0:
		raise ValueError('depth should be 9n + 2 (eg 56 or 110 in [b])')
	# Start model definition.
	num_filters_in = 16
	num_res_blocks = int((depth - 2) / 9)

	inputs = layers.Input(shape=input_shape)
	# v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
	x = resnet_layer(inputs=inputs,
					num_filters=num_filters_in,
					conv_first=True)

	# Instantiate the stack of residual units
	for stage in range(3):
		for res_block in range(num_res_blocks):
			activation = 'relu'
			batch_normalization = True
			strides = 1
			if stage == 0:
				num_filters_out = num_filters_in * 4
				if res_block == 0: # first layer and first stage
					activation = None
					batch_normalization = False
			else:
				num_filters_out = num_filters_in * 2
				if res_block == 0: # first layer but not first stage
					strides = 2 # downsample

			# bottleneck residual unit
			y = resnet_layer(inputs=x,
							num_filters=num_filters_in,
							kernel_size=1,
							strides=strides,
							activation=activation,
							batch_normalization=batch_normalization,
							conv_first=False)
			y = resnet_layer(inputs=y,
							num_filters=num_filters_in,
							conv_first=False)
			y = resnet_layer(inputs=y,
							num_filters=num_filters_out,
							kernel_size=1,
							conv_first=False)
			if res_block == 0:
				# linear projection residual shortcut connection to match
				# changed dims
				x = resnet_layer(inputs=x,
								num_filters=num_filters_out,
								kernel_size=1,
								strides=strides,
								activation=None,
								batch_normalization=False)
			x = keras.layers.add([x, y])

		num_filters_in = num_filters_out

	# Add classifier on top.
	# v2 has BN-ReLU before Pooling
	x = layers.BatchNormalization()(x)
	x = layers.Activation('relu')(x)
	x = layers.AveragePooling2D(pool_size=8)(x)
	y = layers.Flatten()(x)
	outputs = layers.Dense(num_classes,
					activation='softmax',
					kernel_initializer='he_normal')(y)

	# Instantiate model.
	model = models.Model(inputs=inputs, outputs=outputs, name='resnet_v2')
	return model

                 
###########################################
def vgg16E(input_shape, nb_classes):
    # https://www.geeksforgeeks.org/cifar-10-image-classification-in-tensorflow/
    return models.Sequential(
        [
            # 1st Conv Block
            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            
            # 2nd Conv Block
            layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            
            # 3rd Conv Block
            layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            
            # 4th Conv Block
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            
            # 5th Conv Block
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            layers.Dropout(0.4),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(units=4096, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(units=4096, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(units=1000, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(units=nb_classes, activation='softmax')
        ],
        name='vgg16E',
    )

def vgg16D(input_shape, nb_classes):
    # https://www.geeksforgeeks.org/cifar-10-image-classification-in-tensorflow/
    return models.Sequential(
        [
            # 1st Conv Block
            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            
            # 2nd Conv Block
            layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            
            # 3rd Conv Block
            layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            
            # 4th Conv Block
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            
            # 5th Conv Block
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            layers.Dropout(0.4),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(units=512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(units=nb_classes, activation='softmax')
        ],
        name='vgg16D',
    )


def vgg16C(input_shape, nb_classes):
    # https://www.geeksforgeeks.org/cifar-10-image-classification-in-tensorflow/
    return models.Sequential(
        [
            # 1st Conv Block
            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            layers.Dropout(0.2),
            
            # 2nd Conv Block
            layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            layers.Dropout(0.3),
            
            # 3rd Conv Block
            layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            layers.Dropout(0.4),
            
            # 4th Conv Block
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            layers.Dropout(0.4),
            
            # 5th Conv Block
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            layers.Dropout(0.4),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(units=1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(units=1024, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(units=nb_classes, activation='softmax')
        ],
        name='vgg16C',
    )
def vgg16B(input_shape, nb_classes):
    # https://www.geeksforgeeks.org/cifar-10-image-classification-in-tensorflow/
    return models.Sequential(
        [
            # 1st Conv Block
            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            
            # 2nd Conv Block
            layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            
            # 3rd Conv Block
            layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            
            # 4th Conv Block
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            
            # 5th Conv Block
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(units=1024, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(units=1024, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(units=nb_classes, activation='softmax')
        ],
        name='vgg16B',
    )
def vgg16A(input_shape, nb_classes):
    # https://www.geeksforgeeks.org/cifar-10-image-classification-in-tensorflow/
    return models.Sequential(
        [
            # 1st Conv Block
            layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            
            # 2nd Conv Block
            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            
            # 3rd Conv Block
            layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            
            # # 4th Conv Block
            # layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            # layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            # layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            # layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            # # 5th Conv Block
            # layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            # layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            # layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
            # layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(units=1024, activation='relu'),
            layers.Dropout(0.2),
            # layers.Dense(units=4096, activation='relu'),
            layers.Dense(units=nb_classes, activation='softmax')
        ],
        name='vgg16A',
    )

def lenetA(input_shape, nb_classes):
    return models.Sequential(
        [
            layers.Conv2D(
                filters=6,
                kernel_size=(5, 5),
                strides=(1, 1),
                activation='relu',
                input_shape=input_shape,
            ),
            layers.AveragePooling2D(
                pool_size=(2, 2),
                strides=(1, 1),
                padding='valid',
            ),
            layers.Conv2D(
                filters=16,
                kernel_size=(5, 5),
                strides=(1, 1),
                activation='relu',
                padding='valid',
            ),
            layers.AveragePooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='valid',
            ),
            layers.Conv2D(
                filters=120,
                kernel_size=(5, 5),
                strides=(1, 1),
                activation='relu',
                padding='valid',
            ),
            layers.Flatten(),
            layers.Dense(84, activation='relu'),
            layers.Dense(nb_classes, activation='softmax'),
        ],
        name='lenetA',
    )

def fcA(input_shape, nb_classes):
    return models.Sequential(
            [
                layers.Input(shape=input_shape),
                layers.Flatten(),
                layers.Dense(32, activation='relu'),
                layers.Dense(16, activation='relu'),
                layers.Dense(nb_classes, activation='softmax'),
            ],
            name='fcA',
    )