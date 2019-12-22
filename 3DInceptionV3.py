from tensorflow.keras.layers import Conv3D, MaxPool3D, AveragePooling3D, GlobalAveragePooling3D, GlobalMaxPool3D

def Inception3D_v3(classes, shape, input_layer=None, include_top=True, pooling=None):

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 4  #Because now 4 dimensions besides batch
        
    # Ensure that the model takes into account
    # any potential predecessors of `input_layer`.
    if input_layer is not None:
      inputLayer = input_layer
    else:
      inputLayer = Input(shape=shape) #4 dimensions channels last
    
    model = Conv3D(32, kernel_size=(3, 3, 3), padding='valid')(inputLayer)
    model = Conv3D(64, kernel_size=(3, 3, 3), padding='valid')(model)
    model = MaxPool3D((3, 3, 3), strides=(2, 2, 2))(model)

    model = Conv3D(80, kernel_size=(1, 1, 1), padding='valid')(model)
    model = Conv3D(192, kernel_size=(3, 3, 3), padding='valid')(model)
    model = MaxPool3D((3, 3, 3), strides=(2, 2, 2))(model)

    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1 = Conv3D(64, kernel_size=(1, 1, 1))(model)

    branch5x5 = Conv3D(48, kernel_size=(1, 1, 1))(model)
    branch5x5 = Conv3D(64, kernel_size=(5, 5, 5), padding='same')(branch5x5) #Added same padding

    branch3x3dbl = Conv3D(64, kernel_size=(1, 1, 1))(model)
    branch3x3dbl = Conv3D(96, kernel_size=(3, 3, 3), padding='same')(branch3x3dbl) #Added same padding
    branch3x3dbl = Conv3D(96, kernel_size=(3, 3, 3), padding='same')(branch3x3dbl) #Added same padding

    branch_pool = AveragePooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(model)
    branch_pool = Conv3D(32, kernel_size=(1, 1, 1))(branch_pool)

    #problem with dimensions unless same padding is added as is above
    model = Concatenate(axis=channel_axis, name='mixed')([branch1x1, branch5x5, branch3x3dbl, branch_pool])


    # mixed 1: 35 x 35 x 256
    branch1x1 = Conv3D(64, kernel_size=(1, 1, 1))(model)

    branch5x5 = Conv3D(48, kernel_size=(1, 1, 1))(model)
    branch5x5 = Conv3D(64, kernel_size=(5, 5, 5), padding='same')(branch5x5) #Added same padding

    branch3x3dbl = Conv3D(64, kernel_size=(1, 1, 1))(model)
    branch3x3dbl = Conv3D(96, kernel_size=(3, 3, 3), padding='same')(branch3x3dbl) #Added same padding 
    branch3x3dbl = Conv3D(96, kernel_size=(3, 3, 3), padding='same')(branch3x3dbl) #Added same padding

    branch_pool = AveragePooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(model)
    branch_pool = Conv3D(64, kernel_size=(1, 1, 1))(branch_pool)

    #problem with dimensions unless same padding is added as is above
    model = Concatenate(axis=channel_axis, name='mixed1')([branch1x1, branch5x5, branch3x3dbl, branch_pool])

    # mixed 2: 35 x 35 x 256
    branch1x1 = Conv3D(64, kernel_size=(1, 1, 1))(model)

    branch5x5 = Conv3D(48, kernel_size=(1, 1, 1))(model)
    branch5x5 = Conv3D(64, kernel_size=(5, 5, 5), padding='same')(branch5x5) #Added same padding

    branch3x3dbl = Conv3D(64, kernel_size=(1, 1, 1))(model)
    branch3x3dbl = Conv3D(96, kernel_size=(3, 3, 3), padding='same')(branch3x3dbl) #Added same padding
    branch3x3dbl = Conv3D(96, kernel_size=(3, 3, 3), padding='same')(branch3x3dbl) #Added same padding

    branch_pool = AveragePooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(model)
    branch_pool = Conv3D(64, kernel_size=(1, 1, 1))(branch_pool)

    #problem with dimensions unless added same padding
    model = Concatenate(axis=channel_axis, name='mixed2')([branch1x1, branch5x5, branch3x3dbl, branch_pool])


    # mixed 3: 17 x 17 x 768
    branch3x3 = Conv3D(384, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='valid')(model)

    branch3x3dbl = Conv3D(64, kernel_size=(1, 1, 1))(model)
    branch3x3dbl = Conv3D(96, kernel_size=(3, 3, 3), padding='same')(branch3x3dbl) #Added same padding
    branch3x3dbl = Conv3D(96, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='valid')(branch3x3dbl)

    branch_pool = MaxPool3D((3, 3, 3), strides=(2, 2, 2))(model)
    
    #problem with dimensions without same padding
    model = Concatenate(axis=channel_axis, name='mixed3')([branch3x3, branch3x3dbl, branch_pool])

    # mixed 4: 17 x 17 x 768
    branch1x1 = Conv3D(192, kernel_size=(1, 1, 1))(model)

    branch7x7 = Conv3D(128, kernel_size=(1, 1, 1))(model) #Unsure on transformations from here because 3d
    branch7x7 = Conv3D(128, kernel_size=(1, 7, 1), padding='same')(branch7x7) #Added same padding
    branch7x7 = Conv3D(192, kernel_size=(7, 1, 1), padding='same')(branch7x7) #Added same padding

    branch7x7dbl = Conv3D(128, kernel_size=(1, 1, 1))(model)
    branch7x7dbl = Conv3D(128, kernel_size=(7, 1, 1), padding='same')(branch7x7dbl) #Added same padding  
    branch7x7dbl = Conv3D(128, kernel_size=(1, 7, 1), padding='same')(branch7x7dbl) #Added same padding
    branch7x7dbl = Conv3D(128, kernel_size=(7, 1, 1), padding='same')(branch7x7dbl) #Added same padding
    branch7x7dbl = Conv3D(192, kernel_size=(1, 7, 1), padding='same')(branch7x7dbl) #Added same padding

    branch_pool = AveragePooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(model)
    branch_pool = Conv3D(192, kernel_size=(1, 1, 1))(branch_pool)
    
    #problem with dimensions without same padding
    model = Concatenate(axis=channel_axis, name='mixed4')([branch1x1, branch7x7, branch7x7dbl, branch_pool])

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = Conv3D(192, kernel_size=(1, 1, 1))(model)

        branch7x7 = Conv3D(160, kernel_size=(1, 1, 1))(model)
        branch7x7 = Conv3D(160, kernel_size=(1, 7, 1), padding='same')(branch7x7) #Added same padding
        branch7x7 = Conv3D(192, kernel_size=(7, 1, 1), padding='same')(branch7x7) #Added same padding

        branch7x7dbl = Conv3D(160, kernel_size=(1, 1, 1))(model)
        branch7x7dbl = Conv3D(160, kernel_size=(7, 1, 1), padding='same')(branch7x7dbl) #Added same padding
        branch7x7dbl = Conv3D(160, kernel_size=(1, 7, 1), padding='same')(branch7x7dbl) #Added same padding
        branch7x7dbl = Conv3D(160, kernel_size=(7, 1, 1), padding='same')(branch7x7dbl) #Added same padding
        branch7x7dbl = Conv3D(192, kernel_size=(1, 7, 1), padding='same')(branch7x7dbl) #Added same padding

        branch_pool = AveragePooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(model)
        branch_pool = Conv3D(192, kernel_size=(1, 1, 1))(branch_pool)

        #problem with dimensons without same padding
        model = Concatenate(axis=channel_axis, name=('mixed'+str(5 + i)))([branch1x1, branch7x7, branch7x7dbl, branch_pool])
        #model = concatenate(inputs=[branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis, name='mixed'+str(5 + i)) #([branch1x1, branch7x7, branch7x7dbl, branch_pool])

    # mixed 7: 17 x 17 x 768
    branch1x1 = Conv3D(192, kernel_size=(1, 1, 1))(model)

    branch7x7 = Conv3D(192, kernel_size=(1, 1, 1))(model)
    branch7x7 = Conv3D(192, kernel_size=(1, 7, 1), padding='same')(branch7x7) #Added same padding 
    branch7x7 = Conv3D(192, kernel_size=(7, 1, 1), padding='same')(branch7x7) #Added same padding

    branch7x7dbl = Conv3D(192, kernel_size=(1, 1, 1))(model)
    branch7x7dbl = Conv3D(192, kernel_size=(7, 1, 1), padding='same')(branch7x7dbl) #Added same padding 
    branch7x7dbl = Conv3D(192, kernel_size=(1, 7, 1), padding='same')(branch7x7dbl) #Added same padding 
    branch7x7dbl = Conv3D(192, kernel_size=(7, 1, 1), padding='same')(branch7x7dbl) #Added same padding 
    branch7x7dbl = Conv3D(192, kernel_size=(1, 7, 1), padding='same')(branch7x7dbl) #Added same padding 

    branch_pool = AveragePooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(model)
    branch_pool = Conv3D(192, kernel_size=(1, 1, 1))(branch_pool)

    #problem with dimension without same padding 
    model = Concatenate(axis=channel_axis, name='mixed7')([branch1x1, branch7x7, branch7x7dbl, branch_pool])

    # mixed 8: 8 x 8 x 1280
    branch3x3 = Conv3D(192, kernel_size=(1, 1, 1))(model)
    branch3x3 = Conv3D(320, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='valid')(branch3x3)

    branch7x7x3 = Conv3D(192, kernel_size=(1, 1, 1))(model)
    branch7x7x3 = Conv3D(192, kernel_size=(1, 7, 1), padding='same')(branch7x7x3) #Added same padding
    branch7x7x3 = Conv3D(192, kernel_size=(7, 1, 1), padding='same')(branch7x7x3) #Added same padding
    branch7x7x3 = Conv3D(192, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='valid')(branch7x7x3)

    branch_pool = MaxPool3D((3, 3, 3), strides=(2, 2, 2))(model)
    #problem with dimesions without same padding
    model = Concatenate(axis=channel_axis, name='mixed8')([branch3x3, branch7x7x3, branch_pool])
    
    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = Conv3D(320, kernel_size=(1, 1, 1))(model)

        branch3x3 = Conv3D(384, kernel_size=(1, 1, 1))(model)
        branch3x3_1 = Conv3D(384, kernel_size=(1, 3, 1), padding='same')(branch3x3) #Added same padding
        branch3x3_2 = Conv3D(384, kernel_size=(3, 1, 1), padding='same')(branch3x3) #Added same padding
        #problem with dimension without same
        branch3x3 = Concatenate(axis=channel_axis, name='mixed9_' + str(i))([branch3x3_1, branch3x3_2])

        branch3x3dbl = Conv3D(448, kernel_size=(1, 1, 1))(model)
        branch3x3dbl = Conv3D(384, kernel_size=(3, 3, 3), padding='same')(branch3x3dbl) #Added same padding
        branch3x3dbl_1 = Conv3D(384, kernel_size=(1, 3, 1), padding='same')(branch3x3dbl) #Added same padding
        branch3x3dbl_2 = Conv3D(384, kernel_size=(3, 1, 1), padding='same')(branch3x3dbl) #Added same padding
        #problem with dimension without same
        branch3x3dbl = Concatenate(axis=channel_axis)([branch3x3dbl_1, branch3x3dbl_2])

        branch_pool = AveragePooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(model)
        branch_pool = Conv3D(192, kernel_size=(1, 1, 1))(branch_pool)
        #problem with dimension without same
        model = Concatenate(axis=channel_axis, name='mixed' + str(9 + i))([branch1x1, branch3x3, branch3x3dbl, branch_pool])

    if include_top:
        # Classification block
        model = GlobalAveragePooling3D(name='avg_pool')(model)
        print(model)
        model = Dense(classes, activation='softmax', name='predictions')(model)
    else:
        if pooling == 'avg':
            model = GlobalAveragePooling3D()(model)
        elif pooling == 'max':
            model = GlobalMaxPool3D()(model)

    # Create model.
    model = Model(inputs=inputLayer, outputs=model, name='inception_v3_lite')
    return model
