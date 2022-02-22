"""
List of decoding algorithms

reference: https://github.com/KordingLab/Neural_Decoding/tree/master/Neural_Decoding
"""

# import required packages
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from numpy.linalg import inv, pinv
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Layer, InputSpec, Dense, Dropout, SpatialDropout1D, LSTM
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.python.keras.utils.conv_utils import conv_output_length
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
from .generator import data_generator
from .preprocess import flatten_list
from .metrics import compute_rmse

def rmse(y_true, y_pred):
    """
    Define root mean squared error (RMSE) loss function
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

class PredictHistory(Callback):
    """
    Callback for printing validation loss after each epoch
    """
    def __init__(self, X_valid, Y_valid):
        self.X_valid = X_valid
        self.Y_valid = Y_valid
        self.i_epoch = 0
        self.pred_hist = []
    def on_epoch_end(self, epoch, logs={}):
        Y_valid_predict = self.model.predict(self.X_valid)
        rmse_valid = compute_rmse(self.Y_valid, Y_valid_predict)
        self.pred_hist.append(np.mean(rmse_valid))
        self.i_epoch += 1
        print("Epoch %d >> valid loss: %.4f"%(self.i_epoch, np.mean(rmse_valid)))    
        
class LSTMDecoder:
    def __init__(self, num_layers=1, units=100, batch_size=32, epochs=5, dropout=0, stateful=False, shuffle=True, verbose=0):
        self.num_layers = num_layers
        self.units = units
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout
        self.stateful = stateful
        self.shuffle = shuffle
        self.verbose = verbose
    
    def compile(self, params):
        self.num_layers = params['num_layers']
        self.units = params['units']
        self.batch_size = params['batch_size']   
        self.timesteps = params['timesteps']
        self.input_dim = params['input_dim']
        self.output_dim = params['output_dim']
        self.optimizer = params['optimizer']
        self.seed = params['seed']
        
        if self.optimizer == 'RMSprop':
            optim = RMSprop(lr=params['lrate'])
        else:
            optim = Adam(lr=params['lrate'])
        
        # Create new model or load existing model
        if params['load']:
            print("Loading existing model " +params['load_name'])
            model = load_model(params['load_name'], custom_objects={'rmse':rmse})
        else:
            model = Sequential()
            if self.num_layers > 1:
                for i in range(self.num_layers-1):
                    model.add(LSTM(self.units, input_shape=(self.timesteps, self.input_dim), dropout=self.dropout, stateful=self.stateful, return_sequences=True))
                model.add(LSTM(self.units,input_shape=(self.timesteps, self.input_dim), dropout=self.dropout, stateful=self.stateful))
            else:
                model.add(LSTM(self.units, input_shape=(self.timesteps, self.input_dim), dropout=self.dropout, stateful=self.stateful))
            if self.dropout > 0.: 
                model.add(Dropout(self.dropout))    
            model.add(Dense(self.output_dim))
            # Compile model
            model.compile(optimizer=optim, loss=rmse) #Set loss function and optimizer
            
        #print(model.summary())
        # Print parameter count
        num_params = model.count_params()
        print('# network parameters: ' + str(num_params))
        self.model = model
        return model
        
    def fit(self, X_train, Y_train, X_valid, Y_valid, params):
        self.epochs = params['epochs']
        self.dropout = params['dropout']
        self.stateful = params['stateful']
        self.shuffle = params['shuffle']
        self.verbose = params['verbose']
        self.fit_gen = params['fit_gen']
        # Fit model
        loss_train = []
        loss_valid = []
        predict_history = PredictHistory(X_valid, Y_valid)
        if params['retrain']:
            if self.stateful:
                if self.fit_gen:
                    train_generator = data_generator(X_train, Y_train, self.batch_size, self.shuffle)
                    for i in range(self.epochs):                        
                        hist = self.model.fit_generator(generator=train_generator,
                                                        validation_data=(X_valid, Y_valid),
                                                        epochs=1, shuffle=self.shuffle, verbose=self.verbose,
                                                        callbacks=[predict_history])
                        loss_train.append(hist.history['loss'])
                        loss_valid.append(hist.history['val_loss'])
                        self.model.reset_states()
                else:
                    for i in range(self.epochs):
                        hist = self.model.fit(X_train, Y_train,
                                              validation_data=(X_valid, Y_valid),
                                              batch_size=self.batch_size, epochs=1, verbose=self.verbose, shuffle=self.shuffle,
                                              callbacks=[predict_history])
                        loss_train.append(hist.history['loss'])
                        loss_valid.append(hist.history['val_loss'])
                        self.model.reset_states()
            else:
                hist = self.model.fit(X_train, Y_train,
                                 validation_data=(X_valid, Y_valid),
                                 batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose, shuffle=self.shuffle,
                                 callbacks=[predict_history])
                loss_train.append(hist.history['loss'])
                loss_valid.append(hist.history['val_loss'])
            loss_train = flatten_list(loss_train)
            loss_valid = flatten_list(loss_valid)
        
        loss_predict = predict_history.pred_hist
        if params['save']:
            self.model.save(params['save_name'])
            
        fit_out = {'loss_train':loss_train,
                    'loss_valid':loss_valid,
                    'loss_predict':loss_predict,
                    'num_params':self.model.count_params()}                    
        return fit_out
    
    def predict(self, X_test):
        #self.model.reset_states()
        y_pred = self.model.predict(X_test, batch_size=self.batch_size, verbose=self.verbose) #Make predictions
        return y_pred

def _dropout(x, level, noise_shape=None, seed=None):
    x = K.dropout(x, level, noise_shape, seed)
    x *= (1. - level) # compensate for the scaling by the dropout
    return x

class QRNN(Layer):
    '''Quasi RNN
    # Arguments
        units: dimension of the internal projections and the final output.
    # References
        - [Quasi-recurrent Neural Networks](http://arxiv.org/abs/1611.01576)
    '''
    def __init__(self, units, window_size=2, stride=1,
                 return_sequences=False, go_backwards=False, 
                 stateful=False, unroll=False, activation='tanh',
                 kernel_initializer='uniform', bias_initializer='zero',
                 kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, 
                 dropout=0, use_bias=True, input_dim=None, input_length=None,
                 **kwargs):
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll

        self.units = units 
        self.window_size = window_size
        self.strides = (stride, 1)

        self.use_bias = use_bias
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = dropout
        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(QRNN, self).__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.input_dim = input_shape[2]
        self.input_spec = InputSpec(shape=(batch_size, None, self.input_dim))
        self.state_spec = InputSpec(shape=(batch_size, self.units))

        self.states = [None]
        if self.stateful:
            self.reset_states()

        kernel_shape = (self.window_size, 1, self.input_dim, self.units * 3)
        self.kernel = self.add_weight(name='kernel',
                                      shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(name='bias', 
                                        shape=(self.units * 3,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

        self.built = True

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        length = input_shape[1]
        if length:
            length = conv_output_length(length + self.window_size - 1,
                                        self.window_size, 'valid',
                                        self.strides[0])
        if self.return_sequences:
            return (input_shape[0], length, self.units)
        else:
            return (input_shape[0], self.units)

    def compute_mask(self, inputs, mask):
        if self.return_sequences:
            return mask
        else:
            return None

    def get_initial_states(self, inputs):
        # build an all-zero tensor of shape (samples, units)
        initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        initial_state = K.tile(initial_state, [1, self.units])  # (samples, units)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def reset_states(self, states=None):
        if not self.stateful:
            raise AttributeError('Layer must be stateful.')
        if not self.input_spec:
            raise RuntimeError('Layer has never been called '
                               'and thus has no states.')

        batch_size = self.input_spec.shape[0]
        if not batch_size:
            raise ValueError('If a QRNN is stateful, it needs to know '
                             'its batch size. Specify the batch size '
                             'of your input tensors: \n'
                             '- If using a Sequential model, '
                             'specify the batch size by passing '
                             'a `batch_input_shape` '
                             'argument to your first layer.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a '
                             '`batch_shape` argument to your Input layer.')

        if self.states[0] is None:
            self.states = [K.zeros((batch_size, self.units))
                           for _ in self.states]
        elif states is None:
            for state in self.states:
                K.set_value(state, np.zeros((batch_size, self.units)))
        else:
            if not isinstance(states, (list, tuple)):
                states = [states]
            if len(states) != len(self.states):
                raise ValueError('Layer ' + self.name + ' expects ' +
                                 str(len(self.states)) + ' states, '
                                 'but it received ' + str(len(states)) +
                                 'state values. Input received: ' +
                                 str(states))
            for index, (value, state) in enumerate(zip(states, self.states)):
                if value.shape != (batch_size, self.units):
                    raise ValueError('State ' + str(index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected shape=' +
                                     str((batch_size, self.units)) +
                                     ', found shape=' + str(value.shape))
                K.set_value(state, value)

    def __call__(self, inputs, initial_state=None, **kwargs):
        # If `initial_state` is specified,
        # and if it a Keras tensor,
        # then add it to the inputs and temporarily
        # modify the input spec to include the state.
        if initial_state is not None:
            if hasattr(initial_state, '_keras_history'):
                # Compute the full input spec, including state
                input_spec = self.input_spec
                state_spec = self.state_spec
                if not isinstance(state_spec, list):
                    state_spec = [state_spec]
                self.input_spec = [input_spec] + state_spec

                # Compute the full inputs, including state
                if not isinstance(initial_state, (list, tuple)):
                    initial_state = [initial_state]
                inputs = [inputs] + list(initial_state)

                # Perform the call
                output = super(QRNN, self).__call__(inputs, **kwargs)

                # Restore original input spec
                self.input_spec = input_spec
                return output
            else:
                kwargs['initial_state'] = initial_state
        return super(QRNN, self).__call__(inputs, **kwargs)

    def call(self, inputs, mask=None, initial_state=None, training=None):
        # input shape: `(samples, time (padded with zeros), input_dim)`
        # note that the .build() method of subclasses MUST define
        # self.input_spec and self.state_spec with complete input shapes.
        if isinstance(inputs, list):
            initial_states = inputs[1:]
            inputs = inputs[0]
        elif initial_state is not None:
            pass
        elif self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(inputs)

        if len(initial_states) != len(self.states):
            raise ValueError('Layer has ' + str(len(self.states)) +
                             ' states but was passed ' +
                             str(len(initial_states)) +
                             ' initial states.')
        input_shape = K.int_shape(inputs)
        if self.unroll and input_shape[1] is None:
            raise ValueError('Cannot unroll a RNN if the '
                             'time dimension is undefined. \n'
                             '- If using a Sequential model, '
                             'specify the time dimension by passing '
                             'an `input_shape` or `batch_input_shape` '
                             'argument to your first layer. If your '
                             'first layer is an Embedding, you can '
                             'also use the `input_length` argument.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a `shape` '
                             'or `batch_shape` argument to your Input layer.')
        constants = self.get_constants(inputs, training=None)
        preprocessed_input = self.preprocess_input(inputs, training=None)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                            initial_states,
                                            go_backwards=self.go_backwards,
                                            mask=mask,
                                            constants=constants,
                                            unroll=self.unroll,
                                            input_length=input_shape[1])
        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_update(updates, inputs)

        # Properly set learning phase
        if 0 < self.dropout < 1:
            last_output._uses_learning_phase = True
            outputs._uses_learning_phase = True

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def preprocess_input(self, inputs, training=None):
        if self.window_size > 1:
            inputs = K.temporal_padding(inputs, (self.window_size-1, 0))
        inputs = K.expand_dims(inputs, 2)  # add a dummy dimension

        output = K.conv2d(inputs, self.kernel, strides=self.strides,
                          padding='valid',
                          data_format='channels_last')
        output = K.squeeze(output, 2)  # remove the dummy dimension
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')

        if self.dropout is not None and 0. < self.dropout < 1.:
            z = output[:, :, :self.units]
            f = output[:, :, self.units:2 * self.units]
            o = output[:, :, 2 * self.units:]
            f = K.in_train_phase(1 - _dropout(1 - f, self.dropout), f, training=training)
            return K.concatenate([z, f, o], -1)
        else:
            return output

    def step(self, inputs, states):
        prev_output = states[0]

        z = inputs[:, :self.units]
        f = inputs[:, self.units:2 * self.units]
        o = inputs[:, 2 * self.units:]

        z = self.activation(z)
        f = f if self.dropout is not None and 0. < self.dropout < 1. else K.sigmoid(f)
        o = K.sigmoid(o)

        #output = f * prev_output + (1 - f) * z
        #output = o * output
        c_output = f * prev_output + (1 - f) * z
        h_output = o * c_output

        #return output, [output]
        return h_output, [c_output]

    def get_constants(self, inputs, training=None):
        return []
 
    def get_config(self):
        config = {'units': self.units,
                  'window_size': self.window_size,
                  'stride': self.strides[0],
                  'return_sequences': self.return_sequences,
                  'go_backwards': self.go_backwards,
                  'stateful': self.stateful,
                  'unroll': self.unroll,
                  'use_bias': self.use_bias,
                  'dropout': self.dropout,
                  'activation': activations.serialize(self.activation),
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'input_dim': self.input_dim,
                  'input_length': self.input_length}
        base_config = super(QRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class QRNNDecoder:
    def __init__(self, num_layers=1, units=100, window_size=2, batch_size=32, epochs=5, dropout=0, stateful=False, shuffle=True, verbose=0):
        self.num_layers = num_layers
        self.units = units
        self.window_size = window_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout
        self.stateful = stateful
        self.shuffle = shuffle
        self.verbose = verbose
    
    def compile(self, params):
        self.num_layers = params['num_layers']
        self.units = params['units']
        self.window_size = params['window_size']
        self.kernel_init = params['kernel_init']
        self.batch_size = params['batch_size']   
        self.timesteps = params['timesteps']
        self.input_dim = params['input_dim']
        self.output_dim = params['output_dim']
        self.optimizer = params['optimizer']
        self.seed = params['seed']
        
        if self.optimizer=='RMSprop':
            #print("Using RMSprop optimizer")
            optim = RMSprop(lr=params['lrate'])
        elif self.optimizer=='Adam':
            #print("Using Adam optimizer")
            optim = Adam(lr=params['lrate'])
        else:
            optim = SGD(lr=params['lrate'], momentum=0.9, decay=1e-6, nesterov=True, clipnorm=1.)
        
        # Create new model or load existing model
        if params['load']:
            print("Loading existing model " +params['load_name'])
            model = load_model(params['load_name'],custom_objects={'rmse':rmse,'QRNN':QRNN})
        else:
            model = Sequential()
            if self.num_layers > 1:
                for i in range(self.num_layers-1):
                    model.add(QRNN(self.units, window_size=self.window_size, kernel_initializer=self.kernel_init, 
                                   input_shape=(self.timesteps,self.input_dim), dropout=self.dropout, stateful=self.stateful,
                                   return_sequences=True, activation='tanh'))
                model.add(QRNN(self.units, window_size=self.window_size, kernel_initializer=self.kernel_init, 
                               input_shape=(self.timesteps,self.input_dim), dropout=self.dropout, stateful=self.stateful,
                               activation='tanh'))
            else:
                model.add(QRNN(self.units, window_size=self.window_size, kernel_initializer=self.kernel_init,
                              input_shape=(self.timesteps,self.input_dim), dropout=self.dropout, stateful=self.stateful,
                              activation='tanh'))
            if self.dropout>0.: 
                model.add(Dropout(self.dropout))    
            model.add(Dense(self.output_dim,kernel_initializer=self.kernel_init))
            # Compile model
            model.compile(optimizer=optim,loss=rmse) #Set loss function and optimizer
            
        #print(model.summary())
        # Print parameter count
        num_params = model.count_params()
        print('# network parameters: ' + str(num_params))
        self.model = model
        return model
        
    def fit(self, X_train, Y_train, X_valid, Y_valid, params):
            
        self.epochs = params['epochs']
        self.dropout = params['dropout']
        self.stateful = params['stateful']
        self.shuffle = params['shuffle']
        self.verbose = params['verbose']
        self.fit_gen = params['fit_gen']
        # Fit model
        loss_train = []
        loss_valid = []
        predict_history = PredictHistory(X_valid,Y_valid)
        if params['retrain']:
            if self.stateful:
                if self.fit_gen:
                    train_generator = DataGenerator(X_train, Y_train, self.batch_size, self.shuffle)
                    for i in range(self.epochs):                        
                        hist = self.model.fit_generator(generator=train_generator,
                                                        validation_data=(X_valid,Y_valid),
                                                        epochs=1, shuffle=self.shuffle, verbose=self.verbose,
                                                        callbacks=[predict_history])
                        loss_train.append(hist.history['loss'])
                        loss_valid.append(hist.history['val_loss'])
                        self.model.reset_states()
                else:
                    for i in range(self.epochs):
                        #model.fit(X_train,Y_train,batch_size=self.batch_size,epochs=1,verbose=self.verbose,shuffle=self.shuffle)
                        hist = self.model.fit(X_train, Y_train,
                                              validation_data=(X_valid,Y_valid),
                                              batch_size=self.batch_size, epochs=1, verbose=self.verbose, shuffle=self.shuffle,
                                              callbacks=[predict_history])
                        loss_train.append(hist.history['loss'])
                        loss_valid.append(hist.history['val_loss'])
                        self.model.reset_states()
            else:
                #model.fit(X_train,Y_train,batch_size=self.batch_size,epochs=self.epochs,verbose=self.verbose,shuffle=self.shuffle)
                hist = self.model.fit(X_train, Y_train,
                                      validation_data=(X_valid, Y_valid),
                                      batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose, shuffle=self.shuffle,
                                      callbacks=[predict_history])
                loss_train.append(hist.history['loss'])
                loss_valid.append(hist.history['val_loss'])
            loss_train = flatten_list(loss_train)
            loss_valid = flatten_list(loss_valid)
        
        loss_predict = predict_history.pred_hist
        if params['save']:
            self.model.save(params['save_name'])
            
        fit_out = {'loss_train':loss_train,
                    'loss_valid':loss_valid,
                    'loss_predict':loss_predict,
                    'num_params':self.model.count_params()}                    
        return fit_out
    
    def predict(self,X_test):
        #self.model.reset_states()
        Y_pred = self.model.predict(X_test, batch_size=self.batch_size, verbose=self.verbose) #Make predictions
        return Y_pred

class KalmanDecoder:
    """
    Kalman filter decoding algorithm
    """
    def __init__(self, regular=None, alpha_reg=0):
        self.regular = regular # type of regularisation
        self.alpha_reg = alpha_reg # regularisation constant

    def fit(self, X_train, Y_train, params):
        self.regular = params['regular']
        self.alpha_reg = params['alpha_reg']
        if self.regular == 'l1':
            regres = Lasso(alpha=self.alpha_reg)            
        elif self.regular == 'l2':
            regres = Ridge(alpha=self.alpha_reg)
        elif self.regular == 'l12':
            regres = ElasticNet(alpha=self.alpha_reg)
        else:
            regres = LinearRegression()
        
        X = Y_train 
        Z = X_train 
        nt = X.shape[0]              
        X1 = X[:nt-1,:] 
        X2 = X[1:,:] 
        
        regres.fit(X1,X2)
        A = regres.coef_ # shape (2, 2)
        W = np.cov((X2-np.dot(X1,A.T)).T) # np.cov input (features,samples), output shape (2, 2)
        regres.fit(X,Z)
        H = regres.coef_ # shape (96, 2)
        Q = np.cov((Z-np.dot(X,H.T)).T) # shape (96,96)
        params = [A,W,H,Q] # ----> should be in matrix form (not numpy)
        self.model = params
    
    def predict(self, X_test, y_test):
        # extract parameters
        A,W,H,Q = self.model

        X = np.matrix(y_test.T)
        Z = np.matrix(X_test.T)

        # initialise states and covariance matrix
        num_states = X.shape[0] # dimensionality of the state
        states = np.empty(X.shape) # keep track of states over time (states is what will be returned as y_pred)
        P_m = np.matrix(np.zeros([num_states,num_states]))
        P = np.matrix(np.zeros([num_states,num_states]))
        state = X[:,0] # initial state
        states[:,0] = np.copy(np.squeeze(state))

        # get predicted state for every time bin
        for t in range(X.shape[1]-1):
            # do first part of state update - based on transition matrix
            P_m = A*P*A.T+W
            state_m = A*state

            # do second part of state update - based on measurement matrix
            try:
                K = P_m*H.T*inv(H*P_m*H.T+Q) #Calculate Kalman gain
            except np.linalg.LinAlgError:
                K = P_m*H.T*pinv(H*P_m*H.T+Q) #Calculate Kalman gain
            P = (np.matrix(np.eye(num_states))-K*H)*P_m
            state = state_m+K*(Z[:,t+1]-H*state_m)
            states[:,t+1] = np.squeeze(state) #Record state at the timestep
        y_pred = states.T
        return y_pred

class WienerDecoder:
    """
    Wiener filter decoding algorithm
    """
    def __init__(self, regular=None, alpha=0):
        self.regular = None # type of regularisation
        self.alpha = 0 # regularisation constant

    def fit(self, X_train, y_train, params):
        self.regular = params['regular']
        self.alpha = params['alpha']
        if self.regular == 'l1':
            self.model = Lasso(alpha=self.alpha)            
        elif self.regular == 'l2':
            self.model = Ridge(alpha=self.alpha)
        elif self.regular == 'l12':
            self.model = ElasticNet(alpha=self.alpha)
        else:
            self.model = LinearRegression()
        
        self.model.fit(X_train, y_train)

    def predict(self,X_test):
        y_pred = self.model.predict(X_test) #Make predictions
        return y_pred

class WienerCascadeDecoder:
    """
    Wiener cascade filter decoding algorithm
    """
    def __init__(self, regular=None, alpha=0, degree=3):
        self.regular = None
        self.alpha = 0
        self.degree = degree

    def fit(self, X_train, y_train, params):
        num_outputs = y_train.shape[1] #Number of outputs
        models = [] #Initialize list of models (there will be a separate model for each output)
        self.regular = params['regular']
        self.alpha = params['alpha']
        self.degree = params['degree']
        for i in range(num_outputs): #Loop through outputs
            if self.regular == 'l1':
                regres = Lasso(alpha=self.alpha)            
            elif self.regular == 'l2':
                regres = Ridge(alpha=self.alpha)
            elif self.regular == 'l12':
                regres = ElasticNet(alpha=self.alpha)
            else:
                regres = LinearRegression()
            regres.fit(X_train, y_train[:,i]) #Fit linear
            y_pred_linear = regres.predict(X_train) # Get outputs of linear portion of model
            #Fit nonlinear portion of model
            p = np.polyfit(y_pred_linear, y_train[:,i], self.degree)
            #Add model for this output (both linear and nonlinear parts) to the list "models"
            models.append([regres, p])
        self.model = models

    def predict(self, X_test):
        num_outputs = len(self.model) #Number of outputs being predicted. Recall from the "fit" function that self.model is a list of models
        y_pred = np.empty([X_test.shape[0], num_outputs]) #Initialize matrix that contains predicted outputs
        for i in range(num_outputs): #Loop through outputs
            [regres, p] = self.model[i] #Get the linear (regr) and nonlinear (p) portions of the trained model
            y_pred_linear = regres.predict(X_test) #Get predictions on the linear portion of the model
            y_pred[:,i] = np.polyval(p, y_pred_linear) #Run the linear predictions through the nonlinearity to get the final predictions
        return y_pred