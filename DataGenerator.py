#!/usr/bin/python3.6
#-*- coding: utf-8 -*-

import numpy as np
import keras

import p9_util

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
class DataGenerator(keras.utils.Sequence) :
    '''Generates data for Keras input layers.s
    '''
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def __init__(self, dict_X, dict_label, nb_record, \
    partition_size=0,\
    batch_size=0, \
    is_shuffle=False, \
    is_dimension_mux=False,\
    keras_input_dim=(0,0,0), \
    n_channels=0,\
    n_classes=0, \
    keras_nb_channel=0, \
    proj_dimension = None,\
    data_type = None,\
    list_keras_channel = list(),\
    other = None,\
    index_oov=None,\
    binary_threshold = None,\
    ):
    
        'Initialization'
        
        if nb_record <= partition_size :
            self.partition_size = nb_record
        else :    
            self.partition_size = partition_size
        self.keras_input_dim = keras_input_dim
        self.batch_size = batch_size

        self.dict_label = dict_label.copy() 
        self.dict_X = dict_X.copy()

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.is_shuffle = is_shuffle
        self.nb_record = nb_record
        
        self.keras_nb_channel = keras_nb_channel
        self.is_dimension_mux = is_dimension_mux
        self.proj_dimension = proj_dimension
        self.data_type = data_type
        self.list_keras_channel = list_keras_channel
        self.dict_handle_listfileindex = dict()
        self._index_oov = None
                
        #-----------------------------------------------------------------------
        # When True, then batch indexes are those from list_index_map.
        # Otherwise, there is a single batch index : self.index_oov
        #-----------------------------------------------------------------------
        self._is_training = True

        #-----------------------------------------------------------------------
        # This assignation allows to build list_index_map 
        #-----------------------------------------------------------------------
        self.index_oov = index_oov
        
        #-----------------------------------------------------------------------
        # This assignation allows to know what is threshold considered 
        # for binary target value.
        # This threshold value has been fixed into DataPreparator_v2.
        #-----------------------------------------------------------------------
        self.binary_threshold = binary_threshold
        
        if other is not None :
            self.partition_size = other.partition_size
            self.keras_input_dim = other.keras_input_dim
            self.batch_size = other.batch_size

            self.dict_label = other.dict_label.copy() 
            self.dict_X = other.dict_X.copy()

            self.n_channels = other.n_channels
            self.n_classes = other.n_classes

            self.is_shuffle = other.is_shuffle
            self.nb_record = other.nb_record
            
            self.keras_nb_channel = other.keras_nb_channel
            self.is_dimension_mux = other.is_dimension_mux
            self.proj_dimension = other.proj_dimension
            self.data_type = other.data_type
            self.list_keras_channel = other.list_keras_channel.copy()
            self.dict_handle_listfileindex = other.dict_handle_listfileindex.copy()

            #-------------------------------------------------------------------
            # Next sequences for index_oov and is_training has to take place in 
            # the following order.
            #-------------------------------------------------------------------
            self.index_oov = other.index_oov
            if self.index_oov is not None :
                self.is_training = other.is_training
            else :
                pass
            
            self.binary_threshold = other.binary_threshold
        
        
        #-------------------------------------------------------------------------
        # This builds list of indexes and shuffle it if is_shuffle flag is activated.
        #-------------------------------------------------------------------------
        self.on_epoch_end()
    #---------------------------------------------------------------------------
    
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def get_params(self):
        return {'keras_input_dim':self.keras_input_dim, \
                'batch_size':self.batch_size, \
                'n_classes':self.n_classes, \
                'n_channels':self.n_channels,\
                'is_shuffle':self.is_shuffle,\
                'keras_nb_channel': self.keras_nb_channel,\
                'list_keras_channel': self.list_keras_channel,\
                'data_type' : self.data_type,\
                'proj_dimension' : self.proj_dimension,\
                'is_dimension_mux' : self.is_dimension_mux,\
                'binary_threshold' : self.binary_threshold,\
                }
    #---------------------------------------------------------------------------
            
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def __len__(self):
        'Provides the number of batches per epoch'
        #return int(np.floor(len(self.dict_X) / self.batch_size))
        #return int(np.floor(self.len_data / self.batch_size))
        #return self.len_data
        if (0 == self.nb_record ) or ( 0 == self.batch_size) :
            len_ = 0
        else :
            if self.is_training :
                #---------------------------------------------------------------
                # DataGenerator object is used for training steps.
                # List of batchs is provided by list_index_map.
                #---------------------------------------------------------------
                if (self.nb_record / self.batch_size) - round(self.nb_record / self.batch_size) >0. :
                    len_ = round(self.nb_record/self.batch_size)+1 
                else : 
                    len_= round(self.nb_record/self.batch_size)

                if self._index_oov is not None :
                    len_ -=1
                else : 
                    pass
            else :
                #---------------------------------------------------------------
                # DataGenerator object is used for validation step.
                # There is a single batch index : self.index_oov
                #---------------------------------------------------------------
                len_=1
        return len_
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    # Properties
    #---------------------------------------------------------------------------
    def _get_index_oov(self) :
        return self._index_oov
        
    def _set_index_oov(self, index_oov) :
        length = self.__len__()    
        if index_oov is not None :
            if (0 > index_oov) or (length <= index_oov) :
                if not self.is_training :
                    self._index_oov = index_oov
                else : 
                    print("\n*** ERROR : index_oov out of range ! Expected range : [0,{}[".format(length))
            else : 
                self._index_oov = index_oov
                if self._index_oov is not None :
                    length +=1 
                    self.list_index_map = [ind for ind in range(length) if ind!= self._index_oov]
                else :
                    self.list_index_map = [ind for ind in range(length)]
        else :
            self.list_index_map = [ind for ind in range(length)]
            #pass #print("\n*** WARNING : None value for index_oov is not expected !")
            
    def _get_is_training(self) :
        return self._is_training
    
    def _set_is_training(self, is_training) :
        if self.index_oov is None :
            print("\n*** ERROR : training flag can't be changed. Assign a value to index_oov First!")
        else :
            self._is_training = is_training    
    
    def _set_y(self, y) :
        pass
    def _get_y(self, batch_start=0) :
        #-----------------------------------------------------------------------
        # Build a tuple for y shape
        #-----------------------------------------------------------------------
        list_tuple = list()
        
        #-----------------------------------------------------------------------
        # 1st dimension is the total number of records
        #-----------------------------------------------------------------------
        if 0 >= batch_start :
            list_tuple.append(self.nb_record)
        else :
            nb_record = 0
            for batch_index in range(batch_start, len(self)) :
                X, yi = self[batch_index]
                nb_record += len(yi)
            list_tuple.append(nb_record)
        
                 

        #-----------------------------------------------------------------------
        # Get y value from 1st batch; it will give dimensions out of 1st dimension
        #-----------------------------------------------------------------------
        y0 = self[0][1] # Returns a tuple (X,y); 

        #-----------------------------------------------------------------------
        # Complete tuple of dimensions, out of 1st dimension
        #-----------------------------------------------------------------------
        for item,i in zip(y0.shape, range(len(y0.shape))) :
            if i > 0 :
                list_tuple.append(item)
        y_shape = tuple(i for i in list_tuple)
        
        #-----------------------------------------------------------------------
        # Initializaition of recepient Y array
        #-----------------------------------------------------------------------
        _y = np.zeros(y_shape)
        
        #-----------------------------------------------------------------------
        # Transfer arrays values from yi stored into files
        #-----------------------------------------------------------------------
        start_row = 0
        for i in range(batch_start, len(self)) :
            X, yi = self[i]
            end_row = start_row + len(yi)
            _y[start_row:end_row,] = yi.copy()
            start_row = end_row
        return _y
                        
    def _set_X(self, X):
        pass
    def _get_X(self, batch_start=0) :
        #-----------------------------------------------------------------------
        # 1st dimension is the total number of records
        #-----------------------------------------------------------------------
        
        if 0 >= batch_start :
            nb_record = self.nb_record
        else :
            nb_record = 0
            for batch_index in range(batch_start, len(self)) :
                X, yi = self[batch_index]
                nb_record += len(yi)
    
        _X = np.zeros((nb_record, self.keras_input_dim[0], self.keras_input_dim[1]))
        
        start_row = 0
        for i in range(batch_start, len(self)) :
            Xi, yi = self[i]
            end_row = start_row + len(yi)
            _X[start_row:end_row] = Xi
            start_row = end_row
        return _X


    index_oov =  property(_get_index_oov,_set_index_oov)
    is_training =  property(_get_is_training, _set_is_training)
    X = property(_get_X, _set_X)
    y = property(_get_y, _set_y)
    
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def get_batch_index(self, batch_index) :
        '''Select batch_index value depending of is_training flag.
        When is_training flag is True, this objct run in training mode.
        Then index maped to batch_index is picked from list_index_map.
        
        Otherwise, this object runs in validation mode; index macthing 
        with batch_index is 0 value.
        '''

        if self.is_training :
            if max(self.list_index_map) < batch_index :
                print("\n***ERROR : batch ID= {} > {} ".format(batch_index, max(self.list_index_map)))
                batch_index = None
            else : 
                batch_index = self.list_index_map[batch_index]
        else :
            batch_index = 0
        return batch_index
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def __getitem__(self, batch_index):
        'Generate one batch of data'
        
        batch_index = self.get_batch_index(batch_index)
        if batch_index is None :
            return None, None
        else :
            pass

        #-----------------------------------------------------------------------
        # List of indexes of all items belonging to the batch is built.
        # This list has the same size value as batch_size.
        #-----------------------------------------------------------------------
        start_batch_index = batch_index*self.batch_size
        end_batch_index   = min((batch_index+1)*self.batch_size,self.nb_record)
        
        list_batch_index = self.indexes[start_batch_index:end_batch_index]
        self.dict_handle_listfileindex = self.get_dict_handle_listindex(list_batch_index)
        
        # Generate data
        X, y= self.__data_generation(self.dict_handle_listfileindex)
    
        #-----------------------------------------------------------------------
        # return data depending on channels to be built into ANN.
        #-----------------------------------------------------------------------
        if 1 == self.keras_nb_channel :
            return np.array(X), y
        else :
            if self.is_dimension_mux :
                if self.proj_dimension is None :
                    #return [X[i] for i in range(self.keras_nb_channel)], y
                    return [X[i] for i in self.list_keras_channel], y
                else : 
                    if self.proj_dimension is None :
                        return np.array(X), y
                    else :
                        #-------------------------------------------------------
                        # Returns the projection of X over the given dimension
                        #-------------------------------------------------------
                        return np.array(X)[self.proj_dimension], y
                
            else :
                if self.proj_dimension is None :
                    if 0 == self.keras_nb_channel :
                        return np.array(X), y  
                    else :                    
                        return np.array([X]*self.keras_nb_channel), y  
                else :
                    return np.array([X])[self.proj_dimension], y  
                    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.dict_X)*self.partition_size)
        if self.is_shuffle == True:
            np.random.shuffle(self.indexes)

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def on_epoch_end_deprecated(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.dict_X))
    #---------------------------------------------------------------------------
    
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def data_generation(self, dict_handle_listfileindex):
        return self.__data_generation(dict_handle_listfileindex)
    
    def __data_generation(self, dict_handle_listfileindex):
        '''Generates data containing batch_size samples
        dict_handle_listfileindex : this is the list of files idendified with an 
        handle that contains the list of all items indexes contained in this batch.
        '''

        #-----------------------------------------------------------------------
        # n_item is the total number of items into this batch.
        # Compute the size of X, depending on files into dict_handle_listfileindex
        # All files may not contain the same number of elements.
        #-----------------------------------------------------------------------
        n_item=0
        for handle_file in dict_handle_listfileindex.keys():
            list_file_index = dict_handle_listfileindex[handle_file]
            n_item += len(list_file_index)
        
        #print("n_item = {}".format(n_item))
        if self.is_dimension_mux :
            #-------------------------------------------------------------------
            # Multiplexing data by dimension: 
            #   -> 1st dimension is embedding dimension. Each one of these 
            #      dimensions will be allocated to a dedicated channel in 
            #      CNN network.
            #   -> 2nd dimension is number of items (texts) in the batch 
            #   -> 3rd dimension is number of word in a text.
            #   -> 4th dimenstion is 1, allowing to feed Keras input layer 
            #      with a vector of values.
            #-------------------------------------------------------------------
            X = np.empty((self.keras_nb_channel,n_item,self.keras_input_dim[0], 1))
            #X = np.empty((self.keras_nb_channel,self.keras_input_dim[0], 1))
        else :
            X = np.empty((n_item, self.keras_input_dim[0], self.keras_input_dim[1]))
            #print(X.shape)

        y = np.empty(n_item, dtype=float)

        #-----------------------------------------------------------------------
        # Generate data
        # Read each file using dict_handle_listfileindex and store values
        # into X array depending of list of indexes of items in each file.
        #-----------------------------------------------------------------------
        n_item_start=0
        n_item_end = 0

        for handle_file in dict_handle_listfileindex.keys():

            file_name = self.dict_X[handle_file]
            list_file_index = dict_handle_listfileindex[handle_file]
            n_item_end += len(list_file_index)
            arr_index = np.array(list_file_index)
            
            try :
                x = np.load(file_name)[arr_index]
                #print(x.shape)
            except IndexError as indexError:
                print("\n*** ERROR : {}".format(indexError))
                print("\n*** ERROR : index error on file= {} for handle = {}".\
                format(file_name,handle_file))
                print("*** ERROR : item indexes (start,end, index_size, min_index, max_index) = ({},{},{},{},{})".\
                format(n_item_start, n_item_end, len(arr_index), min(arr_index), max(arr_index)))
            
            if self.is_dimension_mux :
                #---------------------------------------------------------------
                # Multiplexing of data in a such way a row from tensor handles
                # all textes in same dimension.
                #---------------------------------------------------------------
                pos_item = 0
                pos_row  = 1
                pos_dim  = 2
                dim_transpose = (pos_dim, pos_item, pos_row)

                xm = p9_util.multiplexer(x, dim_transpose, nb_dim=self.keras_nb_channel)

            try :
                if self.is_dimension_mux	 :
                    n_item_end += self.keras_nb_channel
                    X[n_item_start:n_item_end] =xm.reshape(-1, n_item, self.keras_input_dim[0],1 ).copy()
                else : 
                    #X[n_item_start:n_item_end,] = x.reshape(-1, self.keras_input_dim[0],1).copy()
                    if self.proj_dimension is None :
                        X[n_item_start:n_item_end,:] = x[:n_item,:]
                    else :
                        X[n_item_start:n_item_end,:,0] = x[:n_item,:,self.proj_dimension]
                                    
            except ValueError as valueError:
                print("\n*** ERROR : {}".format(valueError))
                print("File name= {}".format(file_name))
                print("Indexes for X sliding (start,end)=({},{})".format(n_item_start,n_item_end))     
                print("Loaded x shape={}".format(x.shape))
                print("X storage shape={}".format(X.shape))
                if self.is_dimension_mux :
                    print("Multiplexed dimension shape = {}".format(xm.shape))
                
                
                   
            y[n_item_start:n_item_end] = self.dict_label[file_name][arr_index]
            n_item_start = n_item_end            

        if self.n_classes >= 2 :
            # Binary classes : 0 or 1
            # Threshold = 0
            # Direction 1
            # This means : if value > Threshold :
            #                   value = 1
            #               else :
            #                   value = 0
            # NB : case of n_classes > 2 has to be taken into account!!!
            if 2 == self.n_classes :
                threshold = 0
                direction = 1
                y=p9_util.multivalue2_binary(y, threshold, direction)
                return X, keras.utils.to_categorical(y, num_classes=self.n_classes,dtype='int')
            else :
                print("\n*** ERROR : multiple classification > 2 not yet supported!")
                return None, None
        else :
            #print(X.shape)
            return X,y
    #---------------------------------------------------------------------------
    
        
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def get_dict_handle_listindex(self, list_batch_index) :
        '''
            Returns a dictionary where keys are files names (files handles) 
            that belong to a batch and values are list of indexes of data in 
            files.
            
            A batch is defined by list_batch_index, that contains the list of 
            all items belonging to a batch.
            
            Input : 
                *   list_batch_index : list of indexes of all items belonging to 
                                       a batch.
            Output : 
                *   dictionary structures as : {handle:list_index} where handle 
                    is an identifier allowing to access file storing items, 
                    and list_index is the range of indexes of items in file;
                    The file belongs to the batch.
        '''
        #-----------------------------------------------------------------------
        # List of files-handles is built from batch indexes
        #-----------------------------------------------------------------------
        start_batch_index = list_batch_index[0]
        end_batch_index   = list_batch_index[-1]

        #-----------------------------------------------------------------------
        # Order indexes in case of shuffling
        #-----------------------------------------------------------------------
        start_batch_index = list_batch_index[0]
        end_batch_index   = list_batch_index[-1]
        list_batch_ = [start_batch_index,end_batch_index]
        list_batch_.sort()
        start_batch_index = list_batch_[0]
        end_batch_index = list_batch_[-1]
        
        
        start_file_handle = start_batch_index//self.partition_size
        end_file_handle = (end_batch_index//self.partition_size)+1
        list_handle = [filehandle for filehandle in range(start_file_handle,end_file_handle)]
        list_handle.sort()
        if False :
            print("")
            print("(start,end) batches indexes  = ({},{})".format(start_batch_index, end_batch_index))
            print("(start,end) files            = ({},{})".format(start_file_handle, end_file_handle))
            print("List of file-handles in batch= {}".format(list_handle))
    
        
        #-----------------------------------------------------------------------
        # Dictionary of files indexes is built from list of files-handles.
        #
        # start_batch_index : is the index of the first item of the batch.
        # end_batch_index : is the index of the last item of the batch.
        # start_file_index : is the index of the first item contained in file.
        # end_file_index is the index of the last item contained in file.
        # dict_handle_listindex : the dictionary structured as : {file_handle:list_index}
        #   * file_handle : is an handle allowing to acces content in file 
        #                   This is the file name.
        #   * list_index : this is he list of indexes of all items in file .
        #                  
        #  
        #-----------------------------------------------------------------------
        start_file_index = start_batch_index
        dict_handle_listindex = dict()
        for handle in list_handle :
            #-------------------------------------------------------------------
            # For the last file, end_batch_index is the last indexe of items.
            # It has to be reached from end_file_index. 
            # For doing so, end_batch_index+1 stands for end_file_index for last file 
            # from batch.
            #-------------------------------------------------------------------
            end_file_index = min((handle+1)*self.partition_size,end_batch_index+1)
            shift_index = handle*self.partition_size
            dict_handle_listindex[handle] = [index-shift_index for index in range(start_file_index,end_file_index)]
            if False :
                    print("File handle= {} / Start index= {} / End index={} / Number elements= {}"\
                          .format(handle, start_file_index, end_file_index-1, len(dict_handle_listindex[handle])))
            start_file_index = end_file_index
            
        #-----------------------------------------------------------------------
        # Files handles are shuffled
        #-----------------------------------------------------------------------
        if self.is_shuffle == True:
            list_keys = list(dict_handle_listindex.keys())
            np.random.is_shuffle(list_keys)
            dd = dict()
            for key in list_keys :
                dd[key] = dict_handle_listindex[key]
            
            dict_handle_listindex = dd.copy()
        return dict_handle_listindex
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    def get_Xproj_y_from_generator(self, dimension) :
        '''Get all textes projected on a given dimension
        
        Input :
            *   dimension : value of the PCA dimension corpus will be 
            projected over.
            
        Output :
            *   projected corpus over the given dimension and target vector.
        '''
        nb_token = self.get_params()['keras_input_dim'][0]
        x_proj = np.zeros((self.nb_record, nb_token))
        y = np.zeros((self.nb_record))

        end_slide = 0
        start_slide = 0

        for i in range(len(self)) : 
            list_X, Y =self[i]
            X = np.array(list_X)
            nb_record = X.shape[1]
            try :
                end_slide += nb_record
                x_proj[start_slide: end_slide] = X[dimension,:,:,0]
                y[start_slide: end_slide:] = Y
                start_slide = end_slide
            except ValueError as valueError :
                print("Error : {}".format(valueError))
                print("Record : {}".format(i))
                break
        return x_proj,y
    #---------------------------------------------------------------------------

#-------------------------------------------------------------------------------
