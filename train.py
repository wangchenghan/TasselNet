# -- coding:utf-8 --
import os
from model import *
from DatasetReader import DatasetReader
from keras import optimizers
class TassenlNetTrainer:
    def __init__(self, model_name, paths_for_train, paths_for_validate=None, save_dir='log'):
        if len(paths_for_train) != 2:
            raise Exception("应该有两个路径，一个是图片的，一个是密度图的")
        self.model_name = model_name
        self.paths_for_train = paths_for_train
        self.paths_for_validate = paths_for_validate
        self.save_dir = save_dir

    def train(self,image_size=(224,224), input_shape=(31, 31, 3), batch_size=4, epochs=10, learning_rate=0.01, optimizer='adm', loss='mean_squared_error'):
        image_path = self.paths_for_train[0]
        density_map_path = self.paths_for_train[1]
        training_set_reader = DatasetReader(image_path, density_map_path)
        training_set_generator = training_set_reader.generate_slice_and_labels(input_shape[0], batch_size, image_size, accumulate_area=True)
        # acculumate_area为True时是TasselNet的处理方式，为False时是密度图的处理方式
        model = MODELS[self.model_name.lower()]['model'](input_shape)
        #optimizers
        if optimizer.lower() == 'sgd':
            keras_optimizer = optimizers.SGD(lr=learning_rate, clipvalue=0.5)
        elif optimizer.lower()  == 'adagrad':
            keras_optimizer = optimizers.Adagrad(lr=learning_rate, epsilon=None, decay=0.0)
        elif optimizer.lower()  == 'adadelta':
            keras_optimizer = optimizers.Adadelta(lr=learning_rate, rho=0.95, epsilon=None, decay=0.0)
        elif optimizer.lower()  == 'nadam':
            keras_optimizer = optimizers.Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
        elif optimizer.lower()  == 'Adadelta':
            keras_optimizer = optimizers.Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        else:
            keras_optimizer = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        
        model.compile(loss=loss, optimizer=keras_optimizer)
        if self.paths_for_validate:
            raise NotImplementedError
        else:
            model.fit_generator(training_set_generator, epochs=epochs, verbose=1)
        model_save_name = '-'.join([str(i) for i in [image_size[0], image_size[1],
                                                    input_shape[0],input_shape[1], 
                                                    batch_size, epochs, learning_rate,
                                                    optimizer, loss, self.model_name ]
                                                    ]) + '.h5'
        model_save_path = os.path.join(self.save_dir, self.model_name)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        model_save_path = os.path.join(model_save_path, model_save_name)
        model.save(model_save_path)
        print("training is done!")
        return model

if __name__ == '__main__':
    image_path = "/Users/Biomind/Documents/周文静/TasselNet/spruce/202011/train"
    density_map_path = "/Users/Biomind/Documents/周文静/TasselNet/spruce/202011_den/train"

    test_trainer = TassenlNetTrainer("lenet",(image_path, density_map_path))

    # test1
    image_size = (224,224)
    input_shape = (31, 31, 3)
    batch_size = 4
    epochs = 10
    learning_rate = 0.01
    optimizer = 'adm'
    loss = 'mean_squared_error'

    test_trainer.train(image_size, input_shape, batch_size, epochs, learning_rate, optimizer, loss)
