import os 
import glob
import numpy as np
from model import *
from skimage import io
from DatasetReader import DatasetReader

class TassenlNetPredictor:
    def __init__(self, model_name, paths_for_test, save_dir):
        self.save_dir = save_dir
        self.model_name = model_name
        self.paths_for_test = paths_for_test

    def _predict(model, training_set_generator):
        result = model.predict_generator(training_set_generator)

    def normalization(data):
        _range = np.max(data) - np.min(data)
        return 255 * ((data - np.min(data)) / _range)
    def construct_density_map_by_result(prediction, image_size, input_shape=(), accumulate_area=False):
        number_of_density_map = len(prediction) // (image_size[0] * image_size[1])
        density_map_array = np.zeros((number_of_density_map, image_size[0], image_size[1]), np.float)
        if accumulate_area:
            if not input_shape:
                raise Exception("合成密度图需要划窗尺寸")
            for density_map_index in range(number_of_density_map):
                for i in range(image_size[0]):
                    for j in range(image_size[1]):
                        xmin = max(i - (image_size[0] // 2), 0)
                        xmax = min(i + (image_size[0] // 2), image_size[0] - 1)
                        ymin = max(j - (image_size[1] // 2), 0)
                        ymax = min(j + (image_size[1] // 2), image_size[1] - 1)
                        density_map_array[density_map_index,xmin:xmax, ymin,ymax] = prediction[density_map_index * image_size[0] * image_size[1] + i * image_size[0] + j]
        else:
            for density_map_index in range(number_of_density_map):
                for i in range(image_size[0]):
                    for j in range(image_size[1]):
                        density_map_array[density_map_index][i][j] = prediction[density_map_index * image_size[0] * image_size[1] + i * image_size[0] + j]
            

    def predict_by_name(self, image_size=(224,224), input_shape=(31, 31, 3), batch_size=4, epochs="*", learning_rate="*", optimizer="*", loss="*", strict=False):
        if strict:
            saved_model_name = '-'.join([str(i) for i in [image_size[0], image_size[1],
                                                        input_shape[0],input_shape[1], 
                                                        batch_size, epochs, learning_rate,
                                                        optimizer, loss, self.model_name ]
                                                        ]) + '.h5'
        else:
            saved_model_name = '-'.join([str(i) for i in [image_size[0], image_size[1],
                                                        input_shape[0],input_shape[1], 
                                                        "*", epochs, learning_rate,
                                                        optimizer, loss, self.model_name ]
                                                        ]) + '.h5'
        search_path = os.path.join(self.save_dir, self.model_name)
        search_path = os.path.join(search_path, saved_model_name)
        search_result = glob.glob(search_path)

        model = MODELS[self.model_name.lower()]['model'](input_shape)
        if search_result:
            model_path = search_result[0]
            model = model.load_weights(model_path)
        else:
            raise Exception("没找到模型：" + saved_model_name)
        if isinstance(self.paths_for_test, list) or isinstance(self.paths_for_test, list):
            test_set_reader = DatasetReader(self.paths_for_test[0], self.paths_for_test[1])
        elif isinstance(self.paths_for_test, str):
            test_set_reader = DatasetReader(self.paths_for_test[0])

        test_set_generator = test_set_reader.generate_slice_and_labels(input_shape[0], batch_size, image_size)
        predict_result = self._predict(model, test_set_generator)
        density_map_array = self.construct_density_map_by_result(predict_result, image_size, input_shape, accumulate_area=True)
        count_result = np.sum(density_map_array, axis=0)
        image_name_list = test_set_reader.get_order_image_name()

        result_save_dir = os.path.join(self.save_dir, os.path.basename(search_result))
        os.path.makedirs(result_save_dir)
        for i, image_name in enumerate(image_name_list):

            density_map_save_path = os.path.join(result_save_dir, image_name.split('.')[0] + '.npy')
            np.save(density_map_save_path, density_map_array[i])

            scaled_density_map = self.normalization(density_map_array[i]).astype(np.uint8)
            scaled_density_map_save_path = os.path.join(result_save_dir, image_name.split('.')[0] + '.png')
            io.imsave(os.path.join(scaled_density_map_save_path, image_name), scaled_density_map)

        return density_map_array, count_result

