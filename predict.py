import os 
import glob
import numpy as np
from model import *
from skimage import io
from DatasetReader import DatasetReader

class TassenlNetPredictor:
    def __init__(self, model_name, paths_for_test, save_dir="log"):
        self.save_dir = save_dir
        self.model_name = model_name
        self.paths_for_test = paths_for_test

    def _predict(self, model, data_set_generator):
        result = model.predict(data_set_generator)
        return result

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        return 255 * ((data - np.min(data)) / _range)

    def construct_density_map_by_result(self, prediction, image_size, input_shape=(), accumulate_area=False):
        # print(len(prediction), (image_size[0] * image_size[1]))
        number_of_density_map = len(prediction) // (image_size[0] * image_size[1])
        # print('*' * 5, number_of_density_map)
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
                        density_map_array[density_map_index,xmin:xmax, ymin:ymax] = prediction[density_map_index * image_size[0] * image_size[1] + i * image_size[0] + j]
        else:
            for density_map_index in range(number_of_density_map):
                for i in range(image_size[0]):
                    for j in range(image_size[1]):
                        density_map_array[density_map_index][i][j] = prediction[density_map_index * image_size[0] * image_size[1] + i * image_size[0] + j]
        return density_map_array

    def predict_by_name(self, image_size=(224,224), input_shape=(31, 31, 3), batch_size=1, epochs="*", learning_rate="*", optimizer="*", loss="*", strict=False):
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
            model.load_weights(model_path)
        else:
            raise Exception("没找到模型：" + saved_model_name)
        if isinstance(self.paths_for_test, list) or isinstance(self.paths_for_test, list):
            test_set_reader = DatasetReader(self.paths_for_test[0], self.paths_for_test[1])
        else:
            test_set_reader = DatasetReader(self.paths_for_test[0])

        test_set_generator = test_set_reader.generate_slice_and_labels(input_shape[0], batch_size, image_size, filter_content=1)
        predict_result = self._predict(model, test_set_generator)
        # print(predict_result[0].shape)
        # reshpaped_predict_result = np.reshape(predict_result, (predict_result.shape[0] * predict_result.shape[1]))

        # density_map_array = self.construct_density_map_by_result(reshpaped_predict_result, image_size, input_shape, accumulate_area=True)
        density_map_array = self.construct_density_map_by_result(predict_result, image_size, input_shape, accumulate_area=True)
        # print(density_map_array)
        count_result = np.sum(density_map_array, axis=0)
        image_name_list = test_set_reader.get_order_image_name()

        result_save_dir = os.path.join(self.save_dir, self.model_name, os.path.basename(model_path).rstrip('.h5'))
        if not os.path.exists(result_save_dir):
            os.makedirs(result_save_dir)

        for i, image_name in enumerate(image_name_list[:2]):
            print(density_map_array.max())
            density_map_save_path = os.path.join(result_save_dir, image_name.split('.')[0] + '.npy')
            np.save(density_map_save_path, density_map_array[i])

            scaled_density_map = self.normalization(density_map_array[i]).astype(np.uint8)
            scaled_density_map_save_path = os.path.join(result_save_dir, image_name.split('.')[0] + '.png')
            io.imsave(scaled_density_map_save_path, scaled_density_map)
            with open("result.csv", 'w') as file:
                file.write(','.join([image_name, str(count_result[i])]) + '\n')
        return density_map_array, count_result

if __name__ == '__main__':
    image_path = "/Users/Biomind/Documents/周文静/TasselNet/spruce/202011/train"
    density_map_path = "/Users/Biomind/Documents/周文静/TasselNet/spruce/202011_den/train"

    test_predictor = TassenlNetPredictor("lenet",(image_path, density_map_path))

    # test1
    image_size = (224,224)
    input_shape = (31, 31, 3)
    # batch_size = 4
    # epochs = 10
    # learning_rate = 0.01
    # optimizer = 'adm'
    # loss = 'mean_squared_error'

    # test_predictor.predict_by_name(image_size, input_shape, batch_size, epochs, learning_rate, optimizer, loss, strict=False)
    test_predictor.predict_by_name(image_size, input_shape, strict=False)