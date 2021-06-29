import json
from train import TassenlNetTrainer
from predict import TassenlNetPredictor

with open('mission.json', 'w') as file:
    missions = json.load(file)

for training_info in missions['train']:
    image_path = training_info["image_path"]
    density_map_path = training_info["density_map_path"]
    test_trainer = TassenlNetTrainer("lenet",(image_path, density_map_path))
    # test1
    image_size = training_info["image_size"]
    input_shape = training_info["input_shape"]
    batch_size = training_info["batch_size"]
    epochs = training_info["epochs"]
    learning_rate = training_info["learning_rate"]
    optimizer = training_info["optimizer"]
    loss = training_info["loss"]

    test_trainer.train(image_size, input_shape, batch_size, epochs, learning_rate, optimizer, loss)

