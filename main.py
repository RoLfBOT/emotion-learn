from data.data_preprocess import Dataset
from model.model import Model
from predictor import Predictor

if __name__ == "__main__":
    print('Preprocessing data ...')
    dataset = Dataset()
    dataset.get_data()
    
    print('Training model ...')
    net = Model(20)
    net.build_model()
    net.train(dataset.images, dataset.labels)

    