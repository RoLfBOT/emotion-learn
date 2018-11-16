from data.data_preprocess import Dataset
from model.model import Model

if __name__ == "__main__":
    dataset = Dataset()
    dataset.get_data()
    
    net = Model(20)
    net.build_model()
    net.train(dataset.images, dataset.labels)