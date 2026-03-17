




from src.data_processer.datasets.TestDatasets.RegressionDataset import RegressionDataset
from src.config.data_processer.datasets.TestDatasets.RegressionDatasetConfig import RegressionDatasetConfig
from src.data_processer.datasets.data_factory import get_dataset

from src.visualize_tools.annotation_tools.annotation import AnnotationGUI


if __name__ == "__main__":
    # config = RegressionDatasetConfig(data_dir="./data/regression")
    # dataset = get_dataset(config)
    # print(dataset[0])

    app = AnnotationGUI()
    app.run()