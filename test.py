




from src.data_processer.datasets.TestDatasets.RegressionDataset import RegressionDataset
from src.config.data_processer.datasets.TestDatasets.RegressionDatasetConfig import RegressionDatasetConfig
from src.data_processer.datasets.data_factory import get_dataset

from src.visualize_tools.annotation_tools.annotation import AnnotationGUI


from src.figure_paintings.figs_for_thesis.fig2_22_vehicle_RWIV_Sample import main as fig2_22
from src.figure_paintings.figs_for_thesis.fig2_23_RWIV_Inplane_Sample import main as fig2_23
from src.figure_paintings.figs_for_thesis.fig2_24_RWIV_Outplane_Sample import main as fig2_24
from src.figure_paintings.figs_for_thesis.fig2_25_RWIV_Outplan_Sample_ABNORMAL import main as fig2_25
from src.figure_paintings.figs_for_thesis.fig2_26_VIV_With_RWIV import main as fig2_26
from src.figure_paintings.figs_for_thesis.fig2_27_dataset_display import main as fig2_27

import torch
import numpy as np



if __name__ == "__main__":
    # config = RegressionDatasetConfig(data_dir="./data/regression")
    # dataset = get_dataset(config)
    # print(dataset[0])

    # app = AnnotationGUI()
    # app.run()
    # main()
    # fig2_22()
    # fig2_23()
    # fig2_24()
    # fig2_25()
    # fig2_26()
    # fig2_27() 
