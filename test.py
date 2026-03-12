

from src.figure_paintings.figs_for_thesis.fig2_7_wavelet_vib_sample_show import main as fig2_7_wavelet_vib_sample_show
from src.figure_paintings.figs_for_thesis.fig2_7_x_main_module_on_statistic import main as fig2_7_x_main_module_on_statistic
from src.visualize_tools.annotation_tools.annotation import AnnotationGUI
from src.data_processer.preprocess.vibration_io_process.workflow import run as run_vibration_io_process


if __name__ == "__main__":

    run_vibration_io_process(force_recompute = True) 

    # fig2_16_normal_vib_time_series()
    # fig2_8_wavelet_heursure()
    # app = AnnotationGUI()
    # app.run()
    # fig2_7_wavelet_vib_sample_show()
    # fig2_7_x_main_module_on_statistic()



