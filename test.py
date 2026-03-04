
# 工作流测试
from src.data_processer.statistics.vibration_io_process.workflow import run as vibration_workflow
from src.data_processer.statistics.wind_data_io_process.workflow import run as wind_workflow


# 图表测试
from src.figs.figs_for_thesis.fig2_6_lackness_of_wind_data import Lackness_Of_Wind_Data_Analysis as fig2_6
from src.figs.figs_for_thesis.fig2_6_1_wavelet_vib_sample_show import main as fig2_6_1_wavelet_vib_sample_show
from src.figs.figs_for_thesis.fig2_8_wavelet_heursure import main as fig2_8_wavelet_heursure
from src.figs.figs_for_thesis.fig2_11_normal_vib_time_series_fft_3d import main as fig2_11
from src.figs.figs_for_thesis.fig2_12_VIV_vib_time_series import main as fig2_12
from src.figs.figs_for_thesis.fig2_10_normal_vib_time_series import main as fig2_10
from src.figs.figs_for_thesis.fig2_13_VIV_vib_time_series_fft_3d import main as fig2_13
from src.figs.figs_for_thesis.fig2_14_VIV_MultiMode import main as fig2_14
from src.figs.figs_for_thesis.fig2_6_2_cal_visushrink import main as fig2_6_2_cal_visushrink



from src.visualize_tools.annotation import AnnotationGUI
if __name__ == "__main__":
    # app = AnnotationGUI()
    # app.run()
    # fig2_14()
    # fig2_15()
    
    # app = AnnotationGUI()
    # app.run()
    
    fig2_8_wavelet_heursure()
