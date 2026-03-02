
# 工作流测试
from src.data_processer.statistics.vibration_io_process.workflow import run as vibration_workflow
from src.data_processer.statistics.wind_data_io_process.workflow import run as wind_workflow


# 图表测试
from src.figs.figs_for_thesis.fig2_6_lackness_of_wind_data import Lackness_Of_Wind_Data_Analysis as fig2_6
from src.figs.figs_for_thesis.fig2_11_normal_vib_time_series_fft_3d import main as fig2_11
from src.figs.figs_for_thesis.fig2_12_VIV_vib_time_series import main as fig2_12
from src.figs.figs_for_thesis.fig2_10_normal_vib_time_series import main as fig2_10
from src.figs.figs_for_thesis.fig2_13_VIV_vib_time_series_fft_3d import main as fig2_13

if __name__ == "__main__":
    fig2_12()
