
# 工作流测试
from src.data_processer.statistics.vibration_io_process.workflow import run as vibration_workflow
from src.data_processer.statistics.wind_data_io_process.workflow import run as wind_workflow


# 图表测试

from src.figs.figs_for_thesis.fig2_3_rms_statistics import RMS_Statistics_Histogram as fig2_3    
from src.figs.figs_for_thesis.fig2_2_lackness_of_samples import Lackness_Of_Samples_Analysis as fig2_2
from src.figs.figs_for_thesis.fig2_5_rms_calendar import plot_vibration_calendar_results as fig2_5
from src.figs.figs_for_thesis.fig2_6_wind_turbulence import main as fig2_6


if __name__ == "__main__":
    fig2_6()
    # fig2_2()
    # result = wind_workflow(force_recompute=True)
    # result = vibration_workflow(force_recompute=True)

    pass



