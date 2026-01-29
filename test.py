





# from src.figs.fig7_mecc_result import run_pareto_verification_and_plot as fig7_2
# from src.figs.fig7_ecc_result import run_ecc_param_search_and_plot as fig7_1
# from src.figs.fig5_timeseries_vib_display import Fig5

from src.figs.figs_for_thesis.fig2_2_lackness_of_samples import Lackness_Of_Samples_Analysis as fig2_2
from src.figs.figs_for_thesis.fig2_3_vibration_sample import Vibration_Below_Threshold_Analysis as fig2_3
from src.figs.figs_for_thesis.fig2_4_time_series_rms import RMS_Statistics_Histogram as fig2_4
from src.figs.figs_for_thesis.fig2_5_rms_calendar import plot_vibration_calendar_results as fig2_5

# 测试振动路径工作流
from src.data_processer.statistics.vibration_io_process.step0_get_vib_data import get_all_vibration_files as step0
# from src.data_processer.statistics.vibration_io_process.step1_sensor_filter import get_rms_statistics_json as step1

from src.data_processer.statistics.vibration_io_process.workflow import run_vibration_data_workflow as workflow
from src.data_processer.statistics.vibration_io_process.step1_lackness_filter import run_lackness_filter as step1

if __name__ == "__main__":
    # metadatas = workflow()
    fig2_5()
    pass

# run_ecc_param_search_and_plot()


# fig = Fig5()

# fig.plot()

# from src.figs.fig3_wind_velocity_distribution import Wind_Speed_Histogram

# Wind_Speed_Histogram()




