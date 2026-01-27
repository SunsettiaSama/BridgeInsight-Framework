





# from src.figs.fig7_mecc_result import run_pareto_verification_and_plot as fig7_2
# from src.figs.fig7_ecc_result import run_ecc_param_search_and_plot as fig7_1
# from src.figs.fig5_timeseries_vib_display import Fig5

from src.figs.figs_for_thesis.fig2_2_lackness_of_samples import Lackness_Of_Samples_Analysis as fig2_2
from src.figs.figs_for_thesis.fig2_3_vibration_sample import Vibration_Below_Threshold_Analysis as fig2_3
from src.figs.figs_for_thesis.fig2_4_time_series_rms import RMS_Statistics_Histogram as fig2_4
from src.figs.figs_for_thesis.fig2_5_rms_calendar import plot_daily_occurrence_calendar as fig2_5

from src.data_processer.statistics.rms_statistics import main as get_rms_statistics_json

if __name__ == "__main__":
    # fig2_3()
    # fig2_2()
    
    # get_rms_statistics_json()
    fig2_5()



# run_ecc_param_search_and_plot()


# fig = Fig5()

# fig.plot()

# from src.figs.fig3_wind_velocity_distribution import Wind_Speed_Histogram

# Wind_Speed_Histogram()




