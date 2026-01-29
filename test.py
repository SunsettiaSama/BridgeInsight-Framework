


from src.data_processer.statistics.vibration_io_process.workflow import run as workflow
from src.figs.figs_for_thesis.fig2_3_rms_statistics import RMS_Statistics_Histogram as fig2_3    
from src.figs.figs_for_thesis.fig2_2_lackness_of_samples import Lackness_Of_Samples_Analysis as fig2_2
from src.figs.figs_for_thesis.fig2_5_rms_calendar import plot_vibration_calendar_results as fig2_5

if __name__ == "__main__":

    # fig2_2()
    result = workflow(force_recompute=True)
    fig2_5()
    pass