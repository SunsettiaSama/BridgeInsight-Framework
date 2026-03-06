

from src.figs.figs_for_thesis.fig2_8_wavelet_heursure import main as fig2_8_wavelet_heursure
from src.figs.figs_for_thesis.fig2_14_rebuild_wind_velocity import main as fig2_14_rebuild_wind_velocity
from src.figs.figs_for_thesis.fig2_16_normal_vib_time_series import main as fig2_16_normal_vib_time_series

from src.visualize_tools.annotation_tools.annotation import AnnotationGUI

if __name__ == "__main__":
    # fig2_16_normal_vib_time_series()
    fig2_8_wavelet_heursure()
    app = AnnotationGUI()
    app.run()



