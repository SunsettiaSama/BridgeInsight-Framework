# Autor@ 猫毛
from clustering.ASC import adaptive_spectral_cluster, KNN_adaptive_spectral_cluster
from libs.utils import Data_Process, UNPACK, PlotLib, PSD_Proposer, SamplingTool
import scipy.signal as signal
import numpy as np
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
from src.libs.utils import *

def cluster_sep_data(is_test = True):

    unpacker = UNPACK()
    proposer = PSD_Proposer()

    if not is_test:
        data_root = r'F:\Research\data\苏通\VIC\\'
    else:
        data_root = r'F:\Research\Vibration Characteristics In Cable Vibration\Samples\test_samples\\'


    data_dir_lis = unpacker.VIC_Path_Lis(root = data_root)

    print('=' * 20)
    print('Sampling Data...')
    # 数据量过大，抽样一部分
    sampler = SamplingTool()
    data_dir_lis = sampler.sample_and_remain(data_dir_lis, 0.03)

    psds = []
    print('=' * 20)
    print('Spliting Data...')
    params = {
        'total_energy': [],
        'peak_frequency': [],
        'peak_magnitude': []
    }
    with tqdm(total=len(data_dir_lis)) as pbar:
        for path in data_dir_lis:
            data_slice = unpacker.File_Detach_Data(path, time_interval = 20, mode = 'VIC')
            pbar.update(1)
            if len(data_slice ) != 0:
                # 清洗nan数据
                data_slice = proposer.clean_nan(data_slice)

                sample = data_slice
                freq, psd, param = proposer.compute_psd(sample, fs=50, nperseg = 256, window='hann', return_params=True)
            
                psds.extend(psd)
                
                params['total_energy'].extend(param['total_energy'])
                params['peak_frequency'].extend(param['peak_frequency'])
                params['peak_magnitude'].extend(param['peak_magnitude'])
            else:
                continue
        
    
    # 拉平指数的数量级
    print('=' * 20)
    print('Clustering...')
    psds = proposer.log_transform(psds)
    clusters, indices_sorted = KNN_adaptive_spectral_cluster(psds, k = 2)

    fig, ax = None, None
    ploter = PlotLib()
    cluster_index = 0
    colors = [
    '#000000',  # 纯黑（最强对比）
    'red',  # 深红（与白/黑强烈对比）
    '#006400',  # 墨绿（与白/黑强烈对比）
    '#FFA500',  # 橘红（高饱和度，与冷色对比）
    '#000080',  # 深蓝（与暖色对比）
    '#800080',  # 紫红（与白/黑强烈对比）
    '#FF4500',  # 热烈红（高对比度）
    '#008080',  # 海军蓝（冷色调，与暖色对比）
    '#FFD700',  # 金色（明亮且与深色对比）
]  
    # 先生成能量总量和能量最大值频率的图
    for indices in indices_sorted:
        peak_magnitude = np.array([params['peak_magnitude'][index] for index in indices])
        peak_magnitude = proposer.log_transform(peak_magnitude)
        peak_frequency = [params['peak_frequency'][index] for index in indices]

        fig, ax = ploter.scatter(y = peak_magnitude, 
                                  x = peak_frequency, 
                                  xlabel = 'Peak Frequency (Hz)',
                                  ylabel = 'Max Energy (Log)',
                                  title = 'Cluster Distribution (k = 2)',
                                  legend = 'Cluster' + str(cluster_index),
                                  color = colors[cluster_index], 
                                  alpha = 0.7, 
                                  fig = fig, 
                                  ax = ax, 
                                  s = 2,
                                  marker = 'o',  
                                  add_fig = cluster_index == len(indices_sorted) - 1, 
                                  )
        
        cluster_index += 1

    # 再生成能量最大值所在频率和能量最大值量值的图
    fig, ax = None, None
    cluster_index = 0
    # 先生成能量总量和能量最大值频率的图
    for indices in indices_sorted:
        total_energies = np.array([params['total_energy'][index] for index in indices])
        total_energies = proposer.log_transform(total_energies)
        peak_frequency = [params['peak_frequency'][index] for index in indices]

        fig, ax = ploter.scatter(y = total_energies, 
                                  x = peak_frequency, 
                                  xlabel = 'Peak Frequency (Hz)',
                                  ylabel = 'Total Energy (Log)',
                                  title = 'Cluster Distribution (k = 2)',
                                  legend = 'Cluster' + str(cluster_index),
                                  color = colors[cluster_index], 
                                  alpha = 0.7, 
                                  fig = fig, 
                                  ax = ax, 
                                  s = 2,
                                  marker = 'o',  
                                  add_fig = cluster_index == len(indices_sorted) - 1, 
                                  )
        
        cluster_index += 1

    ploter.show()
    return 


if __name__ == '__main__':
    Wind_Rose_Map_Only()