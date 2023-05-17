# -*- coding: utf-8 -*-

def database(benchmark, path):
    """Database Info"""
    if benchmark == 'TID2013':
        data_dir = path + 'tid2013/distorted_images/'
        data_info = './benchmark/TID2013fullinfo.mat'
    elif benchmark == 'CSIQ':
        data_dir = path + 'CSIQ/'
        data_info = './benchmark/CSIQfullinfo.mat'
    elif benchmark == 'KADID':
        data_dir = path + 'kadid10k/images/'
        data_info = './benchmark/KADID-10K.mat'
    elif benchmark == 'LIVE':
        data_dir = path + 'live/'
        data_info = './benchmark/LIVEfullinfo.mat'
    elif benchmark == 'LIVEC':
        data_dir = path + 'LIVEC/Images/'
        data_info = './benchmark/CLIVEinfo.mat'
    elif benchmark == 'KonIQ':
        data_dir = path + 'koniq10k/512x384/'
        data_info = './benchmark/KonIQ-10k.mat'
    elif benchmark == 'BID':
        data_dir = path + 'BID/BID/ImageDatabase/'
        data_info = './benchmark/BIDinfo.mat'
    else:
        data_dir = None
        data_info = None

    return data_info, data_dir


def distortion_type(database):
    """Distortion type info"""
    if database == 'LIVE':
        num_type = 5
    elif database == 'CSIQ':
        num_type = 6
    elif database == 'TID2013':
        num_type = 24
    elif database == 'KADID':
        num_type = 25
    else:
        num_type = None

    return num_type
