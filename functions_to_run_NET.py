"""
Functions that are used to run the NET topside model.

Last edited 04/01/2023. Author: Artem Smirnov (asmirnov@gfz-potsdam.de).
"""
import numpy as np
import pandas as pd
import h5py
import joblib
import keras

def Add_FFT(input_df: pd.DataFrame, feature_list:list, feature_lim:list, max_order: int) -> pd.DataFrame:
    """
    Adds Fourier features for given input variables and given maximum FFT orders to the input dataframe.

    :param input_df: pd.DataFrame
    :param feature_list: list
    :param feature_lim: list. Values should be "int".
    :param max_order: int

    :return: a dataframe with added FFT features.
    """
    for feature_ind in range(len(feature_list)):
        for order in range(1, max_order+1):
            # sine harmonics:
            feature_sine_name = 'sin_{}_{}'.format(order, feature_list[feature_ind])
            sine_values = np.sin(order* input_df[feature_list[feature_ind]].values/feature_lim[feature_ind] *2*np.pi)
            input_df[feature_sine_name]= sine_values
            # cosine harmonics:
            feature_cosine_name = 'cos_{}_{}'.format(order, feature_list[feature_ind])
            cosine_values = np.cos(order* input_df[feature_list[feature_ind]].values/feature_lim[feature_ind] *2*np.pi)
            input_df[feature_cosine_name]= cosine_values
            
        input_df = input_df.drop([feature_list[feature_ind]], axis=1)
    return input_df

def prepare_input(GLat: np.ndarray, GLon: np.ndarray, MLat: np.ndarray, MLon: np.ndarray, MLT: np.ndarray, DOY: np.ndarray, 
                  SYMH: np.ndarray, P107: np.ndarray, Kp: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Adds Fourier features for given input variables and given maximum FFT orders to the input dataframe.

    :params GLat, GLon, MLat, MLon, MLT, DOY, SYM-H, P10.7, Kp - of type np.ndarray

    :return: preprocessed input array for the F2-peak models
    :return: preprocessed input array for the topside models
    """
    # CONFIGURE THE PANDAS DATAFRAME FOR THE F2-PEAK:
    df_F2peak = pd.DataFrame(columns = ['GLat', 'GLon','MLat', 'MLon', 'MLT', 'DOY', 'SYM-H', 'P10.7', 'Kp'],
                            data = np.column_stack((GLat, GLon, MLat, MLon, MLT, DOY, SYMH, P107, Kp)))
    # Adding Fourier features:
    df_F2peak_with_FFT = Add_FFT(df_F2peak, ['GLat','MLat'], [360,360], 2)
    df_F2peak_with_FFT = Add_FFT(df_F2peak_with_FFT, ['GLon','MLon','DOY'], [360,360,365], 3)
    df_F2peak_with_FFT = Add_FFT(df_F2peak_with_FFT, ['MLT'], [24], 4)
    # The correct order of columns to run NmF2 and hmF2 models:
    column_names_peak = ['SYM-H', 'P10.7', 'Kp', 'sin_1_GLat', 'cos_1_GLat', 'sin_2_GLat',
            'cos_2_GLat', 'sin_1_MLat', 'cos_1_MLat', 'sin_2_MLat', 'cos_2_MLat',
            'sin_1_GLon', 'cos_1_GLon', 'sin_2_GLon', 'cos_2_GLon', 'sin_3_GLon',
            'cos_3_GLon', 'sin_1_MLon', 'cos_1_MLon', 'sin_2_MLon', 'cos_2_MLon',
            'sin_3_MLon', 'cos_3_MLon', 'sin_1_DOY', 'cos_1_DOY', 'sin_2_DOY',
            'cos_2_DOY', 'sin_3_DOY', 'cos_3_DOY', 'sin_1_MLT', 'cos_1_MLT',
            'sin_2_MLT', 'cos_2_MLT', 'sin_3_MLT', 'cos_3_MLT', 'sin_4_MLT', 'cos_4_MLT']

    # CONFIGURE THE PANDAS DATAFRAME FOR THE TOPSIDE:
    df_topside = pd.DataFrame(columns = ['GLat', 'GLon', 'toplat','toplon', 'MLat', 'MLon', 'MLT', 'DOY', 'SYM-H', 'P10.7', 'Kp'],
                            data = np.column_stack((GLat, GLon, GLat, GLon, MLat, MLon, MLT, DOY, SYMH, P107, Kp))) # During the model training, toplat and toplon variables from RO files were used due to the fact that COSMIC profiles are not exactly vertical, but during the usual model we assume the vertical profiles and therefore toplat=GLat and toplon=GLon
    
    #Adding Fourier features:
    df_topside_with_FFT = Add_FFT(df_topside, ['GLat','MLat'], [360,360], 2)
    df_topside_with_FFT = Add_FFT(df_topside_with_FFT, ['GLon','MLon','DOY'], [360,360,365], 3)
    df_topside_with_FFT = Add_FFT(df_topside_with_FFT, ['MLT'], [24], 4)
    df_topside_with_FFT = Add_FFT(df_topside_with_FFT, ['toplon'], [360], 1)
    # The correct order of columns to run H0 and dHs/dh models:
    column_names_topside=['toplat', 'SYM-H', 'P10.7', 'Kp', 'sin_1_GLat', 'cos_1_GLat',
           'sin_2_GLat', 'cos_2_GLat', 'sin_1_MLat', 'cos_1_MLat', 'sin_2_MLat',
           'cos_2_MLat', 'sin_1_GLon', 'cos_1_GLon', 'sin_2_GLon', 'cos_2_GLon',
           'sin_3_GLon', 'cos_3_GLon', 'sin_1_MLon', 'cos_1_MLon', 'sin_2_MLon',
           'cos_2_MLon', 'sin_3_MLon', 'cos_3_MLon', 'sin_1_DOY', 'cos_1_DOY',
           'sin_2_DOY', 'cos_2_DOY', 'sin_3_DOY', 'cos_3_DOY', 'sin_1_MLT',
           'cos_1_MLT', 'sin_2_MLT', 'cos_2_MLT', 'sin_3_MLT', 'cos_3_MLT', 'sin_4_MLT',
           'cos_4_MLT', 'sin_1_toplon', 'cos_1_toplon']
    
    return df_F2peak_with_FFT[column_names_peak].to_numpy(), df_topside_with_FFT[column_names_topside].to_numpy()

def NET_NmF2(input_array_NmF2: np.ndarray) -> np.ndarray:
    """
    Runs NmF2 sub-model of NET, and returns log10(NmF2).

    :param input_array_NmF2: np.ndarray

    :return: log10(NmF2), i.e., electron density of the F2-layer peak (el./cm3), in log10-scale.
    """    
    # loading the model and scalers for the input and output arrays:
    model_NmF2 = keras.models.load_model('./model_files_and_scalers/model_NmF2.h5')
    path_scaler_input_NmF2    = './model_files_and_scalers/scaler_input_NmF2.save'
    path_scaler_output_NmF2    = './model_files_and_scalers/scaler_output_NmF2.save'
    scaler_input_NmF2 = joblib.load(path_scaler_input_NmF2)
    scaler_output_NmF2 = joblib.load(path_scaler_output_NmF2)
    # normalizing the input array:    
    input_arr_normalized = scaler_input_NmF2.transform(input_array_NmF2)
    # running the model:
    NmF2_prediction_normalized = model_NmF2.predict(input_arr_normalized, verbose=1, batch_size=2**12) # this is a prediction, not training, batch size. It is put to a large value so that the model runs faster. It can be put to a smaller value if this does not fit into memory.
    # unscaling the prediction back to the values within the data range:
    NmF2_prediction = scaler_output_NmF2.inverse_transform(NmF2_prediction_normalized) # NOTE: NmF2 is predicted in log10-scale
    return NmF2_prediction[:,0]

def NET_hmF2(input_array_hmF2: np.ndarray) -> np.ndarray:
    """
    Runs hmF2 sub-model of NET.

    :param input_array_hmF2: np.ndarray

    :return: hmF2, i.e., the height of the F2-layer peak, in km.
    """    
    # loading the model and scalers for the input and output arrays:
    model_hmF2 = keras.models.load_model('./model_files_and_scalers/model_hmF2.h5')
    path_scaler_input_hmF2    = './model_files_and_scalers/scaler_input_hmF2.save'
    path_scaler_output_hmF2    = './model_files_and_scalers/scaler_output_hmF2.save'
    scaler_input_hmF2 = joblib.load(path_scaler_input_hmF2)
    scaler_output_hmF2 = joblib.load(path_scaler_output_hmF2)
    # normalizing the input array:    
    input_arr_normalized = scaler_input_hmF2.transform(input_array_hmF2)
    # running the model:
    hmF2_prediction_normalized = model_hmF2.predict(input_arr_normalized, verbose=1, batch_size=2**12) # this is a prediction, not training, batch size. It is put to a large value so that the model runs faster. It can be put to a smaller value if this does not fit into memory.
    # unscaling the prediction back to the values within the data range:
    hmF2_prediction = scaler_output_hmF2.inverse_transform(hmF2_prediction_normalized) 
    return hmF2_prediction[:,0]

def NET_H0(input_array_H0: np.ndarray) -> np.ndarray:
    """
    Runs H0 sub-model of NET.

    :param input_array_H0: np.ndarray

    :return: H0, i.e., the intercept of the linear scale height decay, in km.
    """    
    # loading the model and scalers for the input and output arrays:
    model_H0 = keras.models.load_model('./model_files_and_scalers/model_H0.h5')
    path_scaler_input_H0    = './model_files_and_scalers/scaler_input_H0.save'
    path_scaler_output_H0    = './model_files_and_scalers/scaler_output_H0.save'
    scaler_input_H0 = joblib.load(path_scaler_input_H0)
    scaler_output_H0 = joblib.load(path_scaler_output_H0)
    # normalizing the input array:    
    input_arr_normalized = scaler_input_H0.transform(input_array_H0)
    # running the model:
    H0_prediction_normalized = model_H0.predict(input_arr_normalized, verbose=1, batch_size=2**12) # this is a prediction, not training, batch size. It is put to a large value so that the model runs faster. It can be put to a smaller value if this does not fit into memory.
    # unscaling the prediction back to the values within the data range:
    H0_prediction = scaler_output_H0.inverse_transform(H0_prediction_normalized) 
    return H0_prediction[:,0]

def NET_dHs_dh(input_array_dHs_dh: np.ndarray) -> np.ndarray:
    """
    Runs dHs/dh sub-model of NET.

    :param input_array_dHs_dh: np.ndarray

    :return: dHs/dh, i.e., the slope of the linear scale height decay.
    """    
    # loading the model and scalers for the input and output arrays:
    model_dHs_dh = keras.models.load_model('./model_files_and_scalers/model_dHs_dh.h5')
    path_scaler_input_dHs_dh    = './model_files_and_scalers/scaler_input_dHs_dh.save'
    path_scaler_output_dHs_dh    = './model_files_and_scalers/scaler_output_dHs_dh.save'
    scaler_input_dHs_dh = joblib.load(path_scaler_input_dHs_dh)
    scaler_output_dHs_dh = joblib.load(path_scaler_output_dHs_dh)
    # normalizing the input array:    
    input_arr_normalized = scaler_input_dHs_dh.transform(input_array_dHs_dh)
    # running the model:
    dHs_dh_prediction_normalized = model_dHs_dh.predict(input_arr_normalized, verbose=1, batch_size=2**12) # this is a prediction, not training, batch size. It is put to a large value so that the model runs faster. It can be put to a smaller value if this does not fit into memory.
    # unscaling the prediction back to the values within the data range:
    dHs_dh_prediction = scaler_output_dHs_dh.inverse_transform(dHs_dh_prediction_normalized) 
    return dHs_dh_prediction[:,0]
    
def NET_Ne_equation1(altitude: np.ndarray, NmF2: np.ndarray, hmF2: np.ndarray, H0: np.ndarray, dHs_dh: np.ndarray) -> np.ndarray:
    """
    Combines 4 NET sub-models into a linear alpha-Chapman (sometimes also called linear vary-Chap) equation to get electron density.

    :param altitude: np.ndarray
    :param NmF2: np.ndarray
    :param hmF2: np.ndarray
    :param H0: np.ndarray
    :param dHs_dh: np.ndarray

    :return: Ne (electron density in el./cm3).
    """    
    # linear scale height decay with altitude:
    Hs_lin = dHs_dh*(altitude - hmF2) + H0
    z = (altitude - hmF2)/Hs_lin
    Ne = NmF2*np.exp(0.5*(1-z-np.exp(-z)))  
    # NET works for the topside, and therefore predictions for which h<hmF2 are put to NaN:
    mask_bottomside = (altitude<hmF2) 
    Ne[mask_bottomside] = np.nan
    return Ne
    
def run_NET(altitude: np.ndarray, GLat: np.ndarray, GLon: np.ndarray, MLat: np.ndarray, MLon: np.ndarray, MLT: np.ndarray, DOY: np.ndarray, 
            SYMH: np.ndarray, P107: np.ndarray, Kp: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Performs the complete run of NET.

    :param altitude: np.ndarray
    :params GLat, GLon, MLat, MLon, MLT, DOY, SYM-H, P10.7, Kp: of type np.ndarray

    :return: Ne (electron density in el./cm3), 2 parameters of the F2-peak (log10_NmF2 and hmF2), and 2 parameters of linear scale height decay with altitude (slope and intercept).
    """    
    # prepare the input arrays:
    input_array_F2_peak, input_array_topside = prepare_input(GLat, GLon, MLat, MLon, MLT, DOY, SYMH, P107, Kp)
    # run 4 sub-models:
    log10_NmF2_array = NET_NmF2(input_array_F2_peak)
    hmF2_array = NET_hmF2(input_array_F2_peak)
    H0_array = NET_H0(input_array_topside)
    dHs_dh_array = NET_dHs_dh(input_array_topside)
    # combine 4 values into the equation (1) from Smirnov et al., 2023:
    Ne_array = NET_Ne_equation1(altitude, 10**log10_NmF2_array, hmF2_array, H0_array, dHs_dh_array)
    
    return Ne_array, log10_NmF2_array, hmF2_array, H0_array, dHs_dh_array
