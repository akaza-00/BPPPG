import warnings
warnings.filterwarnings("ignore")
from pyampd.ampd import find_peaks
import numpy as np
import scipy
from biosppy.signals import bvp
from scipy import io
from hrvanalysis import get_time_domain_features
from hrvanalysis import get_frequency_domain_features
import mat73

# properties
freq = 125
debug = False
part_id = 1
save_name = "features_{}.csv".format(part_id)
# Address of data
data_path = "data"

try:
    data_mat = mat73.loadmat(data_path + '/Part_{}.mat'.format(part_id))
    if part_id==1:
        data = data_mat['Part_1']
    if part_id==2:
        data = data_mat['Part_2']
    if part_id==3:
        data = data_mat['Part_3']
    if part_id==4:
        data = data_mat['Part_4']
except:
    data_mat = io.loadmat(data_path + '/Part_{}.mat'.format(part_id))
    if part_id==1:
        data = data_mat['Part_1'][0]
    if part_id==2:
        data = data_mat['Part_2'][0]
    if part_id==3:
        data = data_mat['Part_3'][0]
    if part_id==4:
        data = data_mat['Part_4'][0]

# Read all of subjects signals

signals = []
details = []
all_features = []

data_length = len(data)

for subject_id in range(0,data_length):    
        print('subject_id={}'.format(subject_id))        
        [PPG, BP, ECG] = data[subject_id]
        # PPG = PPG[1:int(len(PPG))]
        subject_details = [0,1,2,3,np.max(BP), np.min(BP), 6,7,8]
        details.append(subject_details)
        signals.append(PPG)

# Extracting BP-related features of each signal
for i,PPG in enumerate(signals):   
    try:    
        print("signal = {}".format(i))
        # Filtering the signal        
        # the library uses filtfilt (=> Phase = 0)
        ppg_filtered_low = bvp.st.filter_signal(signal=PPG,ftype='butter',band='lowpass',order=6,frequency=10,sampling_rate=freq)[0]
        ppg_filtered = ppg_filtered_low - scipy.signal.medfilt(ppg_filtered_low, int(15 * 1000/freq)-1)                 
        # Calculating the peaks and minimums of the signal (using AMPD)
        onsets = find_peaks(-1 * ppg_filtered)
        peaks = find_peaks(ppg_filtered)
        nn_intervals_list  = np.diff(onsets) * (1000/freq)
        time_domain_features = list(get_time_domain_features(nn_intervals_list).values())
        freq_domain_features = list(get_frequency_domain_features(nn_intervals_list).values())
         
        HR = 60/(np.median(np.diff(onsets))/freq)
        PPG = np.array(PPG)
        subject_features = []
        for onset in list(range(1,len(onsets)-1,5)):
            try:
                print(".", end="")
                segment_features = []
                for win_num in range(0,5):
                    ppg_window = ppg_filtered[onsets[onset + win_num]:onsets[onset + win_num + 1]]
                    ppg_window = ppg_window - np.min(ppg_window)
                    ppg_window = ppg_window / np.max(ppg_window)                          
                    window_features = []
                    ignore_flag = False
                    # Detecting maximum of the window and dividing the Pulse into two ascending and descending parts
                    ppg_window_systolic = np.argmax(ppg_window)
                    ppg_window_ascending = ppg_window[:ppg_window_systolic]
                    ppg_window_descending = ppg_window[ppg_window_systolic:]
                    # Fitting polynomials to each part and calculating their first and second derivatives
                    ppg_window_ascending_x = np.linspace(0,ppg_window_systolic,len(ppg_window_ascending))
                    ppg_window_descending_x = np.linspace(ppg_window_systolic,len(ppg_window),len(ppg_window_descending))
                    asc_weights = np.ones(np.shape(ppg_window_ascending))
                    asc_weights[int(0.25*len(asc_weights)):int(0.75*len(asc_weights))] = 1
                    asc_weights[int(0.4*len(asc_weights)):int(0.6*len(asc_weights))] = 1
                    des_weights = np.ones(np.shape(ppg_window_descending))
                    des_weights[int(0.25*len(des_weights)):int(0.75*len(des_weights))] = 1
                    des_weights[int(0.4*len(des_weights)):int(0.6*len(des_weights))] = 1
                    des_weights[int(0.8*len(des_weights)):int(1*len(des_weights))] = 1
                    ppg_window_ascending_poly5 = np.polyfit(ppg_window_ascending_x ,ppg_window_ascending,5,w=asc_weights)
                    ppg_window_descending_poly7 = np.polyfit(ppg_window_descending_x,ppg_window_descending,7,w=des_weights)
                    ppg_window_ascending_poly5_der = np.poly1d(np.polyder(ppg_window_ascending_poly5))
                    ppg_window_descending_poly7_der = np.poly1d(np.polyder(ppg_window_descending_poly7))
                    ppg_window_descending_poly7_der2 = np.poly1d(np.polyder(ppg_window_descending_poly7,2))
                    ppg_window_ascending_poly5_der_values = ppg_window_ascending_poly5_der(ppg_window_ascending_x)
                    ppg_window_descending_poly7_der_values = ppg_window_descending_poly7_der(ppg_window_descending_x)
                    ppg_window_descending_poly7_der2_values = ppg_window_descending_poly7_der2(ppg_window_descending_x)
                    # calculating the location of Max. slope point
                    ppg_maxslope = np.argmax(ppg_window_ascending_poly5_der_values)
                    # calculating the location of Diastolic peak, Dicrotic Notch and inflection point based on the method presented in our paper
                    ppg_window_descending_poly7_der_roots = np.sort(np.array([np.real(root) for root in np.roots(ppg_window_descending_poly7_der) if ~np.iscomplex(root)]))
                    ppg_window_descending_poly7_der_roots = np.array([root for root in ppg_window_descending_poly7_der_roots if (root>ppg_window_systolic*1.25 and root<0.95*len(ppg_window))])
                    ppg_window_descending_poly7_der_roots_der2_values = ppg_window_descending_poly7_der2(ppg_window_descending_poly7_der_roots)
                    if len(ppg_window_descending_poly7_der_roots)==0:
                        ppg_no_dia = True
                    else:
                        dia_candidates = []
                        for dia_index, dia_candidate in enumerate(ppg_window_descending_poly7_der_roots):
                            if ppg_window_descending_poly7_der_roots_der2_values[dia_index]<0:
                                dia_candidates.append(dia_candidate)
                        if len(dia_candidates)>0:
                            ppg_dia = dia_candidates[int(len(dia_candidates)/2)]
                            ppg_no_dia = False
                        else:
                            ppg_no_dia = True
                    if ppg_no_dia == True:
                        ppg_des_der2_minimums = scipy.signal.find_peaks(-1*ppg_window_descending_poly7_der2_values)[0] + ppg_window_systolic
                        dia_candidates = np.array([root for root in ppg_des_der2_minimums if (root>ppg_window_systolic*1.25 and root<0.95*len(ppg_window))])
                        if len(dia_candidates)>0:
                            ppg_dia = dia_candidates[0]            
                        else:
                            ppg_dia = ppg_window_systolic + int(len(ppg_window_descending)/2)
                    # find dicrotic notch
                    ppg_des_der2_maximums = scipy.signal.find_peaks(ppg_window_descending_poly7_der2_values)[0] + ppg_window_systolic
                    dicrotic_candidates = np.array([root for root in ppg_des_der2_maximums if (root>ppg_window_systolic*1.2 and root<0.995*ppg_dia)])
                    if len(dicrotic_candidates)>0:
                        ppg_dic = dicrotic_candidates[0]           
                    else:
                        ppg_dic = ppg_des_der2_maximums[0]
                    # find inflection point
                    ppg_window_descending_poly7_der2_roots = np.sort(np.array([np.real(root) for root in np.roots(ppg_window_descending_poly7_der2) if ~np.iscomplex(root)]))
                    inf_candidates = np.array([root for root in ppg_window_descending_poly7_der2_roots if (root>ppg_dic and root<ppg_dia)])
                    if len(inf_candidates)>0:
                        ppg_inf = inf_candidates[0]
                    else:
                        ignore_flag = True
                        ppg_inf = (ppg_dia + ppg_dic)/2
                        
                    ppg_dia = int(ppg_dia)
                    ppg_dic = int(ppg_dic)
                    ppg_inf = int(ppg_inf)
                
                    # When point detection is done, extract the features
                    window_features.extend(details[i])
                    meanBP = (1*details[i][4] + 2*details[i][5]) / 3
                    window_features.extend([0])
                    window_features.extend([meanBP])                                        
                    ppg_window_main = PPG[onsets[onset + win_num]:onsets[onset + win_num + 1]]
                    ppg_window_main = np.array(ppg_window_main)            
                    ppg_dc = np.mean(PPG)
                    ppg_window_main_rev = ppg_window_main[::-1]
                    try:
                        ppg_last_min = len(ppg_window_main)- scipy.signal.find_peaks(-1*ppg_window_main_rev)[0][0]
                    except:
                        ppg_last_min = len(ppg_window_main) -1
                    ppg_last_min = (ppg_window_main[0]+ppg_window_main[ppg_last_min])/2    
                    #
                    ppg_ac = np.median(PPG[peaks]) - np.median(PPG[onsets])
                    # modified Normalized Pulse Volume
                    mNPV1 = ppg_ac / (ppg_dc)
                    mNPV2 = ppg_ac / (ppg_dc+ppg_ac)
                    # Area related features
                    A1 = np.trapz(ppg_window_main[0:ppg_maxslope])
                    A2 = np.trapz(ppg_window_main[ppg_maxslope:ppg_window_systolic])
                    A3 = np.trapz(ppg_window_main[ppg_window_systolic:ppg_dic])
                    A4 = np.trapz(ppg_window_main[ppg_dic:ppg_inf])
                    A5 = np.trapz(ppg_window_main[ppg_inf:ppg_dia])    
                    A6 = np.trapz(ppg_window_main)
                    S1 = A1/A2
                    S2 = A1/A3
                    S3 = A1/A4
                    S4 = A1/A5
                    S5 = A1/A6
                    S6 = A2/A3
                    S7 = A2/A4
                    S8 = A2/A5
                    S9 = A2/A6
                    S10 = A3/A4
                    S11 = A3/A5
                    S12 = A3/A6
                    S13 = A4/A5
                    S14 = A4/A6
                    S15 = A5/A6
                    S16 = (A1+A2+A3+A4)/(A5+A6)
                    #
                    NA1 = np.trapz(ppg_window[0:ppg_maxslope])
                    NA2 = np.trapz(ppg_window[ppg_maxslope:ppg_window_systolic])
                    NA3 = np.trapz(ppg_window[ppg_window_systolic:ppg_dic])
                    NA4 = np.trapz(ppg_window[ppg_dic:ppg_inf])
                    NA5 = np.trapz(ppg_window[ppg_inf:ppg_dia])    
                    NA6 = np.trapz(ppg_window)
                    NS1 = NA1/NA2
                    NS2 = NA1/NA3
                    NS3 = NA1/NA4
                    NS4 = NA1/NA5
                    NS5 = NA1/NA6
                    NS6 = NA2/NA3
                    NS7 = NA2/NA4
                    NS8 = NA2/NA5
                    NS9 = NA2/NA6
                    NS10 = NA3/NA4
                    NS11 = NA3/NA5
                    NS12 = NA3/NA6
                    NS13 = NA4/NA5
                    NS14 = NA4/NA6
                    NS15 = NA5/NA6
                    NS16 = (NA1+NA2+NA3+NA4)/(NA5+NA6)
                    # Reflection Indices
                    RI1 = (ppg_window_main[ppg_maxslope] - ppg_last_min) / (ppg_window_main[ppg_window_systolic] - ppg_last_min)
                    RI2 = (ppg_window_main[ppg_dic] - ppg_last_min) / (ppg_window_main[ppg_window_systolic] - ppg_last_min)
                    RI3 = (ppg_window_main[ppg_inf] - ppg_last_min) / (ppg_window_main[ppg_window_systolic] - ppg_last_min)
                    RI4 = (ppg_window_main[ppg_dia] - ppg_last_min) / (ppg_window_main[ppg_window_systolic] - ppg_last_min)
                    # LASI
                    SI1 = details[i][2]/(ppg_dia-ppg_window_systolic)
                    SI2 = details[i][2]/(ppg_dic-ppg_window_systolic)
                    SI3 = details[i][2]/(ppg_inf-ppg_window_systolic)
                    SI4 = details[i][2]/(ppg_dia-ppg_maxslope)
                    SI5 = details[i][2]/(ppg_window_systolic-ppg_maxslope)
                    SI6 = details[i][2]/(ppg_dia-ppg_inf)
                    # Non-linear features
                    ln_mNPV1 = np.log(mNPV1)
                    ln_mNPV2 = np.log(mNPV2)
                    exp_mNPV1 = np.exp(mNPV1)
                    exp_mNPV2 = np.exp(mNPV2)
                    ln_HR = np.log(HR)
                    exp_HR = np.exp(HR)
                    ln_HRmNPV1 = np.log(mNPV1 * HR)
                    ln_HRmNPV2 = np.log(mNPV2 * HR)
                    ln_RI1 = np.log(RI1)
                    ln_RI2 = np.log(RI2)
                    ln_RI3 = np.log(RI3)
                    ln_RI4 = np.log(RI4)
                    window_features.extend([ppg_dc,ppg_ac,mNPV1,mNPV2,A1,A2,A3,A4,A5,A6,S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13,S14,S15,S16,NA1,NA2,NA3,NA4,NA5,NA6,NS1,NS2,NS3,NS4,NS5,NS6,NS7,NS8,NS9,NS10,NS11,NS12,NS13,NS14,NS15,NS16,RI1
                                            ,RI2,RI3,RI4,SI1,SI2,SI3,SI4,SI5,SI6,ln_mNPV1,ln_mNPV2,exp_mNPV1,exp_mNPV2,ln_HRmNPV1,ln_HRmNPV2,ln_RI1,ln_RI2,ln_RI3,ln_RI4,HR,ln_HR,exp_HR])
                    window_features.extend([ppg_maxslope,ppg_window_systolic,ppg_dic,ppg_dia,ppg_inf])
                    window_features.extend(time_domain_features)
                    window_features.extend(freq_domain_features)
                    segment_features.append(window_features)
                subject_features.append(np.mean(segment_features,0))            
            except:                
                print('^', end='')
                pass
        if len(np.median(subject_features,0))>0:
            all_features.append(np.median(subject_features,0))        
        print("")
        print("++")
    except:                
        print("**")
        pass        

all_features = np.array(all_features)
np.savetxt(save_name, all_features, delimiter=",")