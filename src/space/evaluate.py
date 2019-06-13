'''
Created on 2019. 5. 27.

@author: Inwoo Chung (gutomitai@gmail.com)
'''

import json
import argparse

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import quad
import h5py
from scipy.linalg import norm

from yolov3_detect import BoundBox, bbox_iou

# Constants.
DEBUG = True

MODE_CAL_MAP_FD = 'cal_map_fd'
MODE_CAL_FACE_PAIRS_DISTS = 'cal_face_pairs_dists'
MODE_CAL_VAL_FAR = 'cal_VAL_FAR'
MODE_CAL_ACC_FI = 'cal_acc_fi'

def cal_mAP_fd(gt_path, sol_path, iou_th):
    # Load ground truth, predicted results and calculate IoU.
    sol_df = pd.read_csv(sol_path, header=None)
    sol_df = pd.concat([sol_df, pd.DataFrame(np.zeros(shape=(sol_df.shape[0]), dtype=np.float64), columns=[6])], axis=1) # IoU
    sol_df.iat[:, 6] = -1.0
    sol_df_g = sol_df.groupby(0) #?
        
    gt_df = pd.read_csv(gt_path)
    gt_df = pd.concat([gt_df, pd.DataFrame(np.zeros(shape=(gt_df.shape[0]), dtype=np.float64), columns=[7])], axis=1) # IoU
    gt_df.iat[:, 7] = -1.0
    gt_df_g = gt_df.groupby('FILE') #?
    
    for k, image_id in enumerate(list(gt_df_g.groups.keys())):
        if DEBUG: print(k, '/', len(gt_df_g.groups.keys()), ':', image_id, end='\r')
        df = gt_df_g.get_group(image_id)
        
        try:
            rel_sol_df = sol_df_g.get_group(image_id)
        except KeyError:
            continue
        
        gt_ious = {}
        
        for i in range(df.shape[0]):
            gt_ious[i] = []
            gt_sample = df.iloc[i]
            gt_sample_bb = BoundBox(gt_sample.iloc[3]
                                      , gt_sample.iloc[4]
                                      , gt_sample.iloc[3] + gt_sample.iloc[5]
                                      , gt_sample.iloc[4] + gt_sample.iloc[6])
                    
            # Check exception.
            if rel_sol_df.shape[0] == 0: continue
        
            # Calculate IoUs between a gt region and detected regions
            for j in range(rel_sol_df.shape[0]):
                rel_sol = rel_sol_df.iloc[j]
                rel_sol_bb = BoundBox(rel_sol[1]
                                      , rel_sol[2]
                                      , rel_sol[1] + rel_sol[3]
                                      , rel_sol[2] + rel_sol[4])
            
                iou = bbox_iou(gt_sample_bb, rel_sol_bb)
                if iou > 0.: #? 
                    gt_ious[i].append((i, j, iou))
        
        total_gt_ious = []
        for i in gt_ious:
            total_gt_ious += gt_ious[i]
            
        if len(total_gt_ious) == 0: continue
            
        total_gt_ious_df = pd.DataFrame(total_gt_ious)
        
        # IoU descending order sorting.
        total_gt_ious_df = total_gt_ious_df.sort_values(by=2, ascending=False)
        
        # Determine IoU for each detected region.
        while total_gt_ious_df.shape[0] != 0: #?
            df_p = total_gt_ious_df.iloc[0]
            i = int(df_p.iloc[0])
            j = int(df_p.iloc[1])
            iou = df_p.iloc[2]
            
            #df.iat[i, -1] = iou 
            rel_sol_df.iat[j, -1] = iou 
            
            # Remove assigned samples.
            total_gt_ious_df = total_gt_ious_df[total_gt_ious_df[0] != i]
            total_gt_ious_df = total_gt_ious_df[total_gt_ious_df[1] != j]
            
        if k == 0:
            res_df = rel_sol_df
        else:
            res_df = pd.concat([res_df, rel_sol_df])
        
    # Get the p-r curve.
    # Sort the solution in confidence descending order.
    res_df = res_df.sort_values(by=5, ascending=False)
    ps = []
    rs = []
    tp_count = 0
    count = 0
    gt_count = gt_df.shape[0]
    
    for i in range(res_df.shape[0]):
        count +=1
        
        if i < res_df.shape[0] and res_df.iloc[i, 6] >= iou_th:
            tp_count += 1
            
        ps.append(tp_count / count)
        rs.append(tp_count / gt_count)
    
    ps = np.asarray(ps)
    rs = np.asarray(rs)
    
    func = interp1d(rs, ps)
    mAP = quad(lambda x: func(x), rs[0], rs[-1])
    
    return ps, rs, mAP[0]

def cal_face_pairs_dists():
    # Create same and different face identity pair sets.
    db = pd.read_csv('subject_image_db.csv')
    db = db.iloc[:, 1:]
    db_g = db.groupby('subject_id')
    same_dists = []
    diff_dists = []
    
    with h5py.File('subject_facial_ids.h5', 'r') as f:
        subject_ids = list(db_g.groups.keys())
        
        # Same face identity pairs.
        print('Same face identity pairs.')
        for c, subject_id in enumerate(subject_ids):
            if DEBUG: print(c + 1, '/', len(subject_ids), end='\r')
            
            if subject_id == -1:
                continue
            
            # Get face images of a subject id.
            df = db_g.get_group(subject_id)
            file_names = list(df.iloc[:, 1])
            
            # Check exception.
            if len(file_names) < 2: continue
            
            for i in range(len(file_names) - 1): 
                for j in range(i + 1, len(file_names)): 
                    same_dists.append(norm(f[file_names[i]].value - f[file_names[j]].value)) 
    
        if DEBUG: print()
    
        # Determine pairs of different face identity randomly.
        idxes = range(len(subject_ids))
        num_pairs = len(subject_ids) // 2
        pairs = np.random.choice(idxes, size=(num_pairs, 2), replace=False)
            
        # Different face identity pairs.
        print('Different face identity pairs.')
        for i in range(pairs.shape[0]):
            if DEBUG: print(i + 1, '/', pairs.shape[0], end='\r')
            
            k = pairs[i, 0]
            l = pairs[i, 1]
    
            if subject_ids[k] == -1 or subject_ids[l] == -1:
                continue            
    
            ref_df = db_g.get_group(subject_ids[k])
            ref_file_names = list(ref_df.iloc[:, 1])
    
            comp_df = db_g.get_group(subject_ids[l])
            comp_file_names = list(comp_df.iloc[:, 1])
                            
            for ref_fn in ref_file_names: 
                for comp_fn in comp_file_names: 
                    diff_dists.append(norm(f[ref_fn].value - f[comp_fn].value))            
            
    same_dists = np.asarray(same_dists)
    diff_dists = np.asarray(diff_dists)
    
    with h5py.File('face_pairs_dists.h5', 'w') as f:
        f['same_dists'] = same_dists
        f['diff_dists'] = diff_dists

    return same_dists, diff_dists

def cal_VAL_FAR(sim_th_range):
    # Calculate distances about same and different face identity pairs.
    same_dists, diff_dists = cal_face_pairs_dists()
    sim_ths = []
    vals = []
    fars = []
    
    for sim_th in sim_th_range:
        sim_ths.append(sim_th)
        
        s_r = same_dists <= sim_th
        s_r = s_r.astype(np.int64)
        vals.append(s_r.sum() / same_dists.shape[0])

        d_r = diff_dists <= sim_th
        d_r = d_r.astype(np.int64)
        fars.append(d_r.sum() / diff_dists.shape[0])    

    sim_ths = np.asarray(sim_ths)
    vals = np.asarray(vals)
    fars = np.asarray(fars)

    with h5py.File('val_far.h5', 'w') as f:
        f['sim_ths'] = same_dists
        f['vals'] = vars
        f['fars'] = fars
    
    return sim_ths, vals, fars

def cal_acc_fi(gt_path, sol_path, iou_th):
    # Confusion matrix's components. 
    tp = 0
    fp = 0
    tn = 0 #?
    fn = 0 #?
    
    # Load ground truth, predicted results and calculate IoU.
    sol_df = pd.read_csv(sol_path, header=None)
    sol_df = pd.concat([sol_df, pd.DataFrame(np.zeros(shape=(sol_df.shape[0]), dtype=np.float64), columns=[7])], axis=1) # IoU
    sol_df_g = sol_df.groupby(0) #?
        
    gt_df = pd.read_csv(gt_path)
    gt_df = pd.concat([gt_df, pd.DataFrame(np.zeros(shape=(gt_df.shape[0]), dtype=np.float64), columns=[7])], axis=1) # IoU
    gt_df_g = gt_df.groupby('FILE') #?
    
    print(iou_th)
    for k, image_id in enumerate(list(gt_df_g.groups.keys())):
        print(k, '/', len(gt_df_g.groups.keys()), ':', image_id, end='\r')
        df = gt_df_g.get_group(image_id)
        
        try:
            rel_sol_df = sol_df_g.get_group(image_id)
        except KeyError:
            for i in range(df.shape[0]):
                if df.iloc[i, 2] == -1:
                    tn += 1 # Right?
                else:
                    fn += 1 # Right?        
                    
            continue
        
        gt_ious = {}
        
        for i in range(df.shape[0]):
            gt_ious[i] = []
            gt_sample = df.iloc[i]
            gt_sample_bb = BoundBox(gt_sample.iloc[3]
                                      , gt_sample.iloc[4]
                                      , gt_sample.iloc[3] + gt_sample.iloc[5]
                                      , gt_sample.iloc[4] + gt_sample.iloc[6])
                           
            # Calculate IoUs between a gt region and detected regions
            for j in range(rel_sol_df.shape[0]):
                rel_sol = rel_sol_df.iloc[j]
                rel_sol_bb = BoundBox(rel_sol[2]
                                      , rel_sol[3]
                                      , rel_sol[2] + rel_sol[4]
                                      , rel_sol[3] + rel_sol[5])
            
                iou = bbox_iou(gt_sample_bb, rel_sol_bb)
                if iou > 0.: #? 
                    gt_ious[i].append((i, j, iou))
        
        total_gt_ious = []
        for i in gt_ious:
            total_gt_ious += gt_ious[i]
            
        if len(total_gt_ious) == 0: continue
            
        total_gt_ious_df = pd.DataFrame(total_gt_ious)
        
        # IoU descending order sorting.
        total_gt_ious_df = total_gt_ious_df.sort_values(by=2, ascending=False)
        
        # Determine IoU and an identification result for each detected region.
        while total_gt_ious_df.shape[0] != 0: #?
            df_p = total_gt_ious_df.iloc[0]
            i = int(df_p.iloc[0])
            j = int(df_p.iloc[1])
            iou = df_p.iloc[2]
            
            if iou >= iou_th and df.iloc[i, 2] != -1 and rel_sol_df.iloc[j, 1] != -1 \
                and df.iloc[i, 2] == rel_sol_df.iloc[j, 1]: #?
                tp += 1
            elif iou >= iou_th and rel_sol_df.iloc[j, 1] != -1 and (df.iloc[i, 2] != rel_sol_df.iloc[j, 1]):
                fp += 1
            elif df.iloc[i, 2] == -1:
                tn += 1
            else:
                fn += 1
            
            df.iat[i, -1] = 1    
            rel_sol_df.iat[j, -1] = 1 # Checking flag.
            
            # Remove assigned samples.
            total_gt_ious_df = total_gt_ious_df[total_gt_ious_df[0] != i]
            total_gt_ious_df = total_gt_ious_df[total_gt_ious_df[1] != j]
        
        for i in range(df.shape[0]):
            if df.iloc[i, -1] == 1: continue
            if df.iloc[i, 2] == -1:
                tn += 1 # Right?
            else:
                fn += 1 # Right?        
        
        for i in range(rel_sol_df.shape[0]):
            if rel_sol_df.iloc[i, -1] == 1: continue
            if rel_sol_df.iloc[i, 1] == -1:
                tn += 1 # Right?
            else:
                fp += 1 # Right?

    acc = (tp + tn) / (tp + tn + fp + fn)
    return tp, fp, tn, fn, acc        
                    
def main(args):
    # Parse arguments.
    mode = args.mode
    gt_path = args.gt_path
    sol_path = args.sol_path
    
    if mode == MODE_CAL_MAP_FD:
        ps_ls = []
        rs_ls = []
        mAP_ls = []
        
        for iou_th in np.arange(0.5, 1.0, 0.05):
            ps, rs, mAP = cal_mAP_fd(gt_path, sol_path, iou_th)
            if DEBUG: print('{0:1.2f}'.format(iou_th), mAP)
            ps_ls.append(ps)
            rs_ls.append(rs)
            mAP_ls.append(mAP)
        
        ps_ls = np.asarray(ps_ls)
        rs_ls = np.asarray(rs_ls)
        mAP_ls = np.asarray(mAP_ls)
        
        with h5py.File('p_r_curve.h5', 'w') as f:
            f['ps_ls'] = ps_ls
            f['rs_ls'] = rs_ls
            f['mAP_ls'] = mAP_ls
    elif mode == MODE_CAL_FACE_PAIRS_DISTS:
        cal_face_pairs_dists()
    elif mode == MODE_CAL_VAL_FAR:
        sim_th_range = np.arange(0.1, 1.1, 0.1)
        cal_VAL_FAR(sim_th_range)
    elif mode == MODE_CAL_ACC_FI:
        tp_ls = []
        fp_ls = []
        tn_ls = []
        fn_ls = []
        acc_ls = []
        
        for iou_th in np.arange(0.5, 1.0, 0.05):
            tp, fp, tn, fn, acc = cal_acc_fi(gt_path, sol_path, iou_th)
            if DEBUG: print('\n{0:1.2f}'.format(iou_th), tp, fp, tn, fn, acc)
            
            tp_ls.append(tp)
            fp_ls.append(fp)
            tn_ls.append(tn)
            fn_ls.append(fn)            
            acc_ls.append(acc)
        
        tp_ls = np.asarray(tp_ls)
        fp_ls = np.asarray(fp_ls)
        tn_ls = np.asarray(tn_ls)
        fn_ls = np.asarray(fn_ls)
        acc_ls = np.asarray(acc_ls)
        
        with h5py.File('fi_acc.h5', 'w') as f:
            f['tp_ls'] = tp_ls
            f['fp_ls'] = fp_ls
            f['tn_ls'] = tn_ls
            f['fn_ls'] = fn_ls
            f['acc_ls'] = acc_ls       
    
if __name__ == '__main__':
    # Parse arguments.
    parser = argparse.ArgumentParser(description='Evaluate face recognition\'s performance metric.')
    
    parser.add_argument('-m', '--mode')
    parser.add_argument('-g', '--gt_path')
    parser.add_argument('-s', '--sol_path')
    
    args = parser.parse_args()
    
    main(args)