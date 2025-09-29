#!/usr/bin/env python3
"""
Process diagnosis, past history, physical exam, and lab files in the eICU patient dataset

Diagnosis file processing (diagnosis.csv -> post_diagnosis.csv):
1. Remove rows with empty icd9code
2. Deduplicate by diagnosisstring, keep unique diagnoses

Past history file processing (pastHistory.csv -> post_pastHistory.csv):
1. Deduplicate by pasthistorypath, pasthistoryvalue, pasthistoryvaluetext
2. Remove rows where pasthistoryvaluetext is "Performed"

Physical exam file processing (physicalExam.csv -> post_physicalExam.csv):
1. Deduplicate by physicalexampath, physicalexamvalue, physicalexamtext
2. Remove rows where physicalexamvalue is "Performed - Structured"

Lab file processing (lab.csv -> post_lab.csv):
1. Bedside glucose tests (glucose) are aggregated every 4 hours, keep the mean value
2. Blood gas analysis is merged into complete records by time point
3. Routine tests keep the latest value every 6 hours
4. Remove duplicate tests within 30 minutes
5. Prefer the latest revised value

Count the change in data quantity before and after processing
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path

def process_single_diagnosis_file(diagnosis_file_path, output_file_path):
    """
    Process a single diagnosis file
    
    Args:
        diagnosis_file_path: Input diagnosis file path
        output_file_path: Output file path
    
    Returns:
        tuple: (count before processing, count after processing)
    """
    try:
        # Read original diagnosis file
        df = pd.read_csv(diagnosis_file_path)
        original_count = len(df)
        
        if original_count == 0:
            # If file is empty, create empty output file
            df.to_csv(output_file_path, index=False)
            return 0, 0
        
        # 1. Remove rows with empty icd9code
        # 检查icd9code列是否存在
        if 'icd9code' not in df.columns:
            print(f"Warning: No icd9code column in {diagnosis_file_path}")
            return original_count, 0
        
        # 过滤掉icd9code为空的行
        df_filtered = df.dropna(subset=['icd9code'])
        df_filtered = df_filtered[df_filtered['icd9code'].str.strip() != '']
        
        # 2. Deduplicate by diagnosisstring
        if 'diagnosisstring' in df_filtered.columns:
            df_dedup = df_filtered.drop_duplicates(subset=['diagnosisstring'], keep='first')
        else:
            print(f"Warning: No diagnosisstring column in {diagnosis_file_path}")
            df_dedup = df_filtered
        
        processed_count = len(df_dedup)
        
        # Save processed file
        df_dedup.to_csv(output_file_path, index=False)
        
        return original_count, processed_count
        
    except Exception as e:
        print(f"Error processing file {diagnosis_file_path}: {e}")
        return 0, 0

def process_single_pasthistory_file(pasthistory_file_path, output_file_path):
    """
    Process a single past history file
    
    Args:
        pasthistory_file_path: Input past history file path
        output_file_path: Output file path
    
    Returns:
        tuple: (count before processing, count after processing)
    """
    try:
        # 读取原始既往病史文件
        df = pd.read_csv(pasthistory_file_path)
        original_count = len(df)
        
        if original_count == 0:
            # 如果文件为空，创建空的输出文件
            df.to_csv(output_file_path, index=False)
            return 0, 0
        
        # 检查必要的列是否存在
        required_cols = ['pasthistorypath', 'pasthistoryvalue', 'pasthistoryvaluetext']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Warning: Missing columns in {pasthistory_file_path}: {missing_cols}")
            return original_count, 0
        
        # 1. Deduplicate by pasthistorypath, pasthistoryvalue, pasthistoryvaluetext
        df_dedup = df.drop_duplicates(subset=['pasthistorypath', 'pasthistoryvalue', 'pasthistoryvaluetext'], keep='first')
        
        # 2. Remove rows where pasthistoryvaluetext is "Performed"
        df_filtered = df_dedup[df_dedup['pasthistoryvaluetext'] != 'Performed']
        
        processed_count = len(df_filtered)
        
        # 保存处理后的文件
        df_filtered.to_csv(output_file_path, index=False)
        
        return original_count, processed_count
        
    except Exception as e:
        print(f"Error processing file {pasthistory_file_path}: {e}")
        return 0, 0

def process_single_physicalexam_file(physicalexam_file_path, output_file_path):
    """
    Process a single physical exam file
    
    Args:
        physicalexam_file_path: Input physical exam file path
        output_file_path: Output file path
    
    Returns:
        tuple: (count before processing, count after processing)
    """
    try:
        # 读取原始体格检查文件
        df = pd.read_csv(physicalexam_file_path)
        original_count = len(df)
        
        if original_count == 0:
            # 如果文件为空，创建空的输出文件
            df.to_csv(output_file_path, index=False)
            return 0, 0
        
        # 检查必要的列是否存在
        required_cols = ['physicalexampath', 'physicalexamvalue', 'physicalexamtext']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Warning: Missing columns in {physicalexam_file_path}: {missing_cols}")
            return original_count, 0
        
        # 1. Deduplicate by physicalexampath, physicalexamvalue, physicalexamtext
        df_dedup = df.drop_duplicates(subset=['physicalexampath', 'physicalexamvalue', 'physicalexamtext'], keep='first')
        
        # 2. Remove rows where physicalexamvalue is "Performed - Structured"
        df_filtered = df_dedup[df_dedup['physicalexamvalue'] != 'Performed - Structured']
        
        processed_count = len(df_filtered)
        
        # 保存处理后的文件
        df_filtered.to_csv(output_file_path, index=False)
        
        return original_count, processed_count
        
    except Exception as e:
        print(f"Error processing file {physicalexam_file_path}: {e}")
        return 0, 0

def process_single_lab_file(lab_file_path, output_file_path):
    """
    Process a single lab file
    
    Optimization strategies:
    1. Bedside glucose tests (glucose) are aggregated every 4 hours, keep the mean value
    2. Blood gas analysis is merged into complete records by time point
    3. Routine tests keep the latest value every 6 hours
    4. Remove duplicate tests within 30 minutes
    5. Prefer the latest revised value
    
    Args:
        lab_file_path: Input lab file path
        output_file_path: Output file path
    
    Returns:
        tuple: (count before processing, count after processing)
    """
    try:
        # 读取原始实验室检查文件
        df = pd.read_csv(lab_file_path)
        original_count = len(df)
        
        if original_count == 0:
            # 如果文件为空，创建空的输出文件
            df.to_csv(output_file_path, index=False)
            return 0, 0
        
        # 检查必要的列是否存在
        required_cols = ['labname', 'labresult', 'labresultoffset', 'labresultrevisedoffset', 'labtypeid']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Warning: Missing columns in {lab_file_path}: {missing_cols}")
            return original_count, 0
        
        # 预处理：处理NaN值和异常数据
        df = df.dropna(subset=['labname', 'labresult'])
        df = df[df['labresult'].notna()]
        
        if len(df) == 0:
            # 如果过滤后为空，创建空的输出文件
            df.to_csv(output_file_path, index=False)
            return original_count, 0
        
        processed_parts = []
        
        # 1. Bedside glucose tests (aggregate every 4 hours)
        glucose_df = df[df['labname'].str.contains('glucose', case=False, na=False)]
        if not glucose_df.empty:
            # 转换为4小时时间段
            glucose_df = glucose_df.copy()
            glucose_df['hour_group'] = (glucose_df['labresultrevisedoffset'] // 240) * 240  # 4小时=240分钟
            
            # 按4小时分组聚合
            glucose_aggregated = glucose_df.groupby(['hour_group', 'labname']).agg({
                'labresult': ['mean', 'min', 'max', 'count'],
                'labresultoffset': 'first',
                'labresultrevisedoffset': 'first',
                'labtypeid': 'first',
                'labmeasurenamesystem': 'first' if 'labmeasurenamesystem' in glucose_df.columns else lambda x: '',
                'labmeasurenameinterface': 'first' if 'labmeasurenameinterface' in glucose_df.columns else lambda x: '',
                'labid': 'first' if 'labid' in glucose_df.columns else lambda x: ''
            }).reset_index()
            
            # 重构列名
            glucose_aggregated.columns = [
                'hour_group', 'labname', 'labresult_mean', 'labresult_min', 'labresult_max', 'labresult_count',
                'labresultoffset', 'labresultrevisedoffset', 'labtypeid', 'labmeasurenamesystem', 'labmeasurenameinterface', 'labid'
            ]
            
            # 使用平均值作为主要结果，添加标记表明这是聚合数据
            glucose_aggregated['labresult'] = glucose_aggregated['labresult_mean']
            glucose_aggregated['labname'] = glucose_aggregated['labname'] + '_4h_avg'
            glucose_aggregated['labresultoffset'] = glucose_aggregated['hour_group']
            glucose_aggregated['labresultrevisedoffset'] = glucose_aggregated['hour_group']
            
            # 选择需要的列
            glucose_final = glucose_aggregated[['labname', 'labresult', 'labresultoffset', 'labresultrevisedoffset', 
                                               'labtypeid', 'labmeasurenamesystem', 'labmeasurenameinterface', 'labid']]
            processed_parts.append(glucose_final)
        
        # 2. Blood gas analysis (labtypeid=7) - merge by time point
        blood_gas_df = df[df['labtypeid'] == 7]
        non_glucose_blood_gas = blood_gas_df[~blood_gas_df['labname'].str.contains('glucose', case=False, na=False)]
        
        if not non_glucose_blood_gas.empty:
            # 按时间点（30分钟窗口）分组
            non_glucose_blood_gas = non_glucose_blood_gas.copy()
            non_glucose_blood_gas['time_group'] = (non_glucose_blood_gas['labresultrevisedoffset'] // 30) * 30
            
            # 每个时间组内，每种检测项目保留最新的一个
            blood_gas_dedup = non_glucose_blood_gas.sort_values('labresultrevisedoffset').groupby(['time_group', 'labname']).last().reset_index()
            processed_parts.append(blood_gas_dedup.drop('time_group', axis=1))
        
        # 3. Other routine tests - keep latest value every 6 hours
        other_df = df[(df['labtypeid'] != 7) & (~df['labname'].str.contains('glucose', case=False, na=False))]
        
        if not other_df.empty:
            # 转换为6小时时间段
            other_df = other_df.copy()
            other_df['time_group'] = (other_df['labresultrevisedoffset'] // 360) * 360  # 6小时=360分钟
            
            # 每个时间组内，每种检测项目保留最新的一个
            other_dedup = other_df.sort_values('labresultrevisedoffset').groupby(['time_group', 'labname']).last().reset_index()
            processed_parts.append(other_dedup.drop('time_group', axis=1))
        
        # 合并所有处理后的数据
        if processed_parts:
            df_processed = pd.concat(processed_parts, ignore_index=True)
            
            # 最终去重：移除完全相同的记录
            df_processed = df_processed.drop_duplicates()
            
            # 按labresultoffset时间排序（从小到大升序），模拟实验室检查的时间顺序
            df_processed = df_processed.sort_values(['labresultoffset', 'labname'], ascending=[True, True]).reset_index(drop=True)
        else:
            df_processed = pd.DataFrame()
        
        processed_count = len(df_processed)
        
        # 保存处理后的文件
        df_processed.to_csv(output_file_path, index=False)
        
        return original_count, processed_count
        
    except Exception as e:
        print(f"Error processing file {lab_file_path}: {e}")
        return 0, 0

def main():
    """主函数"""
    # 设置数据目录
    data_dir = "/Users/wushuai/Documents/research/dataset/eicu_patient_datasets"
    
    # 获取所有患者目录
    patient_dirs = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()]
    
    print(f"Found {len(patient_dirs)} patient directories")
    
    # 统计信息
    stats = {
        'total_patients': 0,
        'processed_patients': 0,
        # 诊断文件统计
        'diagnosis_original_counts': [],
        'diagnosis_processed_counts': [],
        # 既往病史文件统计
        'pasthistory_original_counts': [],
        'pasthistory_processed_counts': [],
        # 体格检查文件统计
        'physicalexam_original_counts': [],
        'physicalexam_processed_counts': [],
        # 实验室检查文件统计
        'lab_original_counts': [],
        'lab_processed_counts': [],
        'patient_ids': [],  # 添加患者ID列表
        'patients_with_no_valid_diagnosis': 0,
        'patients_with_no_valid_pasthistory': 0,
        'patients_with_no_valid_physicalexam': 0,
        'patients_with_no_valid_lab': 0,
        'error_patients': []
    }
    
    # 处理每个患者的诊断文件、既往病史文件、体格检查文件和实验室检查文件
    for patient_id in tqdm(patient_dirs, desc="Processing patient data files"):
        patient_dir = os.path.join(data_dir, patient_id)
        
        # 诊断文件路径
        diagnosis_file = os.path.join(patient_dir, 'diagnosis.csv')
        diagnosis_output_file = os.path.join(patient_dir, 'post_diagnosis.csv')
        
        # 既往病史文件路径
        pasthistory_file = os.path.join(patient_dir, 'pastHistory.csv')
        pasthistory_output_file = os.path.join(patient_dir, 'post_pastHistory.csv')
        
        # 体格检查文件路径
        physicalexam_file = os.path.join(patient_dir, 'physicalExam.csv')
        physicalexam_output_file = os.path.join(patient_dir, 'post_physicalExam.csv')
        
        # 实验室检查文件路径
        lab_file = os.path.join(patient_dir, 'lab.csv')
        lab_output_file = os.path.join(patient_dir, 'post_lab.csv')
        
        stats['total_patients'] += 1
        
        # 处理诊断文件
        diagnosis_original_count = 0
        diagnosis_processed_count = 0
        if os.path.exists(diagnosis_file):
            try:
                diagnosis_original_count, diagnosis_processed_count = process_single_diagnosis_file(diagnosis_file, diagnosis_output_file)
                if diagnosis_processed_count == 0:
                    stats['patients_with_no_valid_diagnosis'] += 1
            except Exception as e:
                stats['error_patients'].append(f"{patient_id}: Diagnosis file processing error - {e}")
        else:
            stats['error_patients'].append(f"{patient_id}: Diagnosis file not found")
        
        # 处理既往病史文件
        pasthistory_original_count = 0
        pasthistory_processed_count = 0
        if os.path.exists(pasthistory_file):
            try:
                pasthistory_original_count, pasthistory_processed_count = process_single_pasthistory_file(pasthistory_file, pasthistory_output_file)
                if pasthistory_processed_count == 0:
                    stats['patients_with_no_valid_pasthistory'] += 1
            except Exception as e:
                stats['error_patients'].append(f"{patient_id}: Past history file processing error - {e}")
        else:
            stats['error_patients'].append(f"{patient_id}: Past history file not found")
        
        # 处理体格检查文件
        physicalexam_original_count = 0
        physicalexam_processed_count = 0
        if os.path.exists(physicalexam_file):
            try:
                physicalexam_original_count, physicalexam_processed_count = process_single_physicalexam_file(physicalexam_file, physicalexam_output_file)
                if physicalexam_processed_count == 0:
                    stats['patients_with_no_valid_physicalexam'] += 1
            except Exception as e:
                stats['error_patients'].append(f"{patient_id}: Physical exam file processing error - {e}")
        else:
            stats['error_patients'].append(f"{patient_id}: Physical exam file not found")
        
        # 处理实验室检查文件
        lab_original_count = 0
        lab_processed_count = 0
        if os.path.exists(lab_file):
            try:
                lab_original_count, lab_processed_count = process_single_lab_file(lab_file, lab_output_file)
                if lab_processed_count == 0:
                    stats['patients_with_no_valid_lab'] += 1
            except Exception as e:
                stats['error_patients'].append(f"{patient_id}: Lab file processing error - {e}")
        else:
            stats['error_patients'].append(f"{patient_id}: Lab file not found")
        
        # 记录统计信息
        stats['diagnosis_original_counts'].append(diagnosis_original_count)
        stats['diagnosis_processed_counts'].append(diagnosis_processed_count)
        stats['pasthistory_original_counts'].append(pasthistory_original_count)
        stats['pasthistory_processed_counts'].append(pasthistory_processed_count)
        stats['physicalexam_original_counts'].append(physicalexam_original_count)
        stats['physicalexam_processed_counts'].append(physicalexam_processed_count)
        stats['lab_original_counts'].append(lab_original_count)
        stats['lab_processed_counts'].append(lab_processed_count)
        stats['patient_ids'].append(patient_id)
        stats['processed_patients'] += 1
    
    # 计算统计指标
    diagnosis_original_counts = np.array(stats['diagnosis_original_counts'])
    diagnosis_processed_counts = np.array(stats['diagnosis_processed_counts'])
    pasthistory_original_counts = np.array(stats['pasthistory_original_counts'])
    pasthistory_processed_counts = np.array(stats['pasthistory_processed_counts'])
    physicalexam_original_counts = np.array(stats['physicalexam_original_counts'])
    physicalexam_processed_counts = np.array(stats['physicalexam_processed_counts'])
    lab_original_counts = np.array(stats['lab_original_counts'])
    lab_processed_counts = np.array(stats['lab_processed_counts'])
    
    print("\n" + "="*60)
    print("Patient Data File Processing Statistics Report")
    print("="*60)
    
    print(f"Total patients: {stats['total_patients']}")
    print(f"Successfully processed patients: {stats['processed_patients']}")
    print(f"Patients with no valid diagnosis: {stats['patients_with_no_valid_diagnosis']}")
    print(f"Patients with no valid past history: {stats['patients_with_no_valid_pasthistory']}")
    print(f"Patients with no valid physical exam: {stats['patients_with_no_valid_physicalexam']}")
    print(f"Patients with no valid lab tests: {stats['patients_with_no_valid_lab']}")
    print(f"Patients with processing errors: {len(stats['error_patients'])}")
    
    # ========== Diagnosis file statistics ==========
    if len(diagnosis_original_counts) > 0:
        print("\n" + "="*40)
        print("Diagnosis File Processing Statistics")
        print("="*40)
        
        print("\nDiagnosis count before processing:")
        print(f"  Min: {diagnosis_original_counts.min()}")
        print(f"  Max: {diagnosis_original_counts.max()}")
        print(f"  Mean: {diagnosis_original_counts.mean():.2f}")
        print(f"  Median: {np.median(diagnosis_original_counts):.2f}")
        print(f"  Std: {diagnosis_original_counts.std():.2f}")
        print(f"  Total diagnoses: {diagnosis_original_counts.sum()}")
        
        print("\nDiagnosis count after processing:")
        print(f"  Min: {diagnosis_processed_counts.min()}")
        print(f"  Max: {diagnosis_processed_counts.max()}")
        print(f"  Mean: {diagnosis_processed_counts.mean():.2f}")
        print(f"  Median: {np.median(diagnosis_processed_counts):.2f}")
        print(f"  Std: {diagnosis_processed_counts.std():.2f}")
        print(f"  Total diagnoses: {diagnosis_processed_counts.sum()}")
        
        # 计算数据减少比例
        total_diagnosis_original = diagnosis_original_counts.sum()
        total_diagnosis_processed = diagnosis_processed_counts.sum()
        diagnosis_reduction_rate = (total_diagnosis_original - total_diagnosis_processed) / total_diagnosis_original * 100 if total_diagnosis_original > 0 else 0
        
        print(f"\nDiagnosis data reduction:")
        print(f"  Original total diagnoses: {total_diagnosis_original}")
        print(f"  Processed total diagnoses: {total_diagnosis_processed}")
        print(f"  Reduced count: {total_diagnosis_original - total_diagnosis_processed}")
        print(f"  Reduction rate: {diagnosis_reduction_rate:.2f}%")
        
        # 分布统计
        print(f"\nDiagnosis count distribution (before processing):")
        print(f"  0 diagnoses: {(diagnosis_original_counts == 0).sum()} patients")
        print(f"  1-10 diagnoses: {((diagnosis_original_counts >= 1) & (diagnosis_original_counts <= 10)).sum()} patients")
        print(f"  11-20 diagnoses: {((diagnosis_original_counts >= 11) & (diagnosis_original_counts <= 20)).sum()} patients")
        print(f"  21-50 diagnoses: {((diagnosis_original_counts >= 21) & (diagnosis_original_counts <= 50)).sum()} patients")
        print(f"  50+ diagnoses: {(diagnosis_original_counts > 50).sum()} patients")
        
        print(f"\nDiagnosis count distribution (after processing):")
        print(f"  0 diagnoses: {(diagnosis_processed_counts == 0).sum()} patients")
        print(f"  1-10 diagnoses: {((diagnosis_processed_counts >= 1) & (diagnosis_processed_counts <= 10)).sum()} patients")
        print(f"  11-20 diagnoses: {((diagnosis_processed_counts >= 11) & (diagnosis_processed_counts <= 20)).sum()} patients")
        print(f"  21-50 diagnoses: {((diagnosis_processed_counts >= 21) & (diagnosis_processed_counts <= 50)).sum()} patients")
        print(f"  50+ diagnoses: {(diagnosis_processed_counts > 50).sum()} patients")
    
    # ========== Past history file statistics ==========
    if len(pasthistory_original_counts) > 0:
        print("\n" + "="*40)
        print("Past History File Processing Statistics")
        print("="*40)
        
        print("\nPast history record count before processing:")
        print(f"  Min: {pasthistory_original_counts.min()}")
        print(f"  Max: {pasthistory_original_counts.max()}")
        print(f"  Mean: {pasthistory_original_counts.mean():.2f}")
        print(f"  Median: {np.median(pasthistory_original_counts):.2f}")
        print(f"  Std: {pasthistory_original_counts.std():.2f}")
        print(f"  Total past history records: {pasthistory_original_counts.sum()}")
        
        print("\nPast history record count after processing:")
        print(f"  Min: {pasthistory_processed_counts.min()}")
        print(f"  Max: {pasthistory_processed_counts.max()}")
        print(f"  Mean: {pasthistory_processed_counts.mean():.2f}")
        print(f"  Median: {np.median(pasthistory_processed_counts):.2f}")
        print(f"  Std: {pasthistory_processed_counts.std():.2f}")
        print(f"  Total past history records: {pasthistory_processed_counts.sum()}")
        
        # 计算数据减少比例
        total_pasthistory_original = pasthistory_original_counts.sum()
        total_pasthistory_processed = pasthistory_processed_counts.sum()
        pasthistory_reduction_rate = (total_pasthistory_original - total_pasthistory_processed) / total_pasthistory_original * 100 if total_pasthistory_original > 0 else 0
        
        print(f"\nPast history data reduction:")
        print(f"  Original total past history records: {total_pasthistory_original}")
        print(f"  Processed total past history records: {total_pasthistory_processed}")
        print(f"  Reduced count: {total_pasthistory_original - total_pasthistory_processed}")
        print(f"  Reduction rate: {pasthistory_reduction_rate:.2f}%")
        
        # 分布统计
        print(f"\nPast history record count distribution (before processing):")
        print(f"  0 records: {(pasthistory_original_counts == 0).sum()} patients")
        print(f"  1-5 records: {((pasthistory_original_counts >= 1) & (pasthistory_original_counts <= 5)).sum()} patients")
        print(f"  6-10 records: {((pasthistory_original_counts >= 6) & (pasthistory_original_counts <= 10)).sum()} patients")
        print(f"  11-20 records: {((pasthistory_original_counts >= 11) & (pasthistory_original_counts <= 20)).sum()} patients")
        print(f"  20+ records: {(pasthistory_original_counts > 20).sum()} patients")
        
        print(f"\nPast history record count distribution (after processing):")
        print(f"  0 records: {(pasthistory_processed_counts == 0).sum()} patients")
        print(f"  1-5 records: {((pasthistory_processed_counts >= 1) & (pasthistory_processed_counts <= 5)).sum()} patients")
        print(f"  6-10 records: {((pasthistory_processed_counts >= 6) & (pasthistory_processed_counts <= 10)).sum()} patients")
        print(f"  11-20 records: {((pasthistory_processed_counts >= 11) & (pasthistory_processed_counts <= 20)).sum()} patients")
        print(f"  20+ records: {(pasthistory_processed_counts > 20).sum()} patients")
    
    # ========== Physical exam file statistics ==========
    if len(physicalexam_original_counts) > 0:
        print("\n" + "="*40)
        print("Physical Exam File Processing Statistics")
        print("="*40)
        
        print("\nPhysical exam record count before processing:")
        print(f"  Min: {physicalexam_original_counts.min()}")
        print(f"  Max: {physicalexam_original_counts.max()}")
        print(f"  Mean: {physicalexam_original_counts.mean():.2f}")
        print(f"  Median: {np.median(physicalexam_original_counts):.2f}")
        print(f"  Std: {physicalexam_original_counts.std():.2f}")
        print(f"  Total physical exam records: {physicalexam_original_counts.sum()}")
        
        print("\nPhysical exam record count after processing:")
        print(f"  Min: {physicalexam_processed_counts.min()}")
        print(f"  Max: {physicalexam_processed_counts.max()}")
        print(f"  Mean: {physicalexam_processed_counts.mean():.2f}")
        print(f"  Median: {np.median(physicalexam_processed_counts):.2f}")
        print(f"  Std: {physicalexam_processed_counts.std():.2f}")
        print(f"  Total physical exam records: {physicalexam_processed_counts.sum()}")
        
        # 计算数据减少比例
        total_physicalexam_original = physicalexam_original_counts.sum()
        total_physicalexam_processed = physicalexam_processed_counts.sum()
        physicalexam_reduction_rate = (total_physicalexam_original - total_physicalexam_processed) / total_physicalexam_original * 100 if total_physicalexam_original > 0 else 0
        
        print(f"\nPhysical exam data reduction:")
        print(f"  Original total physical exam records: {total_physicalexam_original}")
        print(f"  Processed total physical exam records: {total_physicalexam_processed}")
        print(f"  Reduced count: {total_physicalexam_original - total_physicalexam_processed}")
        print(f"  Reduction rate: {physicalexam_reduction_rate:.2f}%")
        
        # 分布统计
        print(f"\nPhysical exam record count distribution (before processing):")
        print(f"  0 records: {(physicalexam_original_counts == 0).sum()} patients")
        print(f"  1-10 records: {((physicalexam_original_counts >= 1) & (physicalexam_original_counts <= 10)).sum()} patients")
        print(f"  11-20 records: {((physicalexam_original_counts >= 11) & (physicalexam_original_counts <= 20)).sum()} patients")
        print(f"  21-50 records: {((physicalexam_original_counts >= 21) & (physicalexam_original_counts <= 50)).sum()} patients")
        print(f"  50+ records: {(physicalexam_original_counts > 50).sum()} patients")
        
        print(f"\nPhysical exam record count distribution (after processing):")
        print(f"  0 records: {(physicalexam_processed_counts == 0).sum()} patients")
        print(f"  1-10 records: {((physicalexam_processed_counts >= 1) & (physicalexam_processed_counts <= 10)).sum()} patients")
        print(f"  11-20 records: {((physicalexam_processed_counts >= 11) & (physicalexam_processed_counts <= 20)).sum()} patients")
        print(f"  21-50 records: {((physicalexam_processed_counts >= 21) & (physicalexam_processed_counts <= 50)).sum()} patients")
        print(f"  50+ records: {(physicalexam_processed_counts > 50).sum()} patients")
    
    # ========== Lab file statistics ==========
    if len(lab_original_counts) > 0:
        print("\n" + "="*40)
        print("Lab File Processing Statistics")
        print("="*40)
        
        print("\nLab record count before processing:")
        print(f"  Min: {lab_original_counts.min()}")
        print(f"  Max: {lab_original_counts.max()}")
        print(f"  Mean: {lab_original_counts.mean():.2f}")
        print(f"  Median: {np.median(lab_original_counts):.2f}")
        print(f"  Std: {lab_original_counts.std():.2f}")
        print(f"  Total lab records: {lab_original_counts.sum()}")
        
        print("\nLab record count after processing:")
        print(f"  Min: {lab_processed_counts.min()}")
        print(f"  Max: {lab_processed_counts.max()}")
        print(f"  Mean: {lab_processed_counts.mean():.2f}")
        print(f"  Median: {np.median(lab_processed_counts):.2f}")
        print(f"  Std: {lab_processed_counts.std():.2f}")
        print(f"  Total lab records: {lab_processed_counts.sum()}")
        
        # 计算数据减少比例
        total_lab_original = lab_original_counts.sum()
        total_lab_processed = lab_processed_counts.sum()
        lab_reduction_rate = (total_lab_original - total_lab_processed) / total_lab_original * 100 if total_lab_original > 0 else 0
        
        print(f"\nLab data reduction:")
        print(f"  Original total lab records: {total_lab_original}")
        print(f"  Processed total lab records: {total_lab_processed}")
        print(f"  Reduced count: {total_lab_original - total_lab_processed}")
        print(f"  Reduction rate: {lab_reduction_rate:.2f}%")
        
        # 分布统计
        print(f"\nLab record count distribution (before processing):")
        print(f"  0 records: {(lab_original_counts == 0).sum()} patients")
        print(f"  1-50 records: {((lab_original_counts >= 1) & (lab_original_counts <= 50)).sum()} patients")
        print(f"  51-100 records: {((lab_original_counts >= 51) & (lab_original_counts <= 100)).sum()} patients")
        print(f"  101-200 records: {((lab_original_counts >= 101) & (lab_original_counts <= 200)).sum()} patients")
        print(f"  201-500 records: {((lab_original_counts >= 201) & (lab_original_counts <= 500)).sum()} patients")
        print(f"  500+ records: {(lab_original_counts > 500).sum()} patients")
        
        print(f"\nLab record count distribution (after processing):")
        print(f"  0 records: {(lab_processed_counts == 0).sum()} patients")
        print(f"  1-50 records: {((lab_processed_counts >= 1) & (lab_processed_counts <= 50)).sum()} patients")
        print(f"  51-100 records: {((lab_processed_counts >= 51) & (lab_processed_counts <= 100)).sum()} patients")
        print(f"  101-200 records: {((lab_processed_counts >= 101) & (lab_processed_counts <= 200)).sum()} patients")
        print(f"  201-500 records: {((lab_processed_counts >= 201) & (lab_processed_counts <= 500)).sum()} patients")
        print(f"  500+ records: {(lab_processed_counts > 500).sum()} patients")
    
    # ========== 示例患者信息 ==========
    if len(diagnosis_original_counts) > 0:
        # 打印有11-20个诊断的患者ID
        patient_ids = np.array(stats['patient_ids'])
        patients_11_20_mask = (diagnosis_processed_counts >= 11) & (diagnosis_processed_counts <= 20)
        patients_11_20_ids = patient_ids[patients_11_20_mask]
        patients_11_20_counts = diagnosis_processed_counts[patients_11_20_mask]
        
        if len(patients_11_20_ids) > 0:
            print(f"\nPatients with 11-20 diagnoses:")
            for pid, count in zip(patients_11_20_ids, patients_11_20_counts):
                print(f"  Patient ID: {pid}, Diagnosis count: {count}")
        else:
            print(f"\nNo patients with 11-20 diagnoses")
    
    # 保存详细统计信息到文件
    detailed_stats = {
        'summary': {
            'total_patients': stats['total_patients'],
            'processed_patients': stats['processed_patients'],
            'patients_with_no_valid_diagnosis': stats['patients_with_no_valid_diagnosis'],
            'patients_with_no_valid_pasthistory': stats['patients_with_no_valid_pasthistory'],
            'patients_with_no_valid_physicalexam': stats['patients_with_no_valid_physicalexam'],
            'patients_with_no_valid_lab': stats['patients_with_no_valid_lab'],
            'error_count': len(stats['error_patients'])
        },
        'diagnosis_stats': {
            'original_diagnosis_stats': {
                'min': int(diagnosis_original_counts.min()) if len(diagnosis_original_counts) > 0 else 0,
                'max': int(diagnosis_original_counts.max()) if len(diagnosis_original_counts) > 0 else 0,
                'mean': float(diagnosis_original_counts.mean()) if len(diagnosis_original_counts) > 0 else 0,
                'median': float(np.median(diagnosis_original_counts)) if len(diagnosis_original_counts) > 0 else 0,
                'std': float(diagnosis_original_counts.std()) if len(diagnosis_original_counts) > 0 else 0,
                'total': int(diagnosis_original_counts.sum()) if len(diagnosis_original_counts) > 0 else 0
            },
            'processed_diagnosis_stats': {
                'min': int(diagnosis_processed_counts.min()) if len(diagnosis_processed_counts) > 0 else 0,
                'max': int(diagnosis_processed_counts.max()) if len(diagnosis_processed_counts) > 0 else 0,
                'mean': float(diagnosis_processed_counts.mean()) if len(diagnosis_processed_counts) > 0 else 0,
                'median': float(np.median(diagnosis_processed_counts)) if len(diagnosis_processed_counts) > 0 else 0,
                'std': float(diagnosis_processed_counts.std()) if len(diagnosis_processed_counts) > 0 else 0,
                'total': int(diagnosis_processed_counts.sum()) if len(diagnosis_processed_counts) > 0 else 0
            },
            'diagnosis_reduction_stats': {
                'total_reduction': int(diagnosis_original_counts.sum() - diagnosis_processed_counts.sum()) if len(diagnosis_original_counts) > 0 else 0,
                'reduction_rate_percent': float((diagnosis_original_counts.sum() - diagnosis_processed_counts.sum()) / diagnosis_original_counts.sum() * 100) if len(diagnosis_original_counts) > 0 and diagnosis_original_counts.sum() > 0 else 0
            }
        },
        'pasthistory_stats': {
            'original_pasthistory_stats': {
                'min': int(pasthistory_original_counts.min()) if len(pasthistory_original_counts) > 0 else 0,
                'max': int(pasthistory_original_counts.max()) if len(pasthistory_original_counts) > 0 else 0,
                'mean': float(pasthistory_original_counts.mean()) if len(pasthistory_original_counts) > 0 else 0,
                'median': float(np.median(pasthistory_original_counts)) if len(pasthistory_original_counts) > 0 else 0,
                'std': float(pasthistory_original_counts.std()) if len(pasthistory_original_counts) > 0 else 0,
                'total': int(pasthistory_original_counts.sum()) if len(pasthistory_original_counts) > 0 else 0
            },
            'processed_pasthistory_stats': {
                'min': int(pasthistory_processed_counts.min()) if len(pasthistory_processed_counts) > 0 else 0,
                'max': int(pasthistory_processed_counts.max()) if len(pasthistory_processed_counts) > 0 else 0,
                'mean': float(pasthistory_processed_counts.mean()) if len(pasthistory_processed_counts) > 0 else 0,
                'median': float(np.median(pasthistory_processed_counts)) if len(pasthistory_processed_counts) > 0 else 0,
                'std': float(pasthistory_processed_counts.std()) if len(pasthistory_processed_counts) > 0 else 0,
                'total': int(pasthistory_processed_counts.sum()) if len(pasthistory_processed_counts) > 0 else 0
            },
            'pasthistory_reduction_stats': {
                'total_reduction': int(pasthistory_original_counts.sum() - pasthistory_processed_counts.sum()) if len(pasthistory_original_counts) > 0 else 0,
                'reduction_rate_percent': float((pasthistory_original_counts.sum() - pasthistory_processed_counts.sum()) / pasthistory_original_counts.sum() * 100) if len(pasthistory_original_counts) > 0 and pasthistory_original_counts.sum() > 0 else 0
            }
        },
        'physicalexam_stats': {
            'original_physicalexam_stats': {
                'min': int(physicalexam_original_counts.min()) if len(physicalexam_original_counts) > 0 else 0,
                'max': int(physicalexam_original_counts.max()) if len(physicalexam_original_counts) > 0 else 0,
                'mean': float(physicalexam_original_counts.mean()) if len(physicalexam_original_counts) > 0 else 0,
                'median': float(np.median(physicalexam_original_counts)) if len(physicalexam_original_counts) > 0 else 0,
                'std': float(physicalexam_original_counts.std()) if len(physicalexam_original_counts) > 0 else 0,
                'total': int(physicalexam_original_counts.sum()) if len(physicalexam_original_counts) > 0 else 0
            },
            'processed_physicalexam_stats': {
                'min': int(physicalexam_processed_counts.min()) if len(physicalexam_processed_counts) > 0 else 0,
                'max': int(physicalexam_processed_counts.max()) if len(physicalexam_processed_counts) > 0 else 0,
                'mean': float(physicalexam_processed_counts.mean()) if len(physicalexam_processed_counts) > 0 else 0,
                'median': float(np.median(physicalexam_processed_counts)) if len(physicalexam_processed_counts) > 0 else 0,
                'std': float(physicalexam_processed_counts.std()) if len(physicalexam_processed_counts) > 0 else 0,
                'total': int(physicalexam_processed_counts.sum()) if len(physicalexam_processed_counts) > 0 else 0
            },
            'physicalexam_reduction_stats': {
                'total_reduction': int(physicalexam_original_counts.sum() - physicalexam_processed_counts.sum()) if len(physicalexam_original_counts) > 0 else 0,
                'reduction_rate_percent': float((physicalexam_original_counts.sum() - physicalexam_processed_counts.sum()) / physicalexam_original_counts.sum() * 100) if len(physicalexam_original_counts) > 0 and physicalexam_original_counts.sum() > 0 else 0
            }
        },
        'lab_stats': {
            'original_lab_stats': {
                'min': int(lab_original_counts.min()) if len(lab_original_counts) > 0 else 0,
                'max': int(lab_original_counts.max()) if len(lab_original_counts) > 0 else 0,
                'mean': float(lab_original_counts.mean()) if len(lab_original_counts) > 0 else 0,
                'median': float(np.median(lab_original_counts)) if len(lab_original_counts) > 0 else 0,
                'std': float(lab_original_counts.std()) if len(lab_original_counts) > 0 else 0,
                'total': int(lab_original_counts.sum()) if len(lab_original_counts) > 0 else 0
            },
            'processed_lab_stats': {
                'min': int(lab_processed_counts.min()) if len(lab_processed_counts) > 0 else 0,
                'max': int(lab_processed_counts.max()) if len(lab_processed_counts) > 0 else 0,
                'mean': float(lab_processed_counts.mean()) if len(lab_processed_counts) > 0 else 0,
                'median': float(np.median(lab_processed_counts)) if len(lab_processed_counts) > 0 else 0,
                'std': float(lab_processed_counts.std()) if len(lab_processed_counts) > 0 else 0,
                'total': int(lab_processed_counts.sum()) if len(lab_processed_counts) > 0 else 0
            },
            'lab_reduction_stats': {
                'total_reduction': int(lab_original_counts.sum() - lab_processed_counts.sum()) if len(lab_original_counts) > 0 else 0,
                'reduction_rate_percent': float((lab_original_counts.sum() - lab_processed_counts.sum()) / lab_original_counts.sum() * 100) if len(lab_original_counts) > 0 and lab_original_counts.sum() > 0 else 0
            }
        },
        'errors': stats['error_patients']
    }
    
    # 保存统计结果到JSON文件
    stats_file = os.path.join(data_dir, 'data_processing_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_stats, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed statistics saved to: {stats_file}")
    
    if stats['error_patients']:
        print(f"\nPatients with processing errors:")
        for error in stats['error_patients'][:10]:  # 只显示前10个错误
            print(f"  {error}")
        if len(stats['error_patients']) > 10:
            print(f"  ...还有 {len(stats['error_patients']) - 10} 个错误")
    
    print("\nProcessing complete!")

def analyze_disease_distribution():
    """Analyze disease distribution"""
    data_dir = "/Users/wushuai/Documents/research/dataset/eicu_patient_datasets"
    
    print("\n" + "="*60)
    print("Disease Distribution Analysis Report")
    print("="*60)
    
    # 收集所有处理后的诊断数据
    all_diagnosis_data = []
    patient_dirs = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()]
    
    print(f"Start analyzing disease distribution for {len(patient_dirs)} patients...")
    
    for patient_id in tqdm(patient_dirs, desc="Collecting disease data"):
        patient_dir = os.path.join(data_dir, patient_id)
        post_diagnosis_file = os.path.join(patient_dir, 'post_diagnosis.csv')
        
        if os.path.exists(post_diagnosis_file):
            try:
                df = pd.read_csv(post_diagnosis_file)
                if not df.empty and 'diagnosisstring' in df.columns and 'icd9code' in df.columns:
                    # 添加患者ID列
                    df['patient_id'] = patient_id
                    all_diagnosis_data.append(df)
            except Exception as e:
                print(f"Error reading diagnosis data for patient {patient_id}: {e}")
    
    if not all_diagnosis_data:
        print("No valid diagnosis data found")
        return
    
    # 合并所有数据
    combined_df = pd.concat(all_diagnosis_data, ignore_index=True)
    print(f"\nTotal {len(combined_df)} diagnosis records collected")
    print(f"Involving {combined_df['patient_id'].nunique()} patients")
    
    # 1. 基于 diagnosisstring 的疾病分布分析
    print("\n1. Disease distribution analysis based on diagnosisstring:")
    diagnosis_counts = combined_df['diagnosisstring'].value_counts()
    
    print(f"  - Total {len(diagnosis_counts)} different diagnoses")
    print(f"  - Top 10 common diagnoses:")
    for i, (diagnosis, count) in enumerate(diagnosis_counts.head(10).items(), 1):
        percentage = (count / len(combined_df)) * 100
        print(f"    {i:2d}. {diagnosis[:60]}{'...' if len(diagnosis) > 60 else ''}")
        print(f"        Count: {count} ({percentage:.2f}%)")
    
    # 保存完整的诊断分布
    diagnosis_stats_file = os.path.join(data_dir, 'diagnosis_string_distribution.csv')
    diagnosis_counts.to_csv(diagnosis_stats_file, header=['count'])
    print(f"  - Complete diagnosis distribution saved to: {diagnosis_stats_file}")
    
    # 2. 基于 ICD 编码的疾病分布分析
    print("\n2. Disease distribution analysis based on ICD coding:")
    
    # 解析ICD编码（使用和 eicu_llm_dataset.py 一致的解析方式）
    def parse_icd_codes(icd_string):
        """解析ICD代码字符串，分离ICD9和ICD10代码"""
        if pd.isna(icd_string) or not icd_string:
            return "", ""
        
        # 分割ICD9和ICD10代码 (格式: "icd9_code, icd10_code")
        codes = str(icd_string).split(', ')
        icd9_code = codes[0].strip() if len(codes) > 0 else ""
        icd10_code = codes[1].strip() if len(codes) > 1 else ""
        
        return icd9_code, icd10_code
    
    icd9_codes = []
    icd10_codes = []
    
    for icd_string in combined_df['icd9code'].dropna():
        icd9_code, icd10_code = parse_icd_codes(icd_string)
        if icd9_code:
            icd9_codes.append(icd9_code)
        if icd10_code:
            icd10_codes.append(icd10_code)
    
    # ICD9 分析
    icd9_counts = pd.Series(icd9_codes).value_counts()
    print(f"\n  ICD9 编码分析:")
    print(f"  - 总共有 {len(icd9_counts)} 种不同的 ICD9 编码")
    print(f"  - 总 ICD9 编码出现次数: {len(icd9_codes)}")
    print(f"  - 最常见的10种 ICD9 编码:")
    for i, (code, count) in enumerate(icd9_counts.head(10).items(), 1):
        percentage = (count / len(icd9_codes)) * 100
        print(f"    {i:2d}. {code}: {count} 次 ({percentage:.2f}%)")
    
    # ICD10 分析
    if icd10_codes:
        icd10_counts = pd.Series(icd10_codes).value_counts()
        print(f"\n  ICD10 编码分析:")
        print(f"  - 总共有 {len(icd10_counts)} 种不同的 ICD10 编码")
        print(f"  - 总 ICD10 编码出现次数: {len(icd10_codes)}")
        print(f"  - 最常见的10种 ICD10 编码:")
        for i, (code, count) in enumerate(icd10_counts.head(10).items(), 1):
            percentage = (count / len(icd10_codes)) * 100
            print(f"    {i:2d}. {code}: {count} 次 ({percentage:.2f}%)")
    else:
        print(f"\n  没有找到 ICD10 编码数据")
    
    # 保存ICD编码分布
    icd9_stats_file = os.path.join(data_dir, 'icd9_distribution.csv')
    icd9_counts.to_csv(icd9_stats_file, header=['count'])
    print(f"  - ICD9 编码分布已保存到: {icd9_stats_file}")
    
    if icd10_codes:
        icd10_stats_file = os.path.join(data_dir, 'icd10_distribution.csv')
        icd10_counts.to_csv(icd10_stats_file, header=['count'])
        print(f"  - ICD10 编码分布已保存到: {icd10_stats_file}")
    
    # 2.5 分析诊断字符串和ICD编码的映射关系
    print("\n2.5 诊断字符串与ICD编码映射分析:")
    
    # 创建诊断字符串到ICD编码的映射
    diagnosis_to_icd9 = {}
    diagnosis_to_icd10 = {}
    
    for _, row in combined_df.iterrows():
        diagnosis = row['diagnosisstring']
        icd_string = str(row['icd9code'])
        
        icd9_code, icd10_code = parse_icd_codes(icd_string)
        
        if icd9_code:
            if diagnosis not in diagnosis_to_icd9:
                diagnosis_to_icd9[diagnosis] = set()
            diagnosis_to_icd9[diagnosis].add(icd9_code)
        
        if icd10_code:
            if diagnosis not in diagnosis_to_icd10:
                diagnosis_to_icd10[diagnosis] = set()
            diagnosis_to_icd10[diagnosis].add(icd10_code)
    
    # 找出一个诊断对应多个ICD编码的情况
    multi_icd9_diagnoses = {diag: codes for diag, codes in diagnosis_to_icd9.items() if len(codes) > 1}
    multi_icd10_diagnoses = {diag: codes for diag, codes in diagnosis_to_icd10.items() if len(codes) > 1}
    
    print(f"  - 一个诊断对应多个ICD9编码的情况: {len(multi_icd9_diagnoses)} 种")
    if multi_icd9_diagnoses:
        print("    示例:")
        for i, (diag, codes) in enumerate(list(multi_icd9_diagnoses.items())[:5]):
            print(f"      {diag}: {list(codes)}")
    
    print(f"  - 一个诊断对应多个ICD10编码的情况: {len(multi_icd10_diagnoses)} 种")
    if multi_icd10_diagnoses:
        print("    示例:")
        for i, (diag, codes) in enumerate(list(multi_icd10_diagnoses.items())[:5]):
            print(f"      {diag}: {list(codes)}")
    
    # 创建ICD编码到诊断字符串的反向映射
    icd9_to_diagnoses = {}
    icd10_to_diagnoses = {}
    
    for diagnosis, codes in diagnosis_to_icd9.items():
        for code in codes:
            if code not in icd9_to_diagnoses:
                icd9_to_diagnoses[code] = set()
            icd9_to_diagnoses[code].add(diagnosis)
    
    for diagnosis, codes in diagnosis_to_icd10.items():
        for code in codes:
            if code not in icd10_to_diagnoses:
                icd10_to_diagnoses[code] = set()
            icd10_to_diagnoses[code].add(diagnosis)
    
    # 找出一个ICD编码对应多个诊断的情况
    multi_diag_icd9 = {code: diags for code, diags in icd9_to_diagnoses.items() if len(diags) > 1}
    multi_diag_icd10 = {code: diags for code, diags in icd10_to_diagnoses.items() if len(diags) > 1}
    
    print(f"\n  - 一个ICD9编码对应多个诊断的情况: {len(multi_diag_icd9)} 种")
    if multi_diag_icd9:
        print("    示例 (显示前5个):")
        for i, (code, diags) in enumerate(list(multi_diag_icd9.items())[:5]):
            print(f"      ICD9 {code}:")
            for diag in list(diags)[:3]:  # 每个编码最多显示3个诊断
                print(f"        - {diag}")
            if len(diags) > 3:
                print(f"        ... 还有 {len(diags) - 3} 个诊断")
    
    print(f"\n  - 一个ICD10编码对应多个诊断的情况: {len(multi_diag_icd10)} 种")
    if multi_diag_icd10:
        print("    示例 (显示前5个):")
        for i, (code, diags) in enumerate(list(multi_diag_icd10.items())[:5]):
            print(f"      ICD10 {code}:")
            for diag in list(diags)[:3]:  # 每个编码最多显示3个诊断
                print(f"        - {diag}")
            if len(diags) > 3:
                print(f"        ... 还有 {len(diags) - 3} 个诊断")
    
    # 统计编码合并情况
    total_unique_diagnoses = len(diagnosis_counts)
    total_unique_icd9 = len(icd9_counts)
    total_unique_icd10 = len(icd10_counts) if icd10_codes else 0
    
    print(f"\n  编码合并统计:")
    print(f"  - 总诊断种类: {total_unique_diagnoses}")
    print(f"  - 总ICD9编码种类: {total_unique_icd9}")
    print(f"  - 总ICD10编码种类: {total_unique_icd10}")
    print(f"  - 诊断比ICD9多: {total_unique_diagnoses - total_unique_icd9} 种")
    print(f"  - 诊断比ICD10多: {total_unique_diagnoses - total_unique_icd10} 种")
    print(f"  - 平均每个ICD9编码对应诊断数: {total_unique_diagnoses / total_unique_icd9:.2f}")
    if total_unique_icd10 > 0:
        print(f"  - 平均每个ICD10编码对应诊断数: {total_unique_diagnoses / total_unique_icd10:.2f}")
    
    # 保存映射关系到文件
    mapping_data = {
        'diagnosis_to_icd9_mapping': {diag: list(codes) for diag, codes in diagnosis_to_icd9.items()},
        'diagnosis_to_icd10_mapping': {diag: list(codes) for diag, codes in diagnosis_to_icd10.items()},
        'icd9_to_diagnosis_mapping': {code: list(diags) for code, diags in icd9_to_diagnoses.items()},
        'icd10_to_diagnosis_mapping': {code: list(diags) for code, diags in icd10_to_diagnoses.items()},
        'multi_icd_diagnoses': {
            'icd9': {diag: list(codes) for diag, codes in multi_icd9_diagnoses.items()},
            'icd10': {diag: list(codes) for diag, codes in multi_icd10_diagnoses.items()}
        },
        'multi_diagnosis_codes': {
            'icd9': {code: list(diags) for code, diags in multi_diag_icd9.items()},
            'icd10': {code: list(diags) for code, diags in multi_diag_icd10.items()}
        }
    }
    
    mapping_file = os.path.join(data_dir, 'diagnosis_icd_mapping.json')
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, indent=2, ensure_ascii=False)
    print(f"  - 诊断与ICD编码映射关系已保存到: {mapping_file}")
    
    # 3. 疾病系统分类分析（基于 diagnosisstring 的层级结构）
    print("\n3. 疾病系统分类分析:")
    
    # 提取疾病系统（第一级分类）
    systems = []
    for diagnosis in combined_df['diagnosisstring'].dropna():
        if '|' in str(diagnosis):
            system = str(diagnosis).split('|')[0].strip()
            systems.append(system)
        else:
            systems.append('未分类')
    
    system_counts = pd.Series(systems).value_counts()
    print(f"  - 总共有 {len(system_counts)} 个疾病系统")
    print(f"  - 疾病系统分布:")
    for i, (system, count) in enumerate(system_counts.items(), 1):
        percentage = (count / len(systems)) * 100
        print(f"    {i:2d}. {system}: {count} 次 ({percentage:.2f}%)")
    
    # 保存疾病系统分布
    system_stats_file = os.path.join(data_dir, 'disease_system_distribution.csv')
    system_counts.to_csv(system_stats_file, header=['count'])
    print(f"  - 疾病系统分布已保存到: {system_stats_file}")
    
    # 4. 患者疾病数量分析
    print("\n4. 患者疾病数量分析:")
    patient_disease_counts = combined_df.groupby('patient_id').size()
    
    print(f"  - 每个患者的疾病数量统计:")
    print(f"    最少疾病数: {patient_disease_counts.min()}")
    print(f"    最多疾病数: {patient_disease_counts.max()}")
    print(f"    平均疾病数: {patient_disease_counts.mean():.2f}")
    print(f"    中位数疾病数: {patient_disease_counts.median():.2f}")
    
    # 生成汇总报告
    summary_report = {
        'total_diagnosis_records': len(combined_df),
        'total_patients': combined_df['patient_id'].nunique(),
        'unique_diagnosis_strings': len(diagnosis_counts),
        'unique_icd9_codes': len(icd9_counts),
        'unique_icd10_codes': len(icd10_counts) if icd10_codes else 0,
        'disease_systems': len(system_counts),
        'top_10_diagnoses': diagnosis_counts.head(10).to_dict(),
        'top_10_icd9': icd9_counts.head(10).to_dict(),
        'top_10_icd10': icd10_counts.head(10).to_dict() if icd10_codes else {},
        'disease_system_distribution': system_counts.to_dict(),
        'patient_disease_stats': {
            'min': int(patient_disease_counts.min()),
            'max': int(patient_disease_counts.max()),
            'mean': float(patient_disease_counts.mean()),
            'median': float(patient_disease_counts.median())
        }
    }
    
    # 保存汇总报告
    summary_file = os.path.join(data_dir, 'disease_analysis_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)
    
    print(f"\nDisease analysis summary report saved to: {summary_file}")
    print("\n疾病分布分析完成!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        # 如果命令行参数是 analyze，只运行分析
        analyze_disease_distribution()
    elif len(sys.argv) > 1 and sys.argv[1] == "both":
        # 如果命令行参数是 both，先处理再分析
        main()
        print("\n" + "="*60)
        analyze_disease_distribution()
    else:
        # 默认只运行处理
        main()
