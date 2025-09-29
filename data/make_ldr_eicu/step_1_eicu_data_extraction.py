#!/usr/bin/env python3
"""
eICU Data Extraction Script
1. Extract top 100 diseases
2. Create patient-disease dictionary (only containing top 100 diseases)
3. Sample 20 cases for each disease
"""

import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
import random

def load_data(data_path):
    """Load patient and diagnosis data"""
    print("Loading data...")
    
    # Load diagnosis data
    diagnosis = pd.read_csv(os.path.join(data_path, 'diagnosis.csv.gz'))
    print(f"Diagnosis data: {len(diagnosis)} records")
    
    # Load patient data - patient.csv.gz is in subdirectory
    patient = pd.read_csv(os.path.join(data_path, 'eicu-crd', 'patient.csv.gz'))
    print(f"Patient data: {len(patient)} records")
    
    return diagnosis, patient

def extract_top_diseases(diagnosis, top_n=100):
    """Extract top N diseases with ICD codes"""
    print(f"\nExtracting top {top_n} diseases...")
    
    # Filter diagnoses with ICD codes
    diagnosis_icd = diagnosis[diagnosis['icd9code'].notna()]
    print(f"Diagnoses with ICD codes: {len(diagnosis_icd)} records")
    print(f"Unique diseases: {diagnosis_icd['diagnosisstring'].nunique()}")
    
    # Count disease frequency
    disease_freq = diagnosis_icd['diagnosisstring'].value_counts()
    top_diseases = disease_freq.head(top_n)
    
    print(f"\nTop {top_n} disease statistics:")
    for i, (disease, count) in enumerate(top_diseases.head(10).items(), 1):
        print(f"{i:2d}. {disease[:80]}... : {count} times")
    
    # Save top disease list
    top_diseases_dict = {
        'top_diseases': {str(k): int(v) for k, v in top_diseases.to_dict().items()},
        'total_cases': int(top_diseases.sum()),
        'extraction_date': pd.Timestamp.now().isoformat()
    }
    
    return top_diseases.index.tolist(), top_diseases_dict, diagnosis_icd

def create_patient_disease_dict(diagnosis_icd, patient, top_diseases_list):
    """Create patient-disease dictionary with strict filtering:
    1. Only keep patients whose diseases are all in the top disease list
    2. Only keep patients with both primary and major diagnoses
    """
    print(f"\nCreating patient-disease dictionary (strict filtering mode)...")
    
    # Get all diseases for each patient
    patient_all_diseases = diagnosis_icd.groupby('patientunitstayid')['diagnosisstring'].apply(list).to_dict()
    print(f"Total patients with diagnosis records: {len(patient_all_diseases)}")
    
    # Analyze disease distribution per patient
    disease_counts_per_patient = [len(diseases) for diseases in patient_all_diseases.values()]
    print(f"Number of diseases per patient: mean {np.mean(disease_counts_per_patient):.1f}, median {np.median(disease_counts_per_patient):.1f}")
    print(f"Patient disease count range: {min(disease_counts_per_patient)} - {max(disease_counts_per_patient)}")
    
    # Find patients with only top diseases
    valid_patients = []
    patients_with_non_top_diseases = 0
    
    for patient_id, diseases in patient_all_diseases.items():
        # Check if patient has any disease not in top_diseases_list
        non_top_diseases = [d for d in diseases if d not in top_diseases_list]
        if len(non_top_diseases) > 0:
            patients_with_non_top_diseases += 1
        else:
            # All diseases of this patient are in the top disease list
            valid_patients.append(patient_id)
    
    print(f"Patients with only top diseases: {len(valid_patients)}")
    print(f"Patients with non-top diseases: {patients_with_non_top_diseases}")
    print(f"First filtering rate: {patients_with_non_top_diseases/len(patient_all_diseases)*100:.1f}%")
    
    # Filter diagnosis records for valid patients
    filtered_diagnosis = diagnosis_icd[
        diagnosis_icd['patientunitstayid'].isin(valid_patients)
    ]
    print(f"Filtered diagnosis records: {len(filtered_diagnosis)}")
    
    # Merge patient and diagnosis data
    patient_diagnosis = patient.merge(
        filtered_diagnosis,
        on='patientunitstayid',
        how='inner'
    )
    print(f"Records after merge: {len(patient_diagnosis)}")
    print(f"Unique patients: {patient_diagnosis['patientunitstayid'].nunique()}")
    # print(patient_diagnosis.head())
   
    # Process age data
    def clean_age(age_value):
        if pd.isna(age_value):
            return None
        if isinstance(age_value, str):
            import re
            numbers = re.findall(r'\d+', str(age_value))
            if numbers:
                return int(numbers[0])
            else:
                return None
        try:
            return int(float(age_value))
        except:
            return None
    
    patient_diagnosis['age_numeric'] = patient_diagnosis['age'].apply(clean_age)
    
    # Calculate ICU length of stay
    patient_diagnosis['icu_los_days'] = patient_diagnosis['unitdischargeoffset'] / (60 * 24)
    
    # Create patient dictionary
    patient_dict = {}
    filtered_by_diagnosis_priority = 0
    patients_without_primary = 0
    patients_without_major = 0
    patients_without_both = 0
    total_candidates = patient_diagnosis['patientunitstayid'].nunique()
    
    for patient_id in patient_diagnosis['patientunitstayid'].unique():
        patient_records = patient_diagnosis[
            patient_diagnosis['patientunitstayid'] == patient_id
        ]
        
        # Get basic patient info
        first_record = patient_records.iloc[0]
        
        # Get all top diseases for this patient
        patient_diseases = patient_records['diagnosisstring'].unique().tolist()
        
        # By priority
        primary_diseases = patient_records[
            patient_records['diagnosispriority'] == 'Primary'
        ]['diagnosisstring'].unique().tolist()
        
        major_diseases = patient_records[
            patient_records['diagnosispriority'] == 'Major'
        ]['diagnosisstring'].unique().tolist()
        
        other_diseases = patient_records[
            patient_records['diagnosispriority'] == 'Other'
        ]['diagnosisstring'].unique().tolist()
        
        # Must have both primary and major diagnoses
        if len(primary_diseases) == 0 or len(major_diseases) == 0:
            filtered_by_diagnosis_priority += 1
            if len(primary_diseases) == 0:
                patients_without_primary += 1
            if len(major_diseases) == 0:
                patients_without_major += 1
            if len(primary_diseases) == 0 and len(major_diseases) == 0:
                patients_without_both += 1
            continue  # Skip patients without primary or major diagnoses
        
        # Build patient info
        patient_dict[str(patient_id)] = {
            'patient_id': int(patient_id),
            'age': int(first_record['age_numeric']) if pd.notna(first_record['age_numeric']) else None,
            'gender': str(first_record['gender']),
            'all_diseases': patient_diseases,
            'primary_diseases': primary_diseases,
            'major_diseases': major_diseases,
            'other_diseases': other_diseases,
            'total_disease_count': len(patient_diseases),
            'primary_count': len(primary_diseases),
            'major_count': len(major_diseases),
            'other_count': len(other_diseases)
        }
    
    print(f"Patient dictionary created: {len(patient_dict)} patients")
    print(f"\nDiagnosis priority filtering statistics:")
    print(f"- Total candidate patients: {total_candidates}")
    print(f"- Patients without primary diagnosis: {patients_without_primary}")
    print(f"- Patients without major diagnosis: {patients_without_major}")
    print(f"- Patients without both primary and major diagnosis: {patients_without_both}")
    print(f"- Patients filtered by diagnosis priority: {filtered_by_diagnosis_priority}")
    print(f"- Second filtering rate: {filtered_by_diagnosis_priority/total_candidates*100:.1f}%")
    print(f"- Final retained patients: {len(patient_dict)}")
    print(f"- Overall retention rate: {len(patient_dict)/len(patient_all_diseases)*100:.1f}%")
    
    return patient_dict

def sample_cases_by_disease(patient_dict, top_diseases_list, samples_per_disease=20):
    """Sample specified number of cases for each disease"""
    print(f"\nSampling {samples_per_disease} cases for each disease...")
    
    # Collect patients for each disease
    disease_patients = defaultdict(list)
    
    for patient_id, patient_info in patient_dict.items():
        for disease in patient_info['all_diseases']:
            if disease in top_diseases_list:
                disease_patients[disease].append(patient_id)
    
    # Sample cases
    sampled_cases = {}
    sampling_stats = {}
    
    for disease in top_diseases_list:
        available_patients = disease_patients[disease]
        
        if len(available_patients) == 0:
            print(f"Warning: Disease '{disease[:50]}...' has no available patients")
            sampling_stats[disease] = {
                'available': 0,
                'sampled': 0
            }
            continue
        
        # Sampling
        sample_size = min(samples_per_disease, len(available_patients))
        sampled_patient_ids = random.sample(available_patients, sample_size)
        
        # Collect sampled patient info
        sampled_patients = []
        for patient_id in sampled_patient_ids:
            patient_info = patient_dict[patient_id].copy()
            # Mark the matched disease
            patient_info['target_disease'] = disease
            sampled_patients.append(patient_info)
        
        sampled_cases[disease] = sampled_patients
        sampling_stats[disease] = {
            'available': int(len(available_patients)),
            'sampled': int(sample_size)
        }
    
    # Print sampling statistics
    print(f"\nSampling statistics:")
    total_sampled = 0
    diseases_with_full_samples = 0
    
    for disease, stats in list(sampling_stats.items())[:10]:  # Show first 10
        available = stats['available']
        sampled = stats['sampled']
        total_sampled += sampled
        
        if sampled == samples_per_disease:
            diseases_with_full_samples += 1
        
        print(f"{disease[:60]}...: {sampled}/{available} cases")
    
    print(f"\nTotal:")
    print(f"- Number of diseases: {len(sampling_stats)}")
    print(f"- Diseases with full samples: {diseases_with_full_samples}")
    print(f"- Total sampled cases: {total_sampled}")
    
    return sampled_cases, sampling_stats

def save_results(top_diseases_dict, patient_dict, sampled_cases, sampling_stats, output_dir):
    """Save all results"""
    print(f"\nSaving results to {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save top 100 diseases
    top_diseases_file = os.path.join(output_dir, 'top100_diseases.json')
    with open(top_diseases_file, 'w', encoding='utf-8') as f:
        json.dump(top_diseases_dict, f, ensure_ascii=False, indent=2)
    print(f"Saved top diseases: {top_diseases_file}")
    
    # 2. Save full patient-disease dictionary
    patient_dict_file = os.path.join(output_dir, 'patient_disease_dict.json')
    with open(patient_dict_file, 'w', encoding='utf-8') as f:
        json.dump(patient_dict, f, ensure_ascii=False, indent=2)
    print(f"Saved patient dictionary: {patient_dict_file}")
    
    # 3. Save sampled cases
    sampled_cases_file = os.path.join(output_dir, 'eicu_sample_cases.json')
    with open(sampled_cases_file, 'w', encoding='utf-8') as f:
        json.dump(sampled_cases, f, ensure_ascii=False, indent=2)
    print(f"Saved sampled cases: {sampled_cases_file}")
    
    # 4. Save sampling statistics
    sampling_stats_file = os.path.join(output_dir, 'sampling_statistics.json')
    with open(sampling_stats_file, 'w', encoding='utf-8') as f:
        json.dump(sampling_stats, f, ensure_ascii=False, indent=2)
    print(f"Saved sampling statistics: {sampling_stats_file}")
    
    print("All files saved!")

def analyze_top_diseases_coverage(data_path='eicu-crd-part', top_diseases_file='eicu_extracted_data/top100_diseases.json'):
    """
    Analyze how many cases are covered by the top 100 diseases
    As long as a patient is diagnosed with this disease, the case is considered covered
    """
    print("=== Analyzing Top 100 Disease Coverage ===")
    
    # 1. Load top 100 disease list
    print(f"Loading top disease list: {top_diseases_file}")
    with open(top_diseases_file, 'r', encoding='utf-8') as f:
        top_diseases_data = json.load(f)
    
    top_diseases_list = list(top_diseases_data['top_diseases'].keys())
    print(f"Number of top diseases: {len(top_diseases_list)}")
    
    # 2. Load diagnosis data
    print("\nLoading diagnosis data...")
    diagnosis = pd.read_csv(os.path.join(data_path, 'diagnosis.csv.gz'))
    print(f"Total diagnosis records: {len(diagnosis)}")
    print(f"Total patients: {diagnosis['patientunitstayid'].nunique()}")
    
    # 3. Filter diagnoses with ICD codes (consistent with original logic)
    diagnosis_icd = diagnosis[diagnosis['icd9code'].notna()]
    print(f"Diagnosis records with ICD codes: {len(diagnosis_icd)}")
    print(f"Patients with ICD codes: {diagnosis_icd['patientunitstayid'].nunique()}")
    
    # 4. Count diseases for each patient
    print("\nAnalyzing patient disease coverage...")
    patient_diseases = diagnosis_icd.groupby('patientunitstayid')['diagnosisstring'].apply(set).to_dict()
    print(f"Total patients with disease records: {len(patient_diseases)}")
    
    # 5. Count patients covered by top diseases
    covered_patients = set()
    disease_coverage_stats = {}
    
    for disease in top_diseases_list:
        # Find patients with this disease
        patients_with_disease = [
            patient_id for patient_id, diseases in patient_diseases.items() 
            if disease in diseases
        ]
        
        disease_coverage_stats[disease] = len(patients_with_disease)
        covered_patients.update(patients_with_disease)
    
    # 6. Calculate coverage statistics
    total_patients_with_icd = len(patient_diseases)
    covered_patients_count = len(covered_patients)
    coverage_rate = covered_patients_count / total_patients_with_icd * 100
    
    print(f"\n=== Coverage Statistics ===")
    print(f"Total patients with ICD codes: {total_patients_with_icd:,}")
    print(f"Patients covered by Top 100 diseases: {covered_patients_count:,}")
    print(f"Coverage rate: {coverage_rate:.2f}%")
    print(f"Uncovered patients: {total_patients_with_icd - covered_patients_count:,}")
    
    # 7. Show coverage for each disease (top 20)
    print(f"\n=== Top 20 Diseases Patient Coverage ===")
    sorted_diseases = sorted(disease_coverage_stats.items(), key=lambda x: x[1], reverse=True)
    
    for i, (disease, patient_count) in enumerate(sorted_diseases[:20], 1):
        disease_short = disease[:70] + "..." if len(disease) > 70 else disease
        print(f"{i:2d}. {disease_short}: {patient_count:,} patients")
    
    # 8. Analyze diseases of uncovered patients
    print(f"\n=== Analyzing Disease Distribution of Uncovered Patients ===")
    uncovered_patients = set(patient_diseases.keys()) - covered_patients
    print(f"Number of uncovered patients: {len(uncovered_patients)}")
    
    if len(uncovered_patients) > 0:
        # Count diseases of uncovered patients
        uncovered_diseases = []
        for patient_id in uncovered_patients:
            uncovered_diseases.extend(list(patient_diseases[patient_id]))
        
        uncovered_disease_counts = pd.Series(uncovered_diseases).value_counts()
        print(f"Number of unique diseases in uncovered patients: {len(uncovered_disease_counts)}")
        print(f"\nMost common diseases in uncovered patients (top 10):")
        
        for i, (disease, count) in enumerate(uncovered_disease_counts.head(10).items(), 1):
            disease_short = disease[:70] + "..." if len(disease) > 70 else disease
            print(f"{i:2d}. {disease_short}: {count} times")
    
    # 9. Save detailed statistics
    coverage_stats = {
        'summary': {
            'total_patients_with_icd': int(total_patients_with_icd),
            'covered_patients': int(covered_patients_count),
            'coverage_rate_percent': round(coverage_rate, 2),
            'uncovered_patients': int(total_patients_with_icd - covered_patients_count),
            'analysis_date': pd.Timestamp.now().isoformat()
        },
        'disease_patient_counts': {k: int(v) for k, v in disease_coverage_stats.items()},
        'top_uncovered_diseases': {k: int(v) for k, v in uncovered_disease_counts.head(50).items()} if len(uncovered_patients) > 0 else {}
    }
    
    coverage_file = 'eicu_extracted_data/top100_diseases_coverage_analysis.json'
    os.makedirs(os.path.dirname(coverage_file), exist_ok=True)
    with open(coverage_file, 'w', encoding='utf-8') as f:
        json.dump(coverage_stats, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed statistics saved to: {coverage_file}")
    
    return coverage_stats

def main():
    """Main function"""
    print("=== eICU Data Extraction Started ===")
    
    # Config parameters
    data_path = 'eicu-crd-part'
    output_dir = 'eicu_extracted_data'
    top_n = 100
    samples_per_disease = 20
    
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    
    try:
        # 1. Load data
        diagnosis, patient = load_data(data_path)
        
        # 2. Extract top diseases
        top_diseases_list, top_diseases_dict, diagnosis_icd = extract_top_diseases(
            diagnosis, top_n
        )
        
        # 3. Create patient-disease dictionary
        patient_dict = create_patient_disease_dict(
            diagnosis_icd, patient, top_diseases_list
        )
        
        # 4. Sample cases
        sampled_cases, sampling_stats = sample_cases_by_disease(
            patient_dict, top_diseases_list, samples_per_disease
        )
        
        # 5. Save results
        save_results(
            top_diseases_dict, patient_dict, sampled_cases, 
            sampling_stats, output_dir
        )
        
        # 6. Analyze top disease coverage
        analyze_top_diseases_coverage(data_path, 'eicu_extracted_data/top100_diseases.json')
        
        print("\n=== Data extraction completed! ===")
        print(f"Results saved in: {output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def test_coverage_analysis():
    """Test disease coverage analysis function separately"""
    print("=== Testing Top 100 Disease Coverage Analysis ===")
    
    try:
        coverage_stats = analyze_top_diseases_coverage()
        print("\nAnalysis completed!")
        return coverage_stats
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run main program
    # main()
    
    # Or only run coverage analysis test
    test_coverage_analysis()
