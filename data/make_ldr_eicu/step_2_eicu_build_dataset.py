"""
This file builds a dataset for the cases extracted in eicu_extracted_data/eicu_sample_cases.json

1. Iterate over eicu_sample_cases.json using patient_id as the index (and folder name)
2. For each patient_id (i.e., patientunitstayid) and time interval, extract the corresponding case data and save it to the specified output directory. The following files are needed:
    a. patient.csv.gz: contains patient information
    b. diagnosis.csv.gz: contains diagnosis information
    c. lab.csv.gz: contains laboratory test information
    d. microLab.csv.gz: contains microbiology lab test information
    e. pastHistory.csv.gz: contains patient history information
    f. physicalExam.csv.gz: contains physical exam information
    g. allergy.csv.gz: contains allergy information
3. Save the extracted data to the specified output directory.
"""

import json
import pandas as pd
import os
import gzip
from pathlib import Path
import numpy as np
from typing import Dict, List, Any


class EICUDatasetBuilder:
    def __init__(self, base_data_path: str = "eicu-crd-part", output_dir: str = "eicu_patient_datasets"):
        """
        Initialize the dataset builder
        
        Args:
            base_data_path: Base path for eICU data files
            output_dir: Output directory
        """
        self.base_data_path = Path(base_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Define data file paths
        self.data_files = {
            'patient': self.base_data_path / "eicu-crd" / "patient.csv.gz",
            'diagnosis': self.base_data_path / "eicu-crd" / "diagnosis.csv.gz", 
            'lab': self.base_data_path / "lab.csv.gz",  # in upper directory
            'microLab': self.base_data_path / "eicu-crd" / "microLab.csv.gz",
            'pastHistory': self.base_data_path / "eicu-crd" / "pastHistory.csv.gz",
            'physicalExam': self.base_data_path / "eicu-crd" / "physicalExam.csv.gz",
            'allergy': self.base_data_path / "allergy.csv.gz",  # allergy info, in upper directory
            # 'note': self.base_data_path / "eicu-crd" / "note.csv.gz"
        }
        
        # Preloaded data storage
        self.loaded_data = {}
        self.data_loaded = False
        
        # Check all files exist
        self._check_data_files()
        
    def _check_data_files(self):
        """Check if all required data files exist"""
        missing_files = []
        for name, path in self.data_files.items():
            if not path.exists():
                missing_files.append(f"{name}: {path}")
        
        if missing_files:
            print("Warning: The following files do not exist:")
            for file in missing_files:
                print(f"  - {file}")
        else:
            print("All data files checked, all files present")
    
    def load_sample_cases(self, sample_cases_path: str = "eicu_extracted_data/eicu_sample_cases.json") -> Dict:
        """
        Load sample case data
        
        Args:
            sample_cases_path: Path to sample cases JSON file
            
        Returns:
            Dict: Sample case data
        """
        with open(sample_cases_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_all_data(self):
        """
        Preload all data files into memory to avoid repeated reads
        """
        if self.data_loaded:
            return
            
        print("Start preloading all data files...")
        
        for data_type, file_path in self.data_files.items():
            if not file_path.exists():
                print(f"Skipping missing file: {file_path}")
                continue
                
            try:
                print(f"Loading {data_type} data...")
                # Check file size
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  File size: {file_size_mb:.2f} MB")
                
                df = pd.read_csv(file_path, compression='gzip')
                
                # Check if patientunitstayid column exists
                if 'patientunitstayid' in df.columns:
                    # For efficiency, sort by patientunitstayid
                    df = df.sort_values('patientunitstayid')
                    self.loaded_data[data_type] = df
                    print(f"  Loaded {len(df)} records, {len(df.columns)} columns")
                    
                    # Show memory usage
                    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
                    print(f"  Memory usage: {memory_usage_mb:.2f} MB")
                else:
                    print(f"  Warning: No patientunitstayid column in {data_type} file, skipping")
                    
            except Exception as e:
                print(f"  Error loading {data_type} data: {str(e)}")
        
        self.data_loaded = True
        print(f"\nData preloading complete, loaded {len(self.loaded_data)} data files")
        
        # Calculate total memory usage
        total_memory = sum(df.memory_usage(deep=True).sum() for df in self.loaded_data.values()) / (1024 * 1024)
        print(f"Total memory usage: {total_memory:.2f} MB")
    
    def get_patient_list_from_data(self) -> set:
        """
        Get all available patient IDs from loaded data
        
        Returns:
            set: Set of all available patient IDs
        """
        if not self.data_loaded:
            self.load_all_data()
        
        all_patient_ids = set()
        for data_type, df in self.loaded_data.items():
            if 'patientunitstayid' in df.columns:
                patient_ids = set(df['patientunitstayid'].unique())
                all_patient_ids.update(patient_ids)
                print(f"{data_type}: {len(patient_ids)} unique patients")
        
        print(f"Found {len(all_patient_ids)} unique patient IDs in total")
        return all_patient_ids
    
    def extract_patient_data(self, patient_id: int) -> Dict[str, pd.DataFrame]:
        """
        Extract all relevant data for a given patient (no time filtering)
        
        Args:
            patient_id: Patient ID (patientunitstayid)
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of data types to DataFrames
        """
        # Ensure data is preloaded
        if not self.data_loaded:
            self.load_all_data()
        
        extracted_data = {}
        
        for data_type, df in self.loaded_data.items():
            try:
                # Filter by patientunitstayid to get all relevant data for this patient
                patient_data = df[df['patientunitstayid'] == patient_id].copy()
                
                extracted_data[data_type] = patient_data
                print(f"Extracted {data_type} data: {len(patient_data)} records")
                    
            except Exception as e:
                print(f"Error extracting {data_type} data: {str(e)}")
                
        return extracted_data
    
    def save_patient_dataset(self, patient_id: int, patient_data: Dict[str, pd.DataFrame], 
                           case_info: Dict[str, Any]):
        """
        Save dataset for a single patient
        
        Args:
            patient_id: Patient ID
            patient_data: Patient data dictionary
            case_info: Case information
        """
        # Create patient directory
        patient_dir = self.output_dir / str(patient_id)
        patient_dir.mkdir(exist_ok=True)
        
        # Save case info
        case_info_path = patient_dir / "case_info.json"
        with open(case_info_path, 'w', encoding='utf-8') as f:
            json.dump(case_info, f, ensure_ascii=False, indent=2)
        
        # Save each data type
        for data_type, df in patient_data.items():
            if not df.empty:
                output_path = patient_dir / f"{data_type}.csv"
                df.to_csv(output_path, index=False)
                print(f"Saved {data_type} data to: {output_path}")
            else:
                print(f"{data_type} data for patient {patient_id} is empty, skipping save")
        
        # Create data summary
        self._create_data_summary(patient_id, patient_data, case_info)
    
    def _create_data_summary(self, patient_id: int, patient_data: Dict[str, pd.DataFrame], 
                           case_info: Dict[str, Any]):
        """
        Create data summary file
        
        Args:
            patient_id: Patient ID
            patient_data: Patient data dictionary
            case_info: Case information
        """
        patient_dir = self.output_dir / str(patient_id)
        summary_path = patient_dir / "data_summary.txt"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Patient {patient_id} Data Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic info
            f.write("Basic Information:\n")
            f.write(f"  Age: {case_info.get('age', 'N/A')}\n")
            f.write(f"  Gender: {case_info.get('gender', 'N/A')}\n")
            f.write(f"  ICU LOS (days): {case_info.get('icu_los_days', 'N/A')}\n")
            f.write(f"  Target Disease: {case_info.get('target_disease', 'N/A')}\n")
            f.write(f"  Total Disease Count: {case_info.get('total_disease_count', 'N/A')}\n\n")
            
            # Time info
            f.write("Time Information:\n")
            f.write(f"  Hospital Admit Offset: {case_info.get('hospital_admit_offset', 'N/A')} min\n")
            f.write(f"  Unit Discharge Offset: {case_info.get('unit_discharge_offset', 'N/A')} min\n\n")
            
            # Data overview
            f.write("Data File Overview:\n")
            for data_type, df in patient_data.items():
                f.write(f"  {data_type}: {len(df)} records\n")
                if not df.empty:
                    f.write(f"    Columns: {len(df.columns)}\n")
                    f.write(f"    Column names: {', '.join(df.columns.tolist()[:10])}{'...' if len(df.columns) > 10 else ''}\n")
                f.write("\n")
            
            # Disease info
            if 'all_diseases' in case_info:
                f.write("All Diseases:\n")
                for disease in case_info['all_diseases']:
                    f.write(f"  - {disease}\n")
                f.write("\n")
    
    def build_dataset(self, sample_cases_path: str = "eicu_extracted_data/eicu_sample_cases.json", 
                     max_patients: int = None):
        """
        Build the full dataset
        
        Args:
            sample_cases_path: Path to sample cases JSON file
            max_patients: Maximum number of patients to process (for testing)
        """
        print("Start building eICU dataset...")
        
        # Preload all data
        self.load_all_data()
        
        # Load sample cases
        sample_cases = self.load_sample_cases(sample_cases_path)
        print(f"Loaded sample case data, {len(sample_cases)} disease types in total")
        
        processed_count = 0
        total_patients = sum(len(cases) for cases in sample_cases.values())
        
        if max_patients:
            print(f"Limit to process at most {max_patients} patients")
        
        # Iterate over all disease types and patients
        for disease_type, patients in sample_cases.items():
            print(f"\nProcessing disease type: {disease_type}")
            
            for patient_case in patients:
                if max_patients and processed_count >= max_patients:
                    break
                    
                patient_id = patient_case['patient_id']
                print(f"\nProcessing patient {patient_id} ({processed_count + 1}/{total_patients if not max_patients else max_patients})")
                
                # Extract patient data (all relevant data, no time filtering)
                patient_data = self.extract_patient_data(patient_id)
                
                # Save dataset
                if patient_data:
                    self.save_patient_dataset(patient_id, patient_data, patient_case)
                    processed_count += 1
                else:
                    print(f"No data extracted for patient {patient_id}")
                    
            if max_patients and processed_count >= max_patients:
                break
        
        print(f"\nDataset build complete!")
        print(f"Processed {processed_count} patients in total")
        print(f"Data saved in: {self.output_dir}")
        
        # Create overall statistics
        self._create_overall_statistics(processed_count)
    
    def _create_overall_statistics(self, processed_count: int):
        """Create overall statistics"""
        stats_path = self.output_dir / "dataset_statistics.txt"
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("eICU Dataset Build Statistics\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Total patients processed: {processed_count}\n")
            f.write(f"Output directory: {self.output_dir}\n")
            f.write(f"Data file types: {', '.join(self.data_files.keys())}\n")
            
            # Statistics for each patient directory size
            total_size = 0
            for patient_dir in self.output_dir.iterdir():
                if patient_dir.is_dir() and patient_dir.name.isdigit():
                    dir_size = sum(f.stat().st_size for f in patient_dir.glob('**/*') if f.is_file())
                    total_size += dir_size
            
            f.write(f"Total data size: {total_size / (1024*1024):.2f} MB\n")


def main():
    """Main function"""
    # Create dataset builder
    builder = EICUDatasetBuilder()
    
    # Optional: check available patient IDs
    print("Checking available patient IDs in data...")
    available_patients = builder.get_patient_list_from_data()
    
    # Load sample cases and check intersection
    sample_cases = builder.load_sample_cases()
    sample_patient_ids = set()
    for disease_type, patients in sample_cases.items():
        for patient_case in patients:
            sample_patient_ids.add(patient_case['patient_id'])
    
    print(f"Number of patient IDs in sample cases: {len(sample_patient_ids)}")
    intersection = available_patients.intersection(sample_patient_ids)
    print(f"Number of sample patients found in data: {len(intersection)}")
    
    if len(intersection) == 0:
        print("Warning: No sample patient data found. Please check data files and sample case file consistency.")
        return
    
    # Build dataset (limit number of patients for testing)
    # builder.build_dataset(max_patients=5)  # Test with 5 patients first
    
    # For full build, uncomment below and comment above
    builder.build_dataset()

if __name__ == "__main__":
    main()

