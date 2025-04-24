# %%
"""
Medical Imaging Data Analysis Script
Compares orthopedic surgeon measurements with AI-generated measurements
"""

import os
import logging
import pandas as pd
from typing import Dict, Set, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
EXCEL_FILE_PATH = './res.xlsx'

# Surgeon sheet names
SURGEON_SHEET_NAMES = [
    'OS1', #'Alex Final',
    'OS2', #'Lennart Final',
    'OS3', #'Konstantin Final',
    'OS4', #'Felix',
]

# Operation modes
OP_MODES = [
    'native',
    'low DFO 50%',
    'mcw DFO 50%',
    'mcw HTO 50%',
]

# Side identifiers
RIGHT_IDENTIFIERS = [
    'Rechts', 'R', 'RE', 'right', 'REchts'
]

LEFT_IDENTIFIERS = [
    'Links', 'L', 'LI', 'left', 'LInks'
]

# Relevant angle measurements
RELEVANT_ANGLES = [
    'mFA',
    'mLPFA',
    'mLDFA',
    'mMPTA',
    'mLDTA',
    'Mikulicz',
    'MAD',
    'M auf TP',
    'AMA',
    'JLCA',
    'Winkel',
    'Umstellung'
]

RELEVANT_ANGLES_POSTOP = [
    'mFA',
    'mLPFA',
    'Mikulicz',
    'MAD',
    'AMA',
    'Winkel',
    'Umstellung'
]

RELEVANT_ANGLES_PREOP = [
    'mFA',
    'mLPFA',
    'mLDFA',
    'mMPTA',
    'mLDTA',
    'Mikulicz',
    'MAD',
    'M auf TP',
    'AMA',
    'JLCA',
]




# Mapping between AI and surgeon measurement names
REL_ANGLES_MAP = {
    'MFTA': 'mFA',
    'MLPFA': 'mLPFA',
    'MLDFA': 'mLDFA',
    'MMPTA': 'mMPTA',
    'MLDTA': 'mLDTA',
    'LEG LENGTH': 'Mikulicz',
    'MAD': 'MAD',
    'AMA': 'AMA',
    'JLCA': 'JLCA',
    'Operation Angle': 'Winkel',
    'Correction': 'Umstellung'
}

def normalize_side(side: str) -> Optional[str]:
    """
    Normalize the side indicator to either 'L' or 'R'.
    
    Args:
        side (str): The side indicator from the dataset
        
    Returns:
        str or None: 'L' for left, 'R' for right, None if side is unknown
    """
    if side in RIGHT_IDENTIFIERS:
        return 'R'
    elif side in LEFT_IDENTIFIERS:
        return 'L'
    else:
        return None

def identify_op_mode(field_name: str) -> str:
    """
    Identify operation mode from field name.
    
    Args:
        field_name (str): The column name from the dataset
        
    Returns:
        str: Operation mode
    """
    preop_list = [
        'präoOP', 'präOP',
    ]
    
    if any(preop_term in field_name for preop_term in preop_list):
        return OP_MODES[0]  # native
    elif field_name.endswith('.1'):
        return OP_MODES[2]  # mcw DFO 50%
    elif field_name.endswith('.2'):
        return OP_MODES[3]  # mcw HTO 50%
    else:
        return OP_MODES[1]  # low DFO 50% (default)

def extract_rel_angle(field_name: str) -> Optional[str]:
    """
    Extract relevant angle name from field name.
    
    Args:
        field_name (str): The column name from the dataset
        
    Returns:
        str or None: Relevant angle name if found, None otherwise
    """
    matching_angles = [angle for angle in RELEVANT_ANGLES if angle in field_name]
    if matching_angles:
        return matching_angles[0]
    return None

def init_nested_dict() -> Dict:
    """
    Initialize the nested dictionary structure for storing data.
    
    Returns:
        dict: Initialized nested dictionary
    """
    data = {}
    for sheet_name in SURGEON_SHEET_NAMES:
        data[sheet_name] = {}
        for dataset_mode in ['ex', 'int']:
            data[sheet_name][dataset_mode] = {}
            for opmode in OP_MODES:
                data[sheet_name][dataset_mode][opmode] = {}
                for ang in RELEVANT_ANGLES:
                    data[sheet_name][dataset_mode][opmode][ang] = {}
    return data

def load_surgeon_data() -> tuple[Dict, Set[str]]:
    """
    Load and process surgeon data from Excel sheets.
    
    Returns:
        tuple: (surgeon_data, unique_patient_ids)
    """
    os_data = init_nested_dict()
    unique_ids = set()
    
    try:
        excel_data = pd.ExcelFile(EXCEL_FILE_PATH)
    except Exception as e:
        logger.error(f"Error opening Excel file: {e}")
        return os_data, unique_ids
    
    for sheet_name in SURGEON_SHEET_NAMES:
        try:
            df_raw = pd.read_excel(EXCEL_FILE_PATH, sheet_name=sheet_name, header=0, skiprows=0)
            logger.info(f"Successfully loaded sheet: {sheet_name}")
        except Exception as e:
            logger.error(f"Error loading sheet {sheet_name}: {e}")
            continue
        
        for ind in range(len(df_raw)):
            try:
                # Get patient number and skip if invalid
                pat_num = df_raw.iloc[ind]['Pat. Nummer']
                if pd.isna(pat_num) or isinstance(pat_num, str):
                    continue
                
                # Process side information
                side_raw = df_raw.iloc[ind]['Seite ']
                side = normalize_side(side_raw)
                if side is None:
                    logger.warning(f"Unknown side '{side_raw}' for patient {pat_num}. Skipping.")
                    continue
                
                # Create unique ID and determine dataset mode
                unique_id = f"{pat_num}_{side}"
                if SURGEON_SHEET_NAMES != 'OS4':
                    unique_ids.add(unique_id)
                
                dataset_mode = 'int' if pat_num < 1000 else 'ex'
                
                # Process measurements
                for field_name in df_raw.columns:
                    rel_angle = extract_rel_angle(field_name)
                    if rel_angle:
                        op_mode = identify_op_mode(field_name)
                        value = df_raw.iloc[ind][field_name]
                        
                        if not pd.isna(value):
                            os_data[sheet_name][dataset_mode][op_mode][rel_angle][unique_id] = value
            
            except Exception as e:
                logger.error(f"Error processing row {ind} in sheet {sheet_name}: {e}")
    
    return os_data, unique_ids

def get_patient_id(pat_str: str, unique_ids: Set[str]) -> Optional[str]:
    """
    Determine patient ID from file name.
    
    Args:
        pat_str (str): The file name string
        unique_ids (Set[str]): Set of known unique IDs
        
    Returns:
        str or None: Patient ID if found, None otherwise
    """
    pat_num = pat_str.split('.')[0]
    
    try:
        # First try to extract side from filename
        pat_side = 'L' if '_left' in pat_str else ('R' if '_right' in pat_str else None)
        if pat_side:
            pat_id = f'{pat_num}_{pat_side}'
            return pat_id if pat_id in unique_ids else None
    except Exception:
        pass
    
    # If that fails, try to find matching ID by number
    matching_ids = [uid for uid in unique_ids if uid.startswith(f"{pat_num}_")]
    return matching_ids[0] if matching_ids else None

def load_ai_data(unique_ids: Set[str]) -> Dict:
    """
    Load and process AI data from CSV files.
    
    Args:
        unique_ids (Set[str]): Set of unique patient IDs
        
    Returns:
        dict: Processed AI data
    """
    ai_data = {}
    
    for mode in ['int', 'ex']:
        ai_data[mode] = {}
        for opmode in OP_MODES:
            ai_data[mode][opmode] = {}
            for ang in RELEVANT_ANGLES:
                ai_data[mode][opmode][ang] = {}
            
            file_name = f'./ai_res/{mode} {opmode}.csv'
            if not os.path.exists(file_name):
                logger.warning(f"File {file_name} does not exist. Skipping.")
                continue
            
            try:
                pd_raw = pd.read_csv(file_name, sep=',', encoding='utf-8', header=1)
                logger.info(f"Successfully loaded CSV: {file_name}")
            except Exception as e:
                logger.error(f"Error loading CSV {file_name}: {e}")
                continue
            
            for ind in range(len(pd_raw)):
                try:
                    # Get file name and identify patient
                    pat_str = pd_raw.iloc[ind]['File Name']
                    pat_id = get_patient_id(pat_str, unique_ids)
                    
                    if not pat_id:
                        continue
                    
                    # Process measurements
                    for field_name in pd_raw.columns:
                        # Find matching angle from the AI measurement names
                        matching_angles = [angle for angle in REL_ANGLES_MAP.keys() 
                                         if angle and angle in field_name]
                        
                        if matching_angles:
                            rel_angle_key = matching_angles[0]
                            surgeon_angle_name = REL_ANGLES_MAP[rel_angle_key]
                            value = pd_raw.iloc[ind][field_name]
                            
                            if field_name in ['Operation Angle (deg)', 'JLCA (deg)']:
                                value = abs(value)
                            
                            if not pd.isna(value):
                                ai_data[mode][opmode][surgeon_angle_name][pat_id] = value
                
                except Exception as e:
                    logger.error(f"Error processing row {ind} in file {file_name}: {e}")
    
    return ai_data

def main():
    """
    Main function to execute the data loading and processing.
    """
    logger.info("Starting data analysis")
    
    # Load surgeon data
    logger.info("Loading surgeon data from Excel")
    os_data, unique_ids = load_surgeon_data()
    logger.info(f"Found {len(unique_ids)} unique patient IDs")
    
    # Load AI data
    logger.info("Loading AI data from CSV files")
    ai_data = load_ai_data(unique_ids)
    
    # At this point, you have both datasets loaded for comparison
    logger.info("Data loading complete")
    
    # Return data for further analysis
    return os_data, ai_data, unique_ids


# %%
if __name__ == "__main__":
    os_data, ai_data, unique_ids = main()
    print("Data loaded successfully!")
    print(f"Number of unique patients: {len(unique_ids)}")
    # You can add additional analysis code here
# %%
