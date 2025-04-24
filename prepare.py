# %% external dataset analysis

import os
import pandas as pd

os_sheet_names = [
    'Alex Final',
    'Lennart Final',
    'Konstantin Final',
    'Felix',
]

op_modes = [
    'native',
    'low DFO 50%',
    'mcw DFO 50%',
    'mcw HTO 50%',
]

right = [
    'Rechts', 'R', 'RE', 'right', 'REchts'
]

left = [
    'Links', 'L', 'LI', 'left', 'LInks'
]


rel_angles = [
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

os_pat_col = ['Pat. Nummer']


# Load the Excel file
excel_file_path = './res.xlsx'
excel_data = pd.ExcelFile(excel_file_path)




# %%
os_data = {}
unique_ids = set()

for sheet_name in os_sheet_names:
    os_data[sheet_name] = {}
    df_raw = pd.read_excel(excel_file_path, sheet_name=sheet_name, header=0, skiprows=0)
    
    for dataset_mode in ['ex', 'int']:
        os_data[sheet_name][dataset_mode] = {}
        for opmode in op_modes:
            os_data[sheet_name][dataset_mode][opmode] = {}
            for ang in rel_angles:
                os_data[sheet_name][dataset_mode][opmode][ang] = {}
    
    for ind in range(len(df_raw)):
        pat_num = df_raw.iloc[ind]['Pat. Nummer']
        
        if pd.isna(pat_num):
            continue
        
        if type(pat_num) == str:
            continue
        
        print(ind)
        side = df_raw.iloc[ind]['Seite ']
        if side in right:
            side = 'R'
        elif side in left:
            side = 'L'
        else:
            print(f"Unknown side {side} for patient {pat_num}. Skipping.")
            continue
        
        unique_id = f"{pat_num}_{side}"
        unique_ids.add(unique_id)
        
        dataset_mode = 'ex'
        if pat_num < 1000:
            dataset_mode = 'int'
        
        # iterate over each field name in the current row
        for field_name in df_raw.columns:
            
            # check if the field_name string is contained in any of the strings in the list rel_angles
            if any(rel_angle in field_name for rel_angle in rel_angles):
                # give me the respective rel_angle name
                rel_angle = [rel_angle for rel_angle in rel_angles if rel_angle in field_name][0]
                cur_op_mode = op_modes[1]
                
                if 'präoOP' in field_name:
                    cur_op_mode = op_modes[0]
                
                elif field_name.endswith('.1'):
                    # mode is mcw DFO 50%
                    cur_op_mode = op_modes[2]
                
                elif field_name.endswith('.2'):
                    # mode is mcw HTO 50%
                    cur_op_mode = op_modes[3]
                
                if pd.isna(df_raw.iloc[ind][field_name]):
                    continue

                os_data[sheet_name][dataset_mode][cur_op_mode][rel_angle][unique_id] = df_raw.iloc[ind][field_name]
        
        
#strucutre:
# os -> dataset -> opmode -> amgle -> patient_id -> value

# %%

rel_angles_map = {
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


# load all the csv files for internal and external and store them in a dict
ai_data = {}

for mode in ['int', 'ex']:
    ai_data[mode] = {}
    for opmode in op_modes:
        ai_data[mode][opmode] = {}
        
        for ang in rel_angles:
            ai_data[mode][opmode][ang] = {}
        
        # load file
        file_name = f'./ai_res/{mode} {opmode}.csv'
        if os.path.exists(file_name):
            pd_raw = pd.read_csv(file_name, sep=',', encoding='utf-8', header=1)
        else:
            print(f"File {file_name} does not exist. Skipping.")
        
        for ind in range(len(pd_raw)):
            pat_str = pd_raw.iloc[ind]['File Name']
            print(pat_str)
            pat_num = pat_str.split('.')[0]
            try:
                pat_side = 'L' if pat_str.split('_')[1] == 'left' else 'R'
                pat_id = f'{pat_num}_{pat_side}'
            except IndexError:
                for unique_id in unique_ids:
                    if unique_id.startswith(f"{pat_num}_"):
                        pat_id = unique_id
            
            if pat_id not in unique_ids:
                continue
            
            for field_name in pd_raw.columns:
                
                # check if the field_name string is contained in any of the strings in the list rel_angles
                if any(rel_angle in field_name for rel_angle in list(rel_angles_map.keys())):
                    rel_angle_key = [rel_angle for rel_angle in list(rel_angles_map.keys()) if rel_angle in field_name][0]
                    
                    ai_data[mode][opmode][rel_angles_map[rel_angle_key]][pat_id] = pd_raw.iloc[ind][field_name]
            
            
                
        
        
        
            
# %%
ai_data
# %%
