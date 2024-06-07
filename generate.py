import pandas as pd
from pathlib import Path

# Function to create well description content
def create_well_content(row):
    content = (f"On {row['PIS_DATE']}, a well named {row['GRID_NAME']} was established, with string name {row['STRING_NAME']} "
               f"and alternative name {row['ALT_NAME']} with drilling type {row['DRILL_TYPE']}. The well {row['GRID_NAME']} was drilled on "
               f"{row['STATUS_START_DATE']} and is currently {row['STATUS']} with status type {row['STATUS_TYPE']}. ")
    return content

# Function to create well production description content
def create_production_content(grouped_data):
    production_content = ""
    for _, row in grouped_data.iterrows():
        production_content += (f"On {row['TEST_DATE']} at {row['TEST_TIME']}, this well produced "
                               f"{row['OIL_BOPD']} barrels per day with GAS MSCFPD {row['GAS_MSCFPD']} and water cut percentage {row['WATER_CUT_PERCENT']}. ")
    return production_content

# Load data from the Excel file
file_path = 'D:/Nazirman/LLAMA/data/PoC.xlsx'

# Read data from both sheets
prod_data = pd.read_excel(file_path, sheet_name='PROD_DATA')
last_data = pd.read_excel(file_path, sheet_name='LAST_DATA')

# Ensure consistent formatting
last_data.columns = last_data.columns.str.strip()
prod_data.columns = prod_data.columns.str.strip()

# Create content for each well
well_contents = {}
for _, row in last_data.iterrows():
    grid_name = row['GRID_NAME']
    string_name = row['STRING_NAME']
    alt_name = row['ALT_NAME']
    drill_type = row['DRILL_TYPE']
    pis_date = row['PIS_DATE']
    status_start_date = row['STATUS_START_DATE']
    status = row['STATUS']
    status_type = row['STATUS_TYPE']
    
    # Create well description content
    well_content = create_well_content(row)
    
    # Add production description from all matching rows in prod_data
    prod_rows = prod_data[prod_data['GRID_NAME'] == grid_name]
    production_content = create_production_content(prod_rows)
    
    # Combine well description content with production content
    full_content = well_content + production_content
    
    # Save to dictionary
    well_contents[grid_name] = full_content

# Print content for verification
for grid_name, content in well_contents.items():
    print(f"Content for well {grid_name}:\n{content}\n")

# Save the content to text files
output_dir = Path('./data/output')
output_dir.mkdir(parents=True, exist_ok=True)

for grid_name, content in well_contents.items():
    output_file_path = output_dir / f"{grid_name}.txt"
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(content)

print("Text files generated successfully.")

