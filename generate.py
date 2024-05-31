import pandas as pd
from pathlib import Path

# Fungsi untuk membuat konten deskripsi sumur
def create_well_content(row):
    content = (f"Pada tanggal {row['PIS_DATE']}, sumur bernama {row['GRID_NAME']} didirikan, dengan nama string {row['STRING_NAME']} "
               f"dan nama alternatif {row['ALT_NAME']} dengan tipe pengeboran {row['DRILL_TYPE']}. Sumur {row['GRID_NAME']} ini dibor pada "
               f"{row['STATUS_START_DATE']} dan saat ini berstatus {row['STATUS']} dengan tipe {row['STATUS_TYPE']}. ")
    return content

# Fungsi untuk membuat konten deskripsi produksi sumur
def create_production_content(grouped_data):
    production_content = ""
    for _, row in grouped_data.iterrows():
        production_content += (f"Pada tanggal {row['TEST_DATE']} pukul {row['TEST_TIME']}, sumur ini menghasilkan "
                               f"{row['OIL_BOPD']} barel per hari dengan GAS MSCFPD {row['GAS_MSCFPD']} dan persentase water cut {row['WATER_CUT_PERCENT']}. ")
    return production_content

# Load data from the excel file
file_path = 'E:/LLAMA/data/PoC.xlsx'

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
    
    # Buat konten deskripsi sumur
    well_content = create_well_content(row)
    
    # Tambahkan deskripsi produksi dari semua baris yang sesuai di prod_data
    prod_rows = prod_data[prod_data['GRID_NAME'] == grid_name]
    production_content = create_production_content(prod_rows)
    
    # Gabungkan konten deskripsi sumur dengan konten produksi
    full_content = well_content + production_content
    
    # Simpan ke dictionary
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
