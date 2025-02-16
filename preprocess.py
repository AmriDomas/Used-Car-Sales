import pandas as pd

def preprocess_data(data):
    """
    Fungsi ini melakukan preprocessing pada data input sesuai dengan fitur yang digunakan dalam model.
    """

    # Konversi kolom tanggal
    data['Purchased Date'] = pd.to_datetime(data['Purchased Date'])
    data['Sold Date'] = pd.to_datetime(data['Sold Date'])

    # Ekstraksi fitur tanggal
    data['Purchased Year'] = data['Purchased Date'].dt.year
    data['Purchased Month'] = data['Purchased Date'].dt.month
    data['Purchased Day'] = data['Purchased Date'].dt.day

    data['Sold Year'] = data['Sold Date'].dt.year
    data['Sold Month'] = data['Sold Date'].dt.month
    data['Sold Day'] = data['Sold Date'].dt.day
    
    data['Price per HP'] = data['Price-$'] / data['Engine Power-HP']
    data['Price per KM'] = data['Price-$'] / data['Mileage-KM']
    data['Car Age at Purchase'] = data['Purchased Year'] - data['Manufactured Year']
    data['Car Age at Sale'] = data['Sold Year'] - data['Manufactured Year']
    data['Sales Commission per KM'] = data['Sales Commission-$'] / data['Mileage-KM']
    data['Sales Commission per HP'] = data['Sales Commission-$'] / data['Engine Power-HP']
    data['Sales Commission per Price'] = data['Sales Commission-$'] / data['Price-$']
    data['Margin per KM'] = data['Margin-%'] / data['Mileage-KM']
    data['Margin per HP'] = data['Margin-%'] / data['Engine Power-HP']
    data['Margin per Price'] = data['Margin-%'] / data['Price-$']
    data['Price per KM per HP'] = data['Price-$'] / data['Mileage-KM'] / data['Engine Power-HP']
    data['Price per KM per Price'] = data['Price-$'] / data['Mileage-KM'] / data['Price-$']
    data['Price per HP per Price'] = data['Price-$'] / data['Engine Power-HP'] / data['Price-$']
    data['Price per KM per Price'] = data['Price-$'] / data['Mileage-KM'] / data['Price-$']
    data['Old Car Purchase'] = (data['Car Age at Purchase'] > 5).astype(int)
    data['Old Car Sale'] = (data['Car Age at Sale'] > 5).astype(int)
    data['Avg Sales Rating'] = data.groupby('Sales Agent Name')['Sales Rating'].transform('mean')
    data['Total Sales by Agent'] = data.groupby('Sales Agent Name')['ID'].transform('count')
    data['Avg Sales Commission by Agent'] = data.groupby('Sales Agent Name')['Sales Commission-$'].transform('mean')
    data['Avg Sales Commission by Distributor'] = data.groupby('Distributor Name')['Sales Commission-$'].transform('mean')
    data['Avg Sales Commission by Manufacturer'] = data.groupby('Manufacturer Name')['Sales Commission-$'].transform('mean')
    data['Avg Sales Commission by Car Type'] = data.groupby('Car Type')['Sales Commission-$'].transform('mean')
    data['Avg Sales Commission by Color'] = data.groupby('Color')['Sales Commission-$'].transform('mean')
    data['Avg Sales Commission by Gearbox'] = data.groupby('Gearbox')['Sales Commission-$'].transform('mean')

    # Hapus kolom non-numerik (kecuali kolom target jika ada)
    cols_to_remove = data.select_dtypes(include=['object', 'datetime64[ns]']).columns
    data_clean = data.drop(columns=cols_to_remove, errors='ignore')

    return data_clean
