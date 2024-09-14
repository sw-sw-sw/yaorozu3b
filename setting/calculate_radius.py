'''
config.csvからdnaデーターを読み取ってradiusを計算するコード

'''
import csv
import math

def calculate_radius(size, shell_size, shell_point_size, horn_length):
    return max(size / 2, size * shell_size / 2 + shell_point_size, size * horn_length / 2) + 1

def update_config_with_radius():
    # 一時的なストレージ for 読み取ったデータ
    config_data = []
    radius_row_index = None
    species_count = 0

    # config.csvを読み込む
    with open('config.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            config_data.append(row)
            if row[0] == 'RADIUS':
                radius_row_index = i
            if row[0] == 'SIZE':
                species_count = len([cell for cell in row[1:] if cell]) - 1  # ヘッダーを除く非空のセルの数

    # 種ごとのRADIUSを計算
    species_radius = []
    for i in range(1, species_count + 1):
        try:
            species_size = float(config_data[[row[0] for row in config_data].index('SIZE')][i])
            species_shell_size = float(config_data[[row[0] for row in config_data].index('SHELL_SIZE')][i])
            species_shell_point_size = float(config_data[[row[0] for row in config_data].index('SHELL_POINT_SIZE')][i])
            species_horn_length = float(config_data[[row[0] for row in config_data].index('HORN_LENGTH')][i])
            
            radius = calculate_radius(species_size, species_shell_size, species_shell_point_size, species_horn_length)
            species_radius.append(str(radius))
            print(f'Species {i} RADIUS = {radius}')
        except (ValueError, IndexError):
            print(f'Error calculating radius for species {i}. Using default value.')
            species_radius.append('')

    # RADIUSの行を更新
    radius_row = ['RADIUS'] + species_radius + [''] * (len(config_data[0]) - len(species_radius) - 1) + ['Calculated radius of agents']
    if radius_row_index is not None:
        config_data[radius_row_index] = radius_row
    else:
        config_data.append(radius_row)

    # 更新されたデータをconfig.csvに書き込む
    with open('config.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(config_data)

    print("Species-specific RADIUS values have been calculated and written to config.csv")

# 関数を実行
update_config_with_radius()