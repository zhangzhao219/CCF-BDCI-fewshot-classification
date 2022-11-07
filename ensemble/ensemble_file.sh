# 模型文件集成

python3 ensemble_file.py --datetime $(date +%Y_%m_%d_%H_%M_%S) --file_folder ../goodmodels --mode 0 \
--data_list \
"[
    '2022_11_03_19_41_25-a-0.63234589689/2022_11_03_19_41_25-1-0.61577661518.csv', 
    '2022_11_03_19_41_25-a-0.63234589689/2022_11_03_19_41_25-2-.csv', 
    '2022_11_03_19_41_25-a-0.63234589689/2022_11_03_19_41_25-4-.csv',
    '2022_11_01_04_26_32-3-0.63293263685/2022_11_01_04_26_32-3-0.63293263685.csv',
    '2022_10_27_07_38_29_3-0.63077875449/2022_10_27_07_38_29_3-0.63077875449.csv',
    '2022_10_22_19_12_04-3-0.62876738448/2022_10_22_19_12_04-3-0.62876738448.csv',
    '2022_11_06_04_35_15-a-0.62673116125/result_score_2022_11_06_04_35_15_1.csv',
    '2022_11_06_04_35_15-a-0.62673116125/result_score_2022_11_06_04_35_15_2.csv',
    '2022_11_06_19_09_34-0-/result_score_2022_11_06_19_09_34_1.csv',
]" \
--data_weight "[1,1,1,1, 1, 1,1,1,1]" --modify --score