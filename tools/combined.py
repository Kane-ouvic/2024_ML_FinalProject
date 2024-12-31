import os
import numpy as np

def process_npy_files(folder1, folder2, output_folder, operation="union"):
    """
    對三個資料夾中的相同名稱的 .npy 檔案取聯集或交集，並保存結果到新的資料夾。

    Parameters:
        folder1 (str): 第一個資料夾的路徑
        folder2 (str): 第二個資料夾的路徑
        folder3 (str): 第三個資料夾的路徑
        output_folder (str): 新的資料夾的路徑
        operation (str): "union" 或 "intersection"，指定要執行聯集還是交集
    """
    # 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)

    # 獲取三個資料夾中的檔案名稱集合
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))
    # files3 = set(os.listdir(folder3))

    # 找到三個資料夾中共有的檔案名稱
    common_files = files1 & files2

    for file_name in common_files:
        if file_name.endswith(".npy"):
            # 加載三個資料夾中的相同檔案
            file1_path = os.path.join(folder1, file_name)
            file2_path = os.path.join(folder2, file_name)
            # file3_path = os.path.join(folder3, file_name)

            array1 = np.load(file1_path)
            array2 = np.load(file2_path)
            # array3 = np.load(file3_path)

            # 計算聯集或交集
            if operation == "union":
                result = np.logical_or.reduce([array1, array2]).astype(np.uint8)
            elif operation == "intersection":
                result = np.logical_and.reduce([array1, array2]).astype(np.uint8)
            else:
                raise ValueError("Invalid operation. Use 'union' or 'intersection'.")

            # 保存結果
            output_path = os.path.join(output_folder, file_name)
            np.save(output_path, result)
            print(f"Saved {operation} result for {file_name} to {output_path}")

folder1 = "/home/ouvic/ML/ML_Final/predict_npy/test_2d_x_result_Unet++Improved_1_npy"  # 替換為第一個資料夾的路徑
folder2 = "/home/ouvic/ML/ML_Final/predict_npy/test_2d_x_result_Unet++_2_npy"  # 替換為第二個資料夾的路徑
# folder3 = "/home/ouvic/ML/ML_Final/test_2d_z_result_1_npy"  # 替換為第三個資料夾的路徑
output_folder = "/home/ouvic/ML/ML_Final/predict_npy/test_2d_xy_union_result_Unet++_1_npy"  # 替換為輸出資料夾的路徑
operation = "union"  # 或 "intersection"

process_npy_files(folder1, folder2, output_folder, operation)
