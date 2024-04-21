# Weekly Report 2023-10

## 2023.10.22

### 【本周工作总结】
1. 调研时间序列 tokenizer 方式适配大语言模型的方法。

Date|Method|Conference| Paper Title and Paper Interpretation (In Chinese) |Code
-----|----|-----|-----|-----
| 23-02-23 | [FPT](https://arxiv.org/abs/2302.11939) 🌟 | NIPS 2023 | One Fits All:Power General Time Series Analysis by Pretrained LM | [One-Fits-All](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All)   |
| 23-05-17 | [LLMTime](https://arxiv.org/abs/2310.07820) | NIPS 2023 | Large Language Models Are Zero-Shot Time Series Forecasters | [LLMTime](https://github.com/ngruver/llmtime) |
| 23-08-16 | [TEST](https://arxiv.org/abs/2308.08241) | Arxiv 2023 | TEST: Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series | None |
| 23-08-16 | [LLM4TS](https://arxiv.org/abs/2308.08469) | Arxiv 2023 | LLM4TS: Two-Stage Fine-Tuning for Time-Series Forecasting with Pre-Trained LLMs | None |
| 23-10-03 | [Time-LLM](https://arxiv.org/abs/2310.01728) | Arxiv 2023 | Time-LLM: Time Series Forecasting by Reprogramming Large Language Models | None |
| 23-10-08 | [TEMPO](https://arxiv.org/abs/2310.04948) | Arxiv 2023 | TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting | None |
| 23-10-12 | [Lag-Llama](https://arxiv.org/abs/2310.08278) | Arxiv 2023 | Lag-Llama: Towards Foundation Models for Time Series Forecasting | [Lag-Llama](https://github.com/kashif/pytorch-transformer-ts) |
| 23-10-15 | [UniTime](https://arxiv.org/abs/2310.09751) | Arxiv 2023 | UniTime: A Language-Empowered Unified Model for Cross-Domain Time Series Forecasting | None |

2. 复现两篇 time series forcasting 论文，使用计算机的 ETT 数据集，一篇是基于 LLM 的，一篇是基于卷积的 SOTA 方法，得到了和论文汇报差不多的指标结果。

| Method | Conference | Paper Title |
| ---- | ---- | ---- |
| FPT | NIPS 2023 | [One Fits All:Power General Time Series Analysis by Pretrained LM](https://arxiv.org/abs/2302.11939) | 
| TimesNet | ICLR 2023 | [TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis](https://arxiv.org/abs/2210.02186) |

### 【下周工作计划】
1. 抽取手上 120w+ 条 Kepler 光变曲线的时间序列数据，清洗整理汇集，便于后续模型训练，避免超大规模的零散小文件加载拖慢训练时间。
2. 调整 Kepler 曲线数据集适配计算机的时间序列预测任务，LLM 模型是否会取得不错的效果。如果可以，更大的 LLM 和自监督的方法是否可以取得更好的效果。


## 2023.10.23 - 2023.10.29

### 【本周工作总结】
1. 抽取手上 120w+ 条 Kepler 光变曲线的时间序列数据，清洗整理汇集。这样的数据量在整个计算机界都是稀缺的，先用一部分实验，未来考虑将整个kepler做成便于加载的数据集，测试多种方法，弄一个benchmarking。

### 【下周工作计划】
1. GPT(6)模型在 Kepler 数据集上测试，预训练和随机初始化对比效果，目的：LLM 冻结权重要比随机初始化表现好，这真的是因为模型里有知识吗，还是说只是现有数据集不够大，随机初始化的模型没有充分训练，所以表现不好。
2. 调研 irregular time series analysis 的论文，这是之前漏掉的一个研究方向，比较小众，对于天文观测采样不均匀的特点来说，可能会有作用。


## Code Recording

```python
"""
This script processes astronomical light curve data stored in FITS files, 
    leveraging multiprocessing for enhanced performance.

It normalizes the time and flux data and saves the processed data into CSV files.
"""

import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm

def process_file(file_path, save_root):
    """
    Process a single FITS file: read, normalize data, and save as CSV.
    """
    try:
        with fits.open(file_path) as f:
            # Extracting essential information from the header
            header = f[0].header
            kepid = header.get("KEPLERID")
            quarter = header.get("QUARTER")

            # Time normalization
            bkjd = f[1].data["TIME"]
            bkjd_nan = np.isnan(bkjd)
            bkjd[bkjd_nan] = np.interp(np.flatnonzero(bkjd_nan), np.flatnonzero(~bkjd_nan), bkjd[~bkjd_nan])
            jd = bkjd + 2454833.0  # Convert BKJD to JD
            date = Time(jd, format="jd").to_datetime()
            date = pd.to_datetime(pd.Series(date)).dt.strftime("%Y-%m-%d %H:%M:%S")

            # Flux normalization
            flux = f[1].data["PDCSAP_FLUX"]
            median = np.nanmedian(flux)
            q1 = np.nanpercentile(flux, 25)
            q3 = np.nanpercentile(flux, 75)
            iqr = q3 - q1
            iqr = 1 if iqr == 0 else iqr  # Avoid division by zero
            flux = (flux - median) / iqr  # Scaling

            # Preparing DataFrame and saving as CSV
            df = pd.DataFrame({"date": date, "flux": flux})
            df.set_index("date", inplace=True)
            dir_structure = os.path.dirname(file_path).split(os.sep)[-2:]
            save_dir = os.path.join(save_root, *dir_structure)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{str(kepid).zfill(9)}-{str(quarter).zfill(2)}.csv")
            df.to_csv(save_path)
    except Exception as e:
        print(f"Error processing file: {file_path}. Error: {e}")

def main(root, save_root):
    """
    Execute the processing of FITS files using multiprocessing.
    """
    files_to_process = []

    # Walk through the directory tree and collect all files to process
    for dir, subdirs, files in os.walk(root):
        for file in files:
            file_path = os.path.join(dir, file)
            files_to_process.append((file_path, save_root))

    # Multiprocessing execution setup
    num_processes = multiprocessing.cpu_count()

    # This part is adjusted to work with tqdm, ensuring that the progress bar is displayed correctly.
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Prepare the progress bar
        progress_bar = tqdm(total=len(files_to_process), desc="Processing files", unit="file")

        # We use a list to keep all futures, and we use it to know when a future is completed.
        futures = [executor.submit(process_file, *task) for task in files_to_process]

        for _ in as_completed(futures):  # as each future completes, it returns the result.
            progress_bar.update(1)  # Update the progress bar by 1 for each completed task.

        progress_bar.close()  # Don't forget to close the progress bar at the end.

# Paths for the source and destination directories
root = "/media/jmh/T7/kepler_lc"
save_root = "/home/jmh/Datasets/kepler_csv/"

# Running the main processing function
main(root, save_root)
```

```python
"""
Script: Compress Subdirectories

Description:
    This script is used to compress the subdirectories located in a specified "original" root directory.
    Each subdirectory corresponds to a 'sub_id' and is compressed into an individual .tar.gz file.
    The resulting compressed files are saved in a "new" specified root directory.

    The script assumes that directories under the original root are structured as:
    original_root/sub_id

    The script does not recursively compress subdirectories beyond the immediate children
    of the original root directory. The script only targets directories, ignoring any
    files present in the original root directory.

Usage:
    The script is executed in a Python environment and does not require any command-line arguments.
    Users must modify the 'original_root_dir' and 'new_root_dir' variables to set the paths
    for source and destination directories, respectively.

    original_root_dir = "/path/to/original/root"  # path to the source directory
    new_root_dir = "/path/to/new/root"  # path to the destination directory
"""

import os
import tarfile
import concurrent.futures
from tqdm import tqdm

def compress_directory(sub_id_path, new_root):
    """
    Compress the contents of 'sub_id_path' and save it in 'new_root'.

    :param sub_id_path: Full path to the subdirectory.
    :param new_root: Directory where the .tar.gz files will be saved.
    """
    sub_id = os.path.basename(sub_id_path)
    tar_path = os.path.join(new_root, f"{sub_id}.tar.gz")

    # Open tarball file for writing with gzip compression
    with tarfile.open(tar_path, "w:gz") as tar:
        # Walk through each file in the sub_id directory
        for dirpath, dirnames, filenames in os.walk(sub_id_path):
            for file in filenames:
                # Construct the absolute path of the file
                absolute_file_path = os.path.join(dirpath, file)

                # Identify the relative path of the file, which should be the path within the tarball
                # This step is crucial to avoid including unnecessary path information in the tarball
                arcname = os.path.relpath(absolute_file_path, start=sub_id_path)

                # Add the file with its relative path information to the tarball
                tar.add(absolute_file_path, arcname=arcname)

    return f"Compressed {sub_id_path} into {tar_path}"  # Return status


def main(original_root, new_root):
    # Create new root if it doesn't exist.
    if not os.path.exists(new_root):
        os.makedirs(new_root)

    # Construct a list of directories to be compressed
    dirs_to_compress = [os.path.join(original_root, sub_id) for sub_id in os.listdir(original_root) \
                         if os.path.isdir(os.path.join(original_root, sub_id))]
    
    # Number of directories to compress, for the progress bar
    num_dirs = len(dirs_to_compress)

    # We use a ThreadPoolExecutor to compress directories in separate threads.
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Setup tqdm to track the task's progress
        progress = tqdm(
            concurrent.futures.as_completed([
                executor.submit(compress_directory, dir_path, new_root) for dir_path in dirs_to_compress
            ]), 
            total=num_dirs, 
            unit="dir", 
            desc="Compressing", 
            leave=True
        )

        for future in progress:
            try:
                # Fetch the result and update the progress bar description
                result = future.result()
                progress.set_description(f"Processed {result}")
            except Exception as e:
                print(f"Exception occurred: {e}")

    # Close the progress bar
    progress.close()

# Set your original and new root directories.
original_root_dir = "/home/jmh/Datasets/kepler_csv/"
new_root_dir = "/home/jmh/Datasets/kepler_csv_compressed/"

main(original_root_dir, new_root_dir)
```

```python
"""
Multi-Process Time Series Data Cleaning Script

This script is designed to process large sets of time series data stored in CSV files.
It performs the following operations:
1. Removes any rows with NaNs at the beginning and end of the time series data.
2. Applies third-order spline interpolation for missing values within the series.
3. Saves the cleaned data back to CSV, replacing the original files.

The script leverages multiprocessing for enhanced performance, particularly when dealing
with a large number of files. It also implements progress tracking, giving real-time feedback
on the number of files processed.

Usage:
    Define the root directory containing the CSV files and the save directory for processed files.
    Run the script, and it will process files located in 'root/sub_id/id/*.csv'.
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import splrep, splev
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm

def process_time_series(file_path, save_root):
    """
    Processes a single time series CSV file, handling missing data and saving the modified file.
    
    :param file_path: Path to the original CSV file.
    :param save_root: Root directory where the processed file will be saved.
    """
    try:
        # Load and prepare the data
        data = pd.read_csv(file_path)
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        data.sort_index(inplace=True)
        
        # 1. Remove NaNs at the beginning and the end
        # Identify valid data points (non-NaN) from start and end
        valid_start = data['flux'].first_valid_index()
        valid_end = data['flux'].last_valid_index()

        # Drop the rows with NaNs at the beginning and end of the dataframe
        if valid_start is not None and valid_end is not None:
            data = data.loc[valid_start:valid_end]
        else:
            # If the entire series is NaN, we skip the file
            print(f"File {file_path} was skipped as it contains all NaNs.")
            return
        
        # 2. Spline interpolation for NaNs in the middle of the series
        # Identify where the NaNs are in the series
        nan_positions = data['flux'].isna()

        # If there are NaNs to interpolate
        if nan_positions.any():
            # Extract the positions (indices) and values for non-NaN elements
            x = np.where(~nan_positions)[0]
            y = data.loc[~nan_positions, 'flux']

            # Create a spline of degree 3 (cubic spline)
            spline = splrep(x, y, k=3)

            # Identify the positions where we have NaNs
            x_interp_positions = np.where(nan_positions)[0]

            # Calculate the interpolated values at these positions
            interpolated_values = splev(x_interp_positions, spline)

            # Replace NaNs with interpolated values in the dataframe
            data.loc[nan_positions, 'flux'] = interpolated_values
        
        # Construct the save path from the original file path structure
        dir_structure = os.path.dirname(file_path).split(os.sep)[-2:]
        save_dir = os.path.join(save_root, *dir_structure)
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.basename(file_path)
        save_path = os.path.join(save_dir, file_name)

        # Save the cleaned data
        data.to_csv(save_path)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main(root, save_root):
    """
    Orchestrates the multiprocessing execution for cleaning time series data.
    
    :param root: Root directory containing the unprocessed CSV files.
    :param save_root: Root directory where processed files will be stored.
    """
    # Retrieve file paths
    files_to_process = [os.path.join(dir, file) 
                        for dir, _, files in os.walk(root) for file in files if file.endswith(".csv")]

    # Setup for multiprocessing
    num_processes = multiprocessing.cpu_count()

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Initialize the progress bar
        progress_bar = tqdm(total=len(files_to_process), desc="Processing files", unit="file")

        # Submit tasks to the process pool
        futures = [executor.submit(process_time_series, file_path, save_root) for file_path in files_to_process]

        # Update progress for each completed task
        for _ in as_completed(futures):
            progress_bar.update(1)
            progress_bar.refresh()

        # Cleanup
        progress_bar.close() 

# Specify source and destination directories
root = "/home/jmh/Datasets/kepler/"  # Root directory with original CSV files
save_root = "/home/jmh/Datasets/kepler_filled/"  # Destination for processed data

# Start the processing
main(root, save_root)
```

