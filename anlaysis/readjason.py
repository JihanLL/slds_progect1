import json
import os
def readjosn(Key_name,address):
    # Define the relative path to the JSON file from the current script's location
    # Current script: /d:/.../anlaysis/readjason.py
    # Target file: /d:/.../results/3.3.3/2025-04-30_22-57-31 4layer/metrics.json
    # Need to go up two directories ('../../') from 'anlaysis' to reach the base 'slds_progect1' directory
    relative_path =address #'../results/3.3.3/2/2025-04-30_22-57-31 2layer/metrics.json'#'../../results/3.3.3/2025-04-30_22-57-31 4layer/metrics.json'

    # Get the absolute path to the directory containing the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to the JSON file
    file_path = os.path.join(script_dir, relative_path)

    # Normalize the path (handles '..' etc.)
    file_path = os.path.normpath(file_path)

    # Initialize data variable
    data = None

    try:
        # Open and read the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded data from: {file_path}")
        # You can now work with the 'data' variable (usually a dictionary or list)
        # print(data)

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # Example of accessing data if loading was successful

    if data: # 确保数据已成功加载
        if isinstance(data, dict):
            # 如果是字典，可以通过键访问
            value = data.get(Key_name) # 使用 .get() 避免 KeyError
            #print(f"Key {Key_name} has value: {value}")
            return value
        elif isinstance(data, list):
            # 如果是列表，可以通过索引访问
            if len(data) > 0:
                first_item = data[0]
                print(f"First item: {first_item}")
        # ... 根据你的 JSON 结构进行操作 ...