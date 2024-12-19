# Other functions

Utils 模組提供了多個實用的工具函數和類，這些函數和類旨在方便和加速日常的程式開發和數據處理工作。

---

### [custom-path.py](../capybara/utils/custom_path.py)

**功能**: 提供與文件和目錄路徑操作相關的功能。

- **主要函數及說明**:

  - `get_curdir()`:

    - **目的**: 獲取當前工作區的路徑。
    - **參數**:
      - `path`: 文件路徑，可以是字符串或 Path 對象。
      - `absolute`(預設為 True): 是否返回絕對路徑。
    - **返回**: 文件所在的資料夾路徑。

  - `rm_path()`:

    - **目的**: 刪除指定的文件或目錄。
    - **參數**:
      - `path`: 需要刪除的文件或目錄的路徑。
    - **返回**: 無。

---

### [custom-tqdm.py](../capybara/utils/custom_tqdm.py)

**功能**: 提供一個自定義的進度條功能。

- **主要類和說明**:

  - `Tqdm`: 這是一個自定義的進度條類，繼承自原始的`tqdm`類。

---

### [files-utils.py](../capybara/utils/files_utils.py)

**功能**: 提供文件操作的實用函數。

- **主要函數及說明**:

  - `get_files()`:

    - **目的**: 從指定資料夾中獲取所有文件。
    - **參數**:
      - `folder`: 目標資料夾。
      - `suffix`(可選): 要查找的文件的後綴名。
      - `recursive`(預設為 True): 是否遞歸查找子資料夾。
      - `return_pathlib`(預設為 True): 是否返回 Path 對象。
      - `sort_path`(預設為 True): 是否對路徑進行排序。
      - `ignore_letter_case`(預設為 True): 在比較文件後綴名時是否忽略大小寫。
    - **返回**: 文件列表。

---

### [powerdict.py](../capybara/utils/powerdict.py)

**功能**: 提供操作和管理字典的強大功能。

- **主要類和說明**:

  - `PowerDict`: 這是一個繼承自原始字典的類，它允許用戶以屬性的形式存取鍵值，並提供額外的方法來"凍結"和"解凍"字典。

---

### [time.py](../capybara/utils/time.py)

**功能**: 提供與時間相關的工具函數。

- **主要函數及說明**:

  - `Timer`, `now`, `timestamp2datetime`...: 這些函數提供從不同格式之間進行轉換的功能，例如從時間戳到字符串，從 datetime 對象到時間戳等等。

---

### [utils.py](../capybara/utils/utils.py)

**功能**: 提供多種通用的工具函數。

- **主要函數及說明**:

  - `make_batch()`:

    - **目的**: 將數據分批處理。
    - **參數**:
      - `data`: 需要分批的數據，可以是一個可迭代對象或生成器。
      - `batch_size`: 每批數據的大小。
    - **返回**: 返回分批的數據生成器。

  - `colorstr()`:

    - **目的**: 為輸出的文本添加顏色。
    - **參數**:
      - `obj`: 需要著色的對象。
      - `color`: 文本的顏色。
      - `fmt`: 文本的格式。
    - **返回**: 著色後的文本字符串。

---

### [system_info.py](../capybara/utils/system_info.py)

**功能**: 提供與系統和環境信息收集相關的功能。

- **主要函數及說明**:

  - `get_package_versions()`:

    - **目的**: 獲取常用於深度學習和數據科學的包的版本。
    - **返回**: 包含已安裝包版本的字典。

  - `get_gpu_cuda_versions()`:

    - **目的**: 使用流行的 Python 庫獲取 GPU 和 CUDA 版本。
    - **返回**: 包含 CUDA 和 GPU 驅動版本的字典。

  - `get_cpu_info()`:

    - **目的**: 根據平台獲取 CPU 模型名稱。
    - **返回**: CPU 模型名稱，或在未找到時返回 "N/A"。

  - `get_external_ip()`:

    - **目的**: 獲取外部 IP 地址。
    - **返回**: 外部 IP 地址，或在出錯時返回錯誤信息。

  - `get_system_info()`:
    - **目的**: 獲取系統信息，如操作系統版本、CPU 信息、內存、磁盤使用情況等。
    - **返回**: 包含系統信息的字典。

---
