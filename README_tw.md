[English](./README.md) | **[中文](./README_tw.md)**

# Capybara

<p align="left">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.10+-aff.svg"></a>
    <a href="https://github.com/DocsaidLab/Capybara/releases"><img src="https://img.shields.io/github/v/release/DocsaidLab/Capybara?color=ffa"></a>
    <a href="https://pypi.org/project/capybara_docsaid/"><img src="https://img.shields.io/pypi/v/capybara_docsaid.svg"></a>
    <a href="https://pypi.org/project/capybara_docsaid/"><img src="https://img.shields.io/pypi/dm/capybara_docsaid?color=9cf"></a>
</p>

## 介紹

![title](https://raw.githubusercontent.com/DocsaidLab/Capybara/refs/heads/main/docs/title.webp)

本專案是一個影像處理與深度學習的工具箱，主要包括以下幾個部分：

- **Vision**：提供與電腦視覺相關的功能，例如圖像和影片處理。
- **Structures**：用於處理結構化數據的模組，例如 BoundingBox 和 Polygon。
- **ONNXEngine**：提供 ONNX 推理功能，支援 ONNX 格式的模型。
- **Utils**：放置無法歸類到其他模組的工具函式。
- **Tests**：包含各類功能的測試程式碼，用於驗證函式的正確性。

## 技術文件

若想進一步瞭解安裝與使用方式，請參閱 [**Capybara Documents**](https://docsaid.org/docs/capybara)。

該文件提供本專案的詳細說明與常見問題解答。

## 安裝

在開始安裝 Capybara 之前，請先確保系統符合以下需求：

### Python 版本

- 需要 Python 3.10 或以上版本。

### 依賴套件

請依照作業系統，安裝下列必要的系統套件：

- **Ubuntu**

  ```bash
  sudo apt install libturbojpeg exiftool ffmpeg libheif-dev
  ```

- **MacOS**

  ```bash
  brew install jpeg-turbo exiftool ffmpeg
  ```

  - **特別注意**：經過測試，在 macOS 上使用 libheif 時，存在一些已知問題，主要包括：

    1. **生成的 HEIC 檔案無法打開**：在 macOS 上，libheif 生成的 HEIC 檔案可能無法被某些程式打開。這可能與圖像尺寸有關，特別是當圖像的寬度或高度為奇數時，可能會導致相容性問題。

    2. **編譯錯誤**：在 macOS 上編譯 libheif 時，可能會遇到與 ffmpeg 解碼器相關的未定義符號錯誤。這可能是由於編譯選項或相依性設定不正確所致。

    3. **範例程式無法執行**：在 macOS Sonoma 上，libheif 的範例程式可能無法正常運行，出現動態鏈接錯誤，提示找不到 `libheif.1.dylib`，這可能與動態庫的路徑設定有關。

    由於問題不少，因此我們目前只在 Ubuntu 才會運行 libheif，至於 macOS 的部分則留給未來的版本。

### pdf2image 依賴套件

pdf2image 是用於將 PDF 文件轉換成影像的 Python 模組，請確保系統已安裝下列工具：

- MacOS：需要安裝 poppler

  ```bash
  brew install poppler
  ```

- Linux：大多數發行版已內建 `pdftoppm` 與 `pdftocairo`。如未安裝，請執行：

  ```bash
  sudo apt install poppler-utils
  ```

### ONNXRuntime GPU 依賴

若需使用 ONNXRuntime 進行 GPU 加速推理，請確保已安裝相容版本的 CUDA，如下示範：

```bash
sudo apt install cuda-12-4
# 假設要加入至 .bashrc
echo 'export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
```

### 透過 PyPI 安裝

1. 透過 PyPI 安裝套件：

   ```bash
   pip install capybara-docsaid
   ```

2. 驗證安裝：

   ```bash
   python -c "import capybara; print(capybara.__version__)"
   ```

3. 若顯示版本號，則安裝成功。

### 透過 git clone 安裝

1. 下載本專案：

   ```bash
   git clone https://github.com/DocsaidLab/Capybara.git
   ```

2. 安裝 wheel 套件：

   ```bash
   pip install wheel
   ```

3. 建構 wheel 檔案：

   ```bash
   cd Capybara
   python setup.py bdist_wheel
   ```

4. 安裝建置完成的 wheel 檔：

   ```bash
   pip install dist/capybara_docsaid-*-py3-none-any.whl
   ```

### 透過 docker 安裝（建議）

若想在部署或協同開發時避免環境衝突，建議使用 Docker，以下為簡要示範流程：

1. 下載本專案：

   ```bash
   git clone https://github.com/DocsaidLab/Capybara.git
   ```

2. 進入專案資料夾，執行建置腳本：

   ```bash
   cd Capybara
   bash docker/build.bash
   ```

   這會使用專案中的 [**Dockerfile**](https://github.com/DocsaidLab/Capybara/blob/main/docker/Dockerfile) 來建立映像檔；映像檔預設以 `nvcr.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` 為基底，提供 ONNXRuntime 推理所需的 CUDA 環境。

3. 建置完成後，使用指令掛載工作目錄並執行程式：

   ```bash
   docker run -v ${PWD}:/code -it capybara_infer_image your_scripts.py
   ```

   若需 GPU 加速，可於執行時加入 `--gpus all`。

#### gosu 權限問題

若在容器內執行腳本時，遇到輸出檔案歸屬為 root，導致檔案權限不便的情況，可在 Dockerfile 中加入 `gosu` 進行使用者切換，並在容器啟動時指定 `USER_ID` 與 `GROUP_ID`。
這樣可避免在多位開發者協作時，需要頻繁調整檔案權限的問題。

具體作法可參考技術文件：[**Integrating gosu Configuration**](https://docsaid.org/docs/capybara/advance/#integrating-gosu-configuration)

1. 安裝 `gosu`：

   ```dockerfile
   RUN apt-get update && apt-get install -y gosu
   ```

2. 在容器啟動指令中使用 `gosu` 切換至容器內的非 root 帳號，以利檔案的讀寫。

   ```dockerfile
   # Create the entrypoint script
   RUN printf '#!/bin/bash\n\
       if [ ! -z "$USER_ID" ] && [ ! -z "$GROUP_ID" ]; then\n\
           groupadd -g "$GROUP_ID" -o usergroup\n\
           useradd --shell /bin/bash -u "$USER_ID" -g "$GROUP_ID" -o -c "" -m user\n\
           export HOME=/home/user\n\
           chown -R "$USER_ID":"$GROUP_ID" /home/user\n\
           chown -R "$USER_ID":"$GROUP_ID" /code\n\
       fi\n\
       \n\
       # Check for parameters\n\
       if [ $# -gt 0 ]; then\n\
           exec gosu ${USER_ID:-0}:${GROUP_ID:-0} python "$@"\n\
       else\n\
           exec gosu ${USER_ID:-0}:${GROUP_ID:-0} bash\n\
       fi' > "$ENTRYPOINT_SCRIPT"

   RUN chmod +x "$ENTRYPOINT_SCRIPT"

   ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
   ```

更多進階配置請參考 [**NVIDIA Container Toolkit**](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 及 [**docker**](https://docs.docker.com/) 官方文件。

## 測試

本專案使用 `pytest` 進行單元測試，用戶可自行運行測試以驗證功能的正確性。
安裝並執行測試的方式如下：

```bash
pip install pytest
python -m pytest -vv tests
```

完成後即可確認各模組運作是否正常。若遇到功能異常，請先檢查環境設定與套件版本。

若仍無法解決，可至 Issue 區回報。
