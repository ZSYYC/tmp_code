好的，我們只關注 Vision-Only 模型的 TensorRT 加速部署。這是一個非常明智的選擇，因為視覺模型的計算密集度高，從 TensorRT 加速中獲益最大。

將 `InternVideo2_CLIP_small_vision_only` 模型部署到 TensorRT 是一個多步驟的過程，但遵循正確的流程可以確保成功。我將為您提供一個完整且詳細的指南，包含所有必要的程式碼和解釋。

### 部署流程概覽

整個流程分為三個主要階段：

1.  **PyTorch -\> ONNX**: 將您的 PyTorch 模型導出為 ONNX (Open Neural Network Exchange) 格式。ONNX 是一個中間表示，充當了不同深度學習框架之間的橋樑。
2.  **ONNX -\> TensorRT Engine**: 使用 TensorRT 解析 ONNX 文件，並根據您的目標 GPU 進行高度優化，生成一個稱為 "Engine" 或 "Plan" 的文件。這是 TensorRT 的核心步驟。
3.  **TensorRT Engine 推理**: 在您的部署環境中，加載這個 Engine 文件並執行高速推理。

-----

### 前置準備

在開始之前，請確保您的部署環境已經安裝好以下軟體：

  * **NVIDIA 驅動程式**: 最新版的 NVIDIA 顯卡驅動。
  * **CUDA Toolkit**: 與 TensorRT 版本兼容的 CUDA 版本（例如 CUDA 11.x 或 12.x）。
  * **cuDNN**: 與 CUDA 版本兼容的 cuDNN 庫。
  * **TensorRT**: 推薦安裝 8.x 或更高版本。可以從 NVIDIA 官網下載，解壓後將其 `lib` 目錄添加到環境變量 `LD_LIBRARY_PATH` 中。
  * **Python 依賴**:
    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pip install tensorrt # 如果使用 Python Wheel 安裝
    pip install onnx onnxruntime-gpu
    pip install pycuda # 用於 GPU 記憶體管理
    ```

-----

### 步驟一：將 PyTorch 模型導出為 ONNX

首先，我們需要將 `InternVideo2_CLIP_small_vision_only` 模型實例轉換為 ONNX 文件。這個過程的關鍵是提供一個**虛擬輸入 (dummy input)**，並指定**動態軸 (dynamic axes)**，這樣導出的模型才能處理不同批量大小 (batch size) 或幀數 (frame count) 的輸入。

建立一個名為 `export_onnx.py` 的腳本：

```python
import torch
from internvideo2_clip_small_vision_only import InternVideo2_CLIP_small_vision_only
# 假設您的 config 加載邏輯在這裡
from your_config_loader import config 

def main():
    # --- 1. 加載您的 PyTorch 模型 ---
    print("Loading PyTorch model...")
    # 確保權重路徑正確
    config.model.vision_ckpt_path = "/path/to/your/trained_model.pth"
    
    # 設置為 eval 模式，這很重要，會關閉 dropout 等層
    model = InternVideo2_CLIP_small_vision_only(config=config, is_pretrain=False).to(config.device)
    model.eval()
    print("PyTorch model loaded successfully.")

    # --- 2. 準備虛擬輸入和動態軸 ---
    # 模型的 encode_vision 方法接收 [B, T, C, H, W] 格式的輸入
    # 我們將 Batch Size 和 Time (幀數) 設置為動態
    batch_size = 1
    num_frames = 8 # 您的模型訓練時使用的典型幀數
    img_size = config.model.vision_encoder.img_size
    
    dummy_input = torch.randn(batch_size, num_frames, 3, img_size, img_size, device=config.device)
    
    onnx_output_path = "internvideo2_vision.onnx"

    dynamic_axes = {
        'input_frames': {0: 'batch_size', 1: 'num_frames'}, # 輸入的第0維和第1維是動態的
        'output_features': {0: 'batch_size'} # 輸出的第0維是動態的
    }

    # --- 3. 執行導出 ---
    print(f"Exporting model to ONNX at {onnx_output_path}...")
    torch.onnx.export(
        model.encode_vision,       # 我們導出的是核心的 encode_vision 方法
        dummy_input,               # 虛擬輸入
        onnx_output_path,          # 輸出路徑
        opset_version=14,          # ONNX 操作集版本，14 或更高比較穩定
        input_names=['input_frames'], # 輸入節點名
        output_names=['output_features'], # 輸出節點名
        dynamic_axes=dynamic_axes, # 指定動態軸
        verbose=False
    )
    
    print("ONNX export completed successfully.")
    
    # --- 4. (可選但強烈建議) 驗證 ONNX 模型 ---
    print("Validating ONNX model with onnxruntime...")
    import onnxruntime
    import numpy as np

    # 使用 PyTorch 產生結果
    with torch.no_grad():
        pytorch_output = model.encode_vision(dummy_input).cpu().numpy()

    # 使用 ONNX Runtime 產生結果
    ort_session = onnxruntime.InferenceSession(onnx_output_path, providers=['CUDAExecutionProvider'])
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]

    # 比較結果
    if np.allclose(pytorch_output, ort_output, atol=1e-4):
        print("ONNX model validation successful: Outputs are close.")
    else:
        print("ONNX model validation FAILED: Outputs differ significantly.")

if __name__ == '__main__':
    main()
```

**執行此腳本**: `python export_onnx.py`。成功後，您將得到一個 `internvideo2_vision.onnx` 文件。

-----

### 步驟二：從 ONNX 建立 TensorRT Engine

現在我們有了 ONNX 文件，接下來使用 TensorRT 將其轉換為高度優化的 Engine。在這個階段，您可以選擇優化的精度，如 FP32、FP16 或 INT8。**FP16 (半精度)** 通常是最佳選擇，它能在幾乎不損失精度的情況下，提供顯著的速度提升，尤其是在支持 Tensor Core 的 GPU (如 Turing, Ampere, Ada Lovelace 架構) 上。

建立一個名為 `build_engine.py` 的腳本：

```python
import tensorrt as trt

def build_engine(onnx_path, engine_path, use_fp16=True):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # 1. 創建 Builder, Network 和 Parser
    builder = trt.Builder(TRT_LOGGER)
    # EXPLICIT_BATCH 標誌是 TensorRT 7.0+ 的標準做法
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # 2. 設置 Builder 配置
    config = builder.create_builder_config()
    # 設置工作區大小，根據您的 GPU RAM 調整
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1 GB
    
    if use_fp16 and builder.platform_has_fast_fp16:
        print("Enabling FP16 mode.")
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        print("Using FP32 mode.")
        
    # 3. 解析 ONNX 模型
    print(f"Loading ONNX file from: {onnx_path}")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    print(f"Completed parsing of ONNX file.")
    
    # 4. 處理動態尺寸
    input_tensor = network.get_input(0)
    profile = builder.create_optimization_profile()
    
    # 定義輸入尺寸的最小、最優和最大值
    # [batch_size, num_frames, C, H, W]
    # C, H, W 是固定的
    _, _, C, H, W = input_tensor.shape
    
    # 根據您的部署需求調整這些值
    # min: 最小負載, opt: 常見負載, max: 最大負載
    profile.set_shape(
        input_tensor.name, 
        min=(1, 1, C, H, W),         # Batch=1, Frames=1
        opt=(4, 8, C, H, W),         # Batch=4, Frames=8
        max=(16, 16, C, H, W)        # Batch=16, Frames=16
    )
    config.add_optimization_profile(profile)

    # 5. 建立並序列化 Engine
    print("Building TensorRT engine... (This may take a few minutes)")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("ERROR: Failed to build the engine.")
        return None

    print("Engine built successfully.")
    
    # 6. 保存 Engine 到文件
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    print(f"Engine saved to: {engine_path}")


if __name__ == '__main__':
    ONNX_PATH = "internvideo2_vision.onnx"
    ENGINE_PATH = "internvideo2_vision_fp16.engine"
    build_engine(ONNX_PATH, ENGINE_PATH, use_fp16=True)
```

**執行此腳本**: `python build_engine.py`。這一步可能需要幾分鐘時間。成功後，您將得到一個 `internvideo2_vision_fp16.engine` 文件。這個文件是針對您當前 GPU 架構優化的，**不可跨 GPU 型號移植**。

-----

### 步驟三：使用 TensorRT Engine 進行推理

最後一步是在您的應用程式中加載 Engine 並執行推理。這部分程式碼比 PyTorch 稍微複雜，因為需要手動管理 GPU 記憶體。我會提供一個封裝好的類別，讓使用起來更方便。

建立一個名為 `trt_inference.py` 的腳本：

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torch # 用於後處理

class TensorRTInfer:
    def __init__(self, engine_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        
        # 1. 加載 Engine
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        
        # 2. 創建執行上下文
        self.context = self.engine.create_execution_context()
        
        # 3. 分配記憶體 (Host and Device Buffers)
        self.host_inputs = []
        self.host_outputs = []
        self.device_inputs = []
        self.device_outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.device_inputs.append(device_mem)
            else:
                self.host_outputs.append(host_mem)
                self.device_outputs.append(device_mem)

    def infer(self, input_data: np.ndarray):
        # 設置當前輸入的 shape
        self.context.set_binding_shape(0, input_data.shape)
        
        # 將輸入數據從 Host (CPU) 複製到 Device (GPU)
        np.copyto(self.host_inputs[0], input_data.ravel())
        cuda.memcpy_htod_async(self.device_inputs[0], self.host_inputs[0], self.stream)
        
        # 執行推理
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # 將輸出結果從 Device 複製到 Host
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.device_outputs[0], self.stream)
        
        # 同步流，等待推理完成
        self.stream.synchronize()
        
        # 獲取輸出並 reshape
        output_shape = self.context.get_binding_shape(1) # 獲取輸出 binding 的 shape
        output_data = self.host_outputs[0].reshape(output_shape)
        
        return output_data

def main():
    print("Loading TensorRT engine and creating inference wrapper...")
    ENGINE_PATH = "internvideo2_vision_fp16.engine"
    trt_infer = TensorRTInfer(ENGINE_PATH)
    
    # --- 準備和 PyTorch 模型一樣的輸入 ---
    # 1. 加載配置和預處理
    from your_config_loader import config
    from internvideo2_clip_small_vision_only import InternVideo2_CLIP_small_vision_only
    # 我們需要 PyTorch 模型的 transform 來預處理影像
    pytorch_model = InternVideo2_CLIP_small_vision_only(config=config, is_pretrain=False)
    
    # 2. 創建一個模擬的輸入 tensor
    batch_size = 4
    num_frames = 8
    img_size = config.model.vision_encoder.img_size
    
    # 模擬的影像幀 [B, T, H, W, C] numpy 格式
    dummy_frames_np = np.random.randint(0, 256, 
        (batch_size, num_frames, img_size, img_size, 3), dtype=np.uint8)

    # 3. 進行預處理
    preprocessed_frames = []
    for i in range(batch_size):
        frames_for_one_video = torch.from_numpy(dummy_frames_np[i]) # [T, H, W, C]
        frames_for_one_video = frames_for_one_video.permute(0, 3, 1, 2) # [T, C, H, W]
        
        # 使用 PyTorch 模型的 transform
        processed = pytorch_model.transform(frames_for_one_video) # 輸出 [T, C, H, W]
        preprocessed_frames.append(processed)

    input_tensor = torch.stack(preprocessed_frames) # [B, T, C, H, W]
    input_numpy = input_tensor.numpy()
    
    print(f"Input data shape: {input_numpy.shape}, dtype: {input_numpy.dtype}")

    # --- 執行 TensorRT 推理 ---
    print("Running inference with TensorRT engine...")
    trt_output = trt_infer.infer(input_numpy)
    print(f"TensorRT output shape: {trt_output.shape}")

    # --- 後處理 ---
    # 原始的 get_vid_feat 方法會進行 L2 Normalization
    # 我們在這裡手動完成
    trt_output_torch = torch.from_numpy(trt_output)
    normalized_features = trt_output_torch / trt_output_torch.norm(dim=-1, keepdim=True)
    
    print("Inference and post-processing complete.")
    print(f"Final normalized features shape: {normalized_features.shape}")
    print("Sample of first feature vector:", normalized_features[0, :5])

if __name__ == '__main__':
    main()
```

**執行此腳本**: `python trt_inference.py`。它將加載 `.engine` 文件，執行一次推理，並打印出最終的特徵向量。

### 總結與最佳實踐

1.  **精度選擇**: 從 FP16 開始。如果精度下降過多（很少見），再回退到 FP32。如果追求極致性能，可以研究 INT8 量化，但這需要一個有代表性的校準數據集，過程更複雜。
2.  **優化配置 (`Optimization Profile`)**: `build_engine.py` 中的 `min`, `opt`, `max` 設置非常重要。`opt` 應該設為您最常見的批次大小和幀數，TensorRT 會針對這個尺寸做最優化。`min` 和 `max` 定義了模型可接受的輸入尺寸範圍。
3.  **Engine 的可移植性**: TensorRT Engine 是針對特定的 GPU 型號、CUDA 版本和 TensorRT 版本生成的。如果您的部署環境與開發環境有任何不同，您需要在部署伺服器上重新執行 `build_engine.py` 來生成對應的 Engine。
4.  **預處理/後處理**: TensorRT 只加速神經網路的核心計算部分。數據的預處理（如 `transform`）和後處理（如 `L2-normalize`）仍然需要在您的應用程式中用 PyTorch/NumPy 等工具完成。
5.  **記憶體管理**: 在 `TensorRTInfer` 類中，我們預先分配了最大可能需要的 GPU 記憶體。這是一種高效的做法，避免了每次推理都重新分配記憶體。

遵循以上步驟，您就可以成功地將您的 `InternVideo2_CLIP_small_vision_only` 模型部署到生產環境中，並享受 TensorRT 帶來的極致性能。