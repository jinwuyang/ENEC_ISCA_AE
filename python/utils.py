# # import argparse
# # import os
# # import torch
# # import numpy as np
# # from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

# # def get_torch_dtype(dtype_str):
# #     """映射字符串到 torch 精度类型"""
# #     mapping = {
# #         "BF16": torch.bfloat16,
# #         "FP16": torch.float16,
# #         "FP32": torch.float32
# #     }
# #     return mapping.get(dtype_str.upper(), torch.float32)

# # def split_model(model_path, dtype_str):
# #     # 1. 自动推断模型类型并加载
# #     print(f"\n[Processing] 正在加载模型: {model_path}")
# #     target_dtype = get_torch_dtype(dtype_str)
    
# #     save_dir = os.path.join(model_path, 'split')
# #     if os.path.exists(save_dir):
# #         print(f"跳过: {save_dir} 已存在")
# #         return

# #     try:
# #         # 使用 AutoModelForCausalLM 加载，如果失败则尝试通用 AutoModel (针对 BERT/Wav2Vec2)
# #         try:
# #             model = AutoModelForCausalLM.from_pretrained(
# #                 model_path, 
# #                 trust_remote_code=True, 
# #                 torch_dtype=target_dtype,
# #                 device_map="cpu" # 拆分通常在 CPU 内存完成即可
# #             )
# #         except Exception:
# #             model = AutoModel.from_pretrained(
# #                 model_path, 
# #                 trust_remote_code=True, 
# #                 torch_dtype=target_dtype,
# #                 device_map="cpu"
# #             )

# #         os.makedirs(save_dir, exist_ok=True)
# #         model.eval()

# #         # 2. 遍历参数并保存为二进制
# #         with torch.no_grad():
# #             for name, param in model.named_parameters():
# #                 # 排除 1D 参数（如 bias, LayerNorm），通常只压缩 2D 及以上的权重矩阵
# #                 if param.dim() < 2:
# #                     continue
                
# #                 # 转换为对应的 numpy 类型
# #                 if dtype_str.upper() == 'FP32':
# #                     # FP32 视图转换
# #                     param_np = param.detach().view(torch.float32).cpu().numpy()
# #                 else:
# #                     # BF16/FP16 在内存中通常是 uint16 存储
# #                     param_np = param.detach().view(torch.uint16).cpu().numpy()

# #                 param_path = os.path.join(save_dir, f"{name}.bin")
# #                 param_np.tofile(param_path)
# #                 # print(f"  -> 已保存: {name}.bin | Shape: {param_np.shape}")

# #         print(f"✅ 成功: {model_path} 权重已拆分至 {save_dir}")
        
# #         # 释放内存防止 OOM
# #         del model
# #         torch.cuda.empty_cache()

# #     except Exception as e:
# #         print(f"❌ 失败: 处理 {model_path} 时发生错误: {e}")

# # def main():
# #     parser = argparse.ArgumentParser(description="ENEC 批量模型权重拆分工具")
# #     parser.add_argument("--root_dir", type=str, default="models", help="models 根目录路径")
# #     args = parser.parse_args()

# #     if not os.path.exists(args.root_dir):
# #         print(f"错误: 找不到根目录 {args.root_dir}")
# #         return

# #     # 定义要扫描的子目录（精度）
# #     dtypes = ["BF16", "FP16", "FP32"]

# #     for dtype in dtypes:
# #         dtype_path = os.path.join(args.root_dir, dtype)
# #         if not os.path.exists(dtype_path):
# #             continue
        
# #         print(f"\n{'='*20} 开始处理 {dtype} 目录 {'='*20}")
        
# #         # 遍历精度目录下的各个模型文件夹
# #         for model_name in os.listdir(dtype_path):
# #             model_full_path = os.path.join(dtype_path, model_name)
            
# #             # 确保是文件夹且包含配置文件
# #             if os.path.isdir(model_full_path) and \
# #                (os.path.exists(os.path.join(model_full_path, "config.json")) or \
# #                 os.path.exists(os.path.join(model_full_path, "configuration.json"))):
                
# #                 split_model(model_full_path, dtype)

# # if __name__ == "__main__":
# #     main()

# import argparse
# import os
# import torch
# import numpy as np
# import glob
# from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

# def get_torch_dtype(dtype_str):
#     mapping = {
#         "BF16": torch.bfloat16,
#         "FP16": torch.float16,
#         "FP32": torch.float32
#     }
#     return mapping.get(dtype_str.upper(), torch.float32)

# def split_model(model_path, dtype_str, force=False):
#     """
#     拆分权重。
#     force: 如果为 True，即使 split 文件夹存在也会重新生成。
#     """
#     target_dtype = get_torch_dtype(dtype_str)
#     save_dir = os.path.join(model_path, 'split')
    
#     # --- 增强的跳过检测逻辑 ---
#     if not force and os.path.exists(save_dir):
#         # 检查目录下是否有 bin 文件，防止上次处理中断留下空文件夹
#         bin_files = glob.glob(os.path.join(save_dir, "*.bin"))
#         if len(bin_files) > 0:
#             print(f"  [Skip] {model_path} (已存在 {len(bin_files)} 个权重文件)")
#             return False # 返回 False 表示没有进行新的处理
#         else:
#             print(f"  [Retry] {save_dir} 为空，将重新开始...")
    
#     print(f"\n[Processing] 正在处理: {model_path} ({dtype_str})")

#     try:
#         # 加载模型逻辑
#         try:
#             model = AutoModelForCausalLM.from_pretrained(
#                 model_path, 
#                 trust_remote_code=True, 
#                 torch_dtype=target_dtype,
#                 device_map="cpu",
#                 low_cpu_mem_usage=True # 优化内存占用
#             )
#         except Exception:
#             model = AutoModel.from_pretrained(
#                 model_path, 
#                 trust_remote_code=True, 
#                 torch_dtype=target_dtype,
#                 device_map="cpu",
#                 low_cpu_mem_usage=True
#             )

#         os.makedirs(save_dir, exist_ok=True)
#         model.eval()

#         processed_count = 0
#         with torch.no_grad():
#             for name, param in model.named_parameters():
#                 if param.dim() < 2:
#                     continue
                
#                 # 统一视图转换逻辑
#                 if dtype_str.upper() == 'FP32':
#                     param_np = param.detach().view(torch.float32).cpu().numpy()
#                 else:
#                     # BF16/FP16 使用 uint16 视图以保留位信息导出
#                     param_np = param.detach().view(torch.uint16).cpu().numpy()

#                 param_path = os.path.join(save_dir, f"{name}.bin")
#                 param_np.tofile(param_path)
#                 processed_count += 1

#         print(f"  ✅ 成功完成: 共导出 {processed_count} 个 2D+ 权重")
        
#         del model
#         import gc
#         gc.collect()
#         torch.cuda.empty_cache()
#         return True

#     except Exception as e:
#         print(f"  ❌ 失败: {model_path} | 错误: {e}")
#         return False

# def main():
#     parser = argparse.ArgumentParser(description="ENEC 批量模型权重拆分工具 (增强版)")
#     parser.add_argument("--root_dir", type=str, default="models", help="models 根目录路径")
#     parser.add_argument("--force", action="store_true", help="强制覆盖已存在的 split 结果")
#     args = parser.parse_args()

#     if not os.path.exists(args.root_dir):
#         print(f"错误: 找不到根目录 {args.root_dir}")
#         return

#     dtypes = ["BF16", "FP16", "FP32"]
    
#     # 统计信息
#     stats = {"processed": 0, "skipped": 0}

#     for dtype in dtypes:
#         dtype_path = os.path.join(args.root_dir, dtype)
#         if not os.path.exists(dtype_path):
#             continue
        
#         print(f"\n{'#'*30}")
#         print(f"## 进入精度目录: {dtype}")
#         print(f"{'#'*30}")
        
#         model_folders = [f for f in os.listdir(dtype_path) if os.path.isdir(os.path.join(dtype_path, f))]
        
#         for model_name in model_folders:
#             model_full_path = os.path.join(dtype_path, model_name)
            
#             # 基础结构检查
#             is_model = any(os.path.exists(os.path.join(model_full_path, cfg)) 
#                           for cfg in ["config.json", "configuration.json"])
            
#             if is_model:
#                 did_work = split_model(model_full_path, dtype, force=args.force)
#                 if did_work:
#                     stats["processed"] += 1
#                 else:
#                     stats["skipped"] += 1

#     print(f"\n{'='*40}")
#     print(f"任务结束汇总:")
#     print(f" - 新处理模型数: {stats['processed']}")
#     print(f" - 已存在跳过数: {stats['skipped']}")
#     print(f"{'='*40}\n")

# if __name__ == "__main__":
#     main()

import argparse
import os
import torch
import numpy as np
import glob
import gc
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from safetensors.torch import load_file

def get_torch_dtype(dtype_str):
    mapping = {
        "BF16": torch.bfloat16,
        "FP16": torch.float16,
        "FP32": torch.float32
    }
    return mapping.get(dtype_str.upper(), torch.float32)

def save_tensor_to_bin(name, param, save_dir, dtype_str):
    """提取权重并保存为二进制"""
    if param.dim() < 2:
        return False
    
    # 统一视图转换逻辑
    if dtype_str.upper() == 'FP32':
        param_np = param.detach().view(torch.float32).cpu().numpy()
    else:
        # BF16/FP16 使用 uint16 视图以保留原始位信息
        param_np = param.detach().view(torch.uint16).cpu().numpy()

    param_path = os.path.join(save_dir, f"{name.replace('/', '.')}.bin")
    param_np.tofile(param_path)
    return True

def split_model(model_path, dtype_str, force=False):
    target_dtype = get_torch_dtype(dtype_str)
    save_dir = os.path.join(model_path, 'split')
    
    # 断点续传检查
    if not force and os.path.exists(save_dir):
        bin_files = glob.glob(os.path.join(save_dir, "*.bin"))
        if len(bin_files) > 0:
            print(f"  [Skip] {model_path} (已存在 {len(bin_files)} 个权重)")
            return False
    
    print(f"\n[Processing] 正在处理: {model_path} ({dtype_str})")
    os.makedirs(save_dir, exist_ok=True)

    try:
        loaded_params = {}
        
        # 策略 A: 尝试作为扩散模型加载 (查找 unet 目录)
        unet_path = os.path.join(model_path, "unet")
        if os.path.exists(unet_path):
            print("  -> 检测到扩散模型结构，正在提取 UNet 权重...")
            from diffusers import UNet2DConditionModel, UNetSpatioTemporalConditionModel
            try:
                # 针对 SVD 等时空模型
                m = UNetSpatioTemporalConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=target_dtype)
            except:
                # 针对普通 SD 模型
                m = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=target_dtype)
            loaded_params = dict(m.named_parameters())
            del m

        # 策略 B: 尝试作为标准 Transformers 加载
        if not loaded_params:
            try:
                print("  -> 尝试作为 Transformers 模型加载...")
                try:
                    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=target_dtype, device_map="cpu", low_cpu_mem_usage=True)
                except:
                    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=target_dtype, device_map="cpu", low_cpu_mem_usage=True)
                loaded_params = dict(model.named_parameters())
                del model
            except Exception as e:
                print(f"  -> 无法通过类加载，尝试直接读取权重文件... ({e})")

        # 策略 C: 万能读取 (不依赖模型定义，直接读取磁盘上的 safetensors/bin)
        if not loaded_params:
            # 优先找 safetensors，再找 bin
            weight_files = glob.glob(os.path.join(model_path, "**/*.safetensors"), recursive=True) + \
                           glob.glob(os.path.join(model_path, "**/*.bin"), recursive=True)
            
            for wf in weight_files:
                if "split" in wf: continue # 避开自己生成的目录
                try:
                    if wf.endswith(".safetensors"):
                        weights = load_file(wf, device="cpu")
                    else:
                        weights = torch.load(wf, map_location="cpu")
                    
                    if isinstance(weights, dict):
                        for k, v in weights.items():
                            if isinstance(v, torch.Tensor):
                                loaded_params[f"{os.path.basename(wf)}_{k}"] = v.to(target_dtype)
                except:
                    continue

        # 执行保存
        if not loaded_params:
            raise ValueError("未能从该目录提取到任何权重参数")

        count = 0
        for name, param in loaded_params.items():
            if save_tensor_to_bin(name, param, save_dir, dtype_str):
                count += 1
        
        print(f"  ✅ 成功完成: 导出 {count} 个 2D+ 权重")
        
    except Exception as e:
        print(f"  ❌ 失败: {model_path} | 错误: {e}")
        return False
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return True

def main():
    parser = argparse.ArgumentParser(description="ENEC 权重拆分工具 (全模型兼容版)")
    parser.add_argument("--root_dir", type=str, default="models", help="models 根目录路径")
    parser.add_argument("--force", action="store_true", help="强制覆盖结果")
    args = parser.parse_args()

    if not os.path.exists(args.root_dir):
        print(f"错误: 找不到根目录 {args.root_dir}")
        return

    dtypes = ["BF16", "FP16", "FP32"]
    stats = {"processed": 0, "skipped": 0}

    for dtype in dtypes:
        dtype_path = os.path.join(args.root_dir, dtype)
        if not os.path.exists(dtype_path): continue
        
        print(f"\n{'#'*40}\n## 精度目录: {dtype}\n{'#'*40}")
        
        for model_name in os.listdir(dtype_path):
            model_full_path = os.path.join(dtype_path, model_name)
            if not os.path.isdir(model_full_path): continue
            
            # 检查是否是模型目录（包含任意常见的模型配置文件）
            config_files = ["config.json", "configuration.json", "model_index.json", "hash.txt"]
            is_model = any(os.path.exists(os.path.join(model_full_path, c)) for c in config_files) or \
                       glob.glob(os.path.join(model_full_path, "*.safetensors"))
            
            if is_model:
                if split_model(model_full_path, dtype, force=args.force):
                    stats["processed"] += 1
                else:
                    stats["skipped"] += 1

    print(f"\n汇总: 处理完成 {stats['processed']} 个，跳过 {stats['skipped']} 个。")

if __name__ == "__main__":
    main()