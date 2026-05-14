import torch
import os
from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

def export_static_12hz_decoder_to_onnx(
    model_dir: str, 
    output_onnx_path: str, 
    batch_size: int = 1, 
    codes_length: int = 300
):
    print(f"Loading tokenizer from: {model_dir}")
    
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        model_dir,
        attn_implementation="eager" 
    )
    
    decoder_model = tokenizer.model.decoder
    decoder_model.eval()

    num_quantizers = decoder_model.config.num_quantizers
    
    print(f"Static input shape: (batch_size={batch_size}, num_quantizers={num_quantizers}, codes_length={codes_length})")
    
    dummy_codes = torch.randint(
        low=0,
        high=decoder_model.config.codebook_size,
        size=(batch_size, num_quantizers, codes_length),
        dtype=torch.long,
        device=tokenizer.device
    )

    print(f"Exporting static ONNX model to {output_onnx_path}...")
    
    # 4. 执行 ONNX 导出
    with torch.no_grad():
        torch.onnx.export(
            model=decoder_model,                   # 导出的模型实例
            args=(dummy_codes,),                   # 模型的输入元组
            f=output_onnx_path,                    # 导出的路径
            export_params=True,                    # 权重一并打包到 onnx 文件中
            opset_version=19,                      # 使用 19+ 版本，能很好支持 RoPE 和复杂激活函数
            do_constant_folding=True,              # 对常量进行折叠优化
            input_names=["codes"],                 # 定义输入的变量名
            output_names=["wav"],                  # 定义输出的变量名
            # 删除了 dynamic_axes 参数，ONNX 模型的所有维度将被硬编码锁定
        )

    print("Static ONNX Export completed successfully!")

if __name__ == "__main__":

    # MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign/speech_tokenizer"
    MODEL_PATH = "/home/m5stack/rsp/Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer"
    # OUTPUT_ONNX = "qwen3_tts_12hz_1.7B-VoiceDesign-decoder_static.onnx"
    OUTPUT_ONNX = "qwen3_tts_12hz_0.6B-Base-decoder_static.onnx"
    
    export_static_12hz_decoder_to_onnx(MODEL_PATH, OUTPUT_ONNX, batch_size=1, codes_length=300)