export QWEN3_TTS_DUMP_DIR=./debug_bin/py_dump

python3 scripts/infer.py \
    --qwen_tts_root ~/Workspace/Qwen3-TTS \
    --hf_model_path ~/rsp/Qwen3-TTS-12Hz-0.6B-Base/ \
    --talker_compiled_model_path ~/rsp/Qwen3-TTS-12Hz-0.6B-Base-AX650/talker \
    --code_predictor_compiled_model_path ~/rsp/Qwen3-TTS-12Hz-0.6B-Base-AX650/code-predictor \
    --talker_axengine_device "0" \
    --code_predictor_axengine_device "1" \
    --ref_audio likes.wav \
    --ref_text "可莉喜欢毛茸茸的东西。比如嘟嘟可、蒲公英，还有雷泽的头发。" \
    --text "为了守护蒙德城周边的安定，我曾经发动过不少次「远征」，但比起这一次，都算不上什么…比如清剿达达乌帕谷、联合千岩军扫荡石门、从鹰翔海滩出发迎击外海魔物…嗯？你说难怪在这些地方都遇不到什么强敌…我应该还是留了些下来给人练手的吧？" \
    --no-do_sample \
    --no-subtalker_dosample \
    --dump_cpp_input_dir ./debug_bin \
    --skip_wav_generation \
    --dump_output_codes_dir ./scripts \
    --non_streaming_mode