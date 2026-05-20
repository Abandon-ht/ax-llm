# python 与 cpp 调试指南

1. python 使用 shell 脚本生成 debug_bin dump 目录

```shell
conda activate qwen3-tts # 注意，先激活 python 虚拟环境。
./scripts/infer.sh # 默认 dump 路径为脚本指定，不要重定向到 /tmp 等目录
```

./scripts/infer.py 的代码通过此脚本传递参数和执行，执行过程较长，建议后台执行并重定向日志输出。

2. cpp 使用 build_ax650.sh 编译

调试代码为 tools/qwen3_tts_infer.cpp src/runner/LLM_cp_tts_insert.inc 等文件。

axmodel 在 /home/m5stack/rsp/Qwen3-TTS-12Hz-0.6B-Base-AX650 目录下，其中模型配置文件在 talker/config.json 采样配置文件在 talker/post_config.json

```shell
./build_ax650.sh
```

3. cpp 使用 scp 上传编译好的程序到 pyramid 主机，详见 ~/.ssh/config ，上传之前建议先删除旧的程序。避免因程序未覆盖影响调试分析，必要可使用 md5sum 等工具做哈希校验。

```shell
ssh pyramid
rm -rf build
```

```shell
scp -r build/install pyramid:build
```

4. cpp 使用 python dump 的文件调试，需要同步上传，详见 debug_bin 目录

```shell
scp -r debug_bin pyramid:
```

5. 使用 ssh pyramid 连接到远程主机执行。

```shell
ssh pyramid
./mount.sh # 初次执行需要挂载主机的磁盘，确保 ~/rsp 与 pyramid 主机上的 /root/rsp 是同一目录。
```

```shell
./build/bin/qwen3_tts_infer rsp/Qwen3-TTS-12Hz-0.6B-Base-AX650/talker/ debug_bin/ --max_new_tokens 300 --seed 1234 # 注意，这一步是在 pyramid 主机上执行，要选择适当的 max_new_tokens 长度，太长可能会导致执行过慢。该任务需要 30s 以上才能执行完。建议后台执行并重定向日志输出。
```

6. 使用 pyrmaid 的 python 虚拟环境

```shell
ssh pyramid
source qwen3-tts/bin/activate # 此环境已经预装了 axengine qwen-tts 等依赖包
```

7. 将 cpp 的输出结果传回主机

```shell
scp -r pyramid:debug_bin . # 注意传回的 cpp dump 文件路径，可以重命名
```

8. 在 scripts 目录下创建 python 分析脚本

```shell
conda activate qwen3-tts
python scripts/compare_cp_dumps.py # 注意此脚本只是演示，后续需要根据需求自行创建。
```

9. 一定要记录调试的关键步骤，并总结输出到 docs 目录下的文档。

10. output_codes.bin output_meta.json 等文件为 scripts/infer_bin.py 输入，最终 decode 出音频。这一步暂不需要执行，同样的，infer.sh 也传递了 --skip_wav_generation 跳过 decode 。只要确保 talker + cp 输出的 token 是正确的（注意，受采样影响，无法一致）。我人为试听判断生成的音频。（不要试图调用 ASR 工具识别 wav ）

11. 必要时可以去  ~/Qwen3-TTS/examples/test_model_12hz_base_single_batch.py 执行相关代码，推理 huggingface 的模型。注意同样需要使用 conda activate qwen3-tts 虚拟环境。