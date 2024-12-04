import subprocess

left_models = ['left_model/left_tcn_hit.tflite', 'left_model/left_tcn_fbou.tflite', 'left_model/left_tcn_kill.tflite']
right_models = ['right_model/right_tcn_hit.tflite', 'right_model/right_tcn_fbou.tflite',
                'right_model/right_tcn_kill.tflite']

models = left_models + right_models
for model in models:

    # 定义命令
    command = [
        "/home/sinnus/workspace/mindspore/mindspore-lite-2.0.0-linux-x64/tools/converter/converter/converter_lite",
        "--fmk=TFLITE",
        "--modelFile=" + model,
        "--outputFile=" + model[:-6] + "ms"
    ]

    # 执行命令并捕获输出
    try:
        print(' '.join(command))
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Command executed successfully:")
        print(result.stdout)  # 打印命令的标准输出
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        print(f"stderr: {e.stderr}")

