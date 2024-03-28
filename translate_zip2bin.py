import os
import shutil
import subprocess
import zipfile

default_path = os.path.join(os.path.dirname(__file__), "weights")


def translate():
    global default_path
    file_list = [item for item in os.listdir(default_path) if item.endswith(".zip")]
    file_list = [os.path.join(default_path, item) for item in file_list]
    for file_path in file_list:
        zip_file = os.path.splitext(os.path.basename(file_path))[0] + ".bin"
        # 已经有了文件就不再转化了
        if zip_file not in os.listdir(default_path):
            # 解压到临时目录
            tmp_dir = os.path.join(
                os.path.dirname(__file__),
                "temp",
                os.path.splitext(os.path.basename(file_path))[0],
            )
            os.makedirs(tmp_dir, exist_ok=True)
            powershell_command = (
                f"Expand-Archive -Path '{file_path}' -DestinationPath '{tmp_dir}'"
            )
            subprocess.run(["powershell", "-Command", powershell_command])
            # 判断是否是，若不是，则删除临时目录返回
            audio_dir = os.path.join(tmp_dir, "参考音频")
            audio_fn = (
                os.listdir(audio_dir)[0] if len(os.listdir(audio_dir)) > 0 else ""
            )
            if audio_fn.endswith(".wav"):
                tmp = audio_fn.split("-")
                tmp[0] = "中文"
                tmp = "-".join(tmp)
                os.rename(
                    os.path.join(audio_dir, audio_fn), os.path.join(audio_dir, tmp)
                )
                audio_fn = tmp
            shutil.move(os.path.join(audio_dir, audio_fn), tmp_dir)

            # 将三个文件打包压缩
            fn_list = []
            for fn in os.listdir(tmp_dir):
                if os.path.isfile(os.path.join(tmp_dir, fn)):
                    if fn.endswith((".pth", ".ckpt", ".wav")):
                        fn_list.append(fn)
            os.chdir(tmp_dir)
            with zipfile.ZipFile(zip_file, "w") as zf:
                for fn in fn_list:
                    zf.write(fn)
            if zip_file not in os.listdir(default_path):
                shutil.move(zip_file, default_path)
                os.chdir(os.path.dirname(__file__))
            shutil.rmtree(tmp_dir)


translate()
