import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from webui_simple import *
import gradio as gr

import ast
import zipfile
import shutil
import numpy as np

global PEOPLE_NUM, CHAT_NUM
global now_model_path

PEOPLE_NUM = 5
CHAT_NUM = 100
default_path = os.path.join(os.path.dirname(__file__), "weights")
now_model_path = default_path
model_list = [item for item in os.listdir(default_path) if item.endswith(".bin")] + [
    "空"
]
name_textbox = []
model_dropdown = []
name_dropdown = []
text_content = []
language_dropdown = []
num_label = []


def get_model(model_path):
    now_model_path = model_path
    if model_path is None or not os.path.isdir(model_path):
        model_list = os.listdir(default_path)
    else:
        model_list = os.listdir(model_path)
    model_list = [item for item in model_list if item.endswith(".bin")]
    model_list.append("空")
    updated_models = []
    for _ in range(PEOPLE_NUM):
        updated_models.append(
            {
                "choices": model_list,
                "__type__": "update",
            }
        )
    return updated_models


def get_name(*names):
    name_list = []
    updated_names = []
    for name in names:
        if len(name) > 0:
            name_list.append(name)
    for _ in range(CHAT_NUM):
        updated_names.append(
            {
                "choices": name_list,
                "__type__": "update",
            }
        )
    return updated_names


def process_func(params):
    params = ast.literal_eval(params)
    audio_opt = {}
    for p in params:
        # 解压模型到tmp文件夹
        file_path = os.path.join(now_model_path, p["model"])
        tmp_dir = os.path.join(
            os.path.dirname(__file__),
            "TEMP",
            os.path.splitext(os.path.basename(file_path))[0],
        )
        os.makedirs(tmp_dir, exist_ok=True)
        with zipfile.ZipFile(file_path, "r") as zipf:
            zipf.extractall(path=tmp_dir)

        ref_wav_path = [
            os.path.join(tmp_dir, file)
            for file in os.listdir(tmp_dir)
            if file.endswith(".wav")
        ][0]
        prompt_language, prompt_text = os.path.splitext(os.path.basename(ref_wav_path))[
            0
        ].split("-")

        sovits_path = [
            os.path.join(tmp_dir, file)
            for file in os.listdir(tmp_dir)
            if file.endswith(".pth")
        ][0]
        gpt_path = [
            os.path.join(tmp_dir, file)
            for file in os.listdir(tmp_dir)
            if file.endswith(".ckpt")
        ][0]
        text_language = p["language"]
        # 载入模型
        change_sovits_weights(sovits_path)
        change_gpt_weights(gpt_path)

        # 循环生成audio
        for key, text in p["text"].items():
            sr, audio = next(
                get_tts_wav(
                    ref_wav_path,
                    prompt_text,
                    prompt_language,
                    text,
                    text_language,
                    top_k=20,
                    top_p=0.6,
                    temperature=0.6,
                )
            )
            audio_opt.update({key: audio})

    shutil.rmtree(os.path.join(os.path.dirname(__file__), "TEMP"))
    # 连接audio
    audio = np.zeros(int(sr * 0.5)).astype(np.int16)
    audio_quite = np.zeros(int(sr * 0.5)).astype(np.int16)
    sorted_items = sorted(audio_opt.items())
    for k, v in sorted_items:
        audio = np.concatenate((audio, v, audio_quite))
    return (sr, audio)


def check_peo_num(peo_num):
    if int(peo_num) > PEOPLE_NUM:
        return {"label": "数量超限", "__type__": "update"}
    else:
        return {"label": "", "__type__": "update"}


def check_chat_num(chat_num):
    if int(chat_num) > CHAT_NUM:
        return {"label": "数量超限", "__type__": "update"}
    else:
        return {"label": "", "__type__": "update"}


def check_params(*arg):
    start_bn = {"visible": False, "__type__": "update"}
    # 将所有数据选出来
    if len(arg) == 3 * PEOPLE_NUM + 2 * CHAT_NUM + 1:
        model_path = arg[0]
        choose_names = list(arg[1 : PEOPLE_NUM + 1])
        choose_models = list(arg[PEOPLE_NUM + 1 : 2 * PEOPLE_NUM + 1])
        choose_language = list(arg[2 * PEOPLE_NUM + 1 : 3 * PEOPLE_NUM + 1])
        chat_names = list(arg[3 * PEOPLE_NUM + 1 : 3 * PEOPLE_NUM + CHAT_NUM + 1])
        text_contents = list(arg[3 * PEOPLE_NUM + CHAT_NUM + 1 :])
    # 判断人物是否重复，模型是否重复
    eff_names = [item for item in choose_names if len(item) > 0]
    eff_models = [item for item in choose_models if len(item) > 0]
    if len(set(eff_names)) < len(eff_names) or len(set(eff_models)) < len(eff_models):
        return {"value": "人名或者模型重复，请更改", "__type__": "update"}, start_bn
    while choose_names and len(choose_names[-1]) == 0:
        choose_names.pop()
    choose_models = choose_models[0 : len(choose_names)]
    choose_language = choose_language[0 : len(choose_names)]
    # 再确定对话人,对话列表
    while chat_names and (chat_names[-1] is None or len(chat_names[-1]) == 0):
        chat_names.pop()
    for i in range(len(chat_names)):
        if len(chat_names[i]) == 0:
            return {
                "value": f"第{i+1}行没有选择人物",
                "__type__": "update",
            }, start_bn
    text_contents = [text for text in text_contents[0 : len(chat_names)]]
    for i in range(len(text_contents)):
        if len(text_contents[i]) == 0 or len(text_contents[i]) > 50:
            return {
                "value": f"第{i+1}行有人物没有写对话,或者对话字数超过50字",
                "__type__": "update",
            }, start_bn

    # 确定几人参与对话
    names = set(chat_names)
    # audio_params = [AudioParams() for _ in range(len(names))]
    audio_params = [
        dict({"name": "", "model": "", "language": "", "text": {}})
        for _ in range(len(names))
    ]
    for item in audio_params:
        item["name"] = names.pop()
        for i in range(len(choose_names)):
            if item["name"] == choose_names[i]:
                if len(choose_models[i]) > 0:
                    item["model"] = choose_models[i]
                    item["language"] = choose_language[i]
                else:
                    return {
                        "value": "模型为空",
                        "__type__": "update",
                    }, start_bn
    for i in range(len(chat_names)):
        for item in audio_params:
            if chat_names[i] == item["name"]:
                item["text"].update({i: text_contents[i]})

    start_bn = {"visible": True, "__type__": "update"}
    return [audio_params, start_bn]


def download_audio():
    pass


def check_text(text):
    if len(text) > 50:
        return {
            "value": text,
            "label": "文本长度大于50，请分成多段",
            "__type__": "update",
        }
    else:
        return {
            "value": text,
            "label": "",
            "__type__": "update",
        }


def check_name_textbox_repeat(*names):
    n = len(names)
    updated_names = [
        {
            "value": item,
            "label": "",
            "__type__": "update",
        }
        for item in names
    ]
    for i in range(n):
        for j in range(n):
            if i != j and len(names[i]) > 0 and names[i] == names[j]:
                updated_names[i] = {
                    "value": names[i],
                    "label": "人物名重复，请修改",
                    "__type__": "update",
                }
                break
    return updated_names


def check_model_repeat(*md):
    n = len(md)
    updated_md = [
        {
            "label": "",
            "__type__": "update",
        }
        for _ in range(n)
    ]
    for i in range(n):
        for j in range(n):
            if i != j and len(md[i]) > 0 and md[i] == md[j]:
                updated_md[i] = {
                    "label": "模型重复，请修改",
                    "__type__": "update",
                }
                break
    return updated_md


def check_name_dropdown(text, name):
    if len(text) > 0 and len(name) == 0:
        return {"label": "请选择人物", "__type__": "update"}
    else:
        return {"label": "", "__type__": "update", "value": name}


def cancel_name_label(name):
    if name is not None or len(name) > 0:
        return {"label": "", "__type__": "update", "value": name}


def show_gr(peo_num, chat_num):
    choose_names = [{"visible": False, "__type__": "update"} for _ in range(PEOPLE_NUM)]
    choose_models = [
        {"visible": False, "__type__": "update"} for _ in range(PEOPLE_NUM)
    ]
    languages = [{"visible": False, "__type__": "update"} for _ in range(PEOPLE_NUM)]
    chat_names = [{"visible": False, "__type__": "update"} for _ in range(CHAT_NUM)]
    text_contents = [{"visible": False, "__type__": "update"} for _ in range(CHAT_NUM)]
    num_label = [{"visible": False, "__type__": "update"} for _ in range(CHAT_NUM)]
    if int(peo_num) > PEOPLE_NUM or int(chat_num) > CHAT_NUM:
        return (
            choose_names
            + choose_models
            + languages
            + num_label
            + chat_names
            + text_contents
        )
    for i in range(int(peo_num)):
        choose_names[i] = {"visible": True, "__type__": "update"}
        choose_models[i] = {"visible": True, "__type__": "update"}
        languages[i] = {"visible": True, "__type__": "update"}
    for i in range(int(chat_num)):
        num_label[i] = {"visible": True, "__type__": "update"}
        chat_names[i] = {"visible": True, "__type__": "update"}
        text_contents[i] = {"visible": True, "__type__": "update"}

    return (
        choose_names
        + choose_models
        + languages
        + num_label
        + chat_names
        + text_contents
    )


with gr.Blocks(title="多人对话GPT-SoVITS") as app:
    gr.Markdown(value="<h1>感谢“花儿不哭”大佬的无私开源！</h1>")
    gr.Markdown(
        value="<h2>第一步：确认对话人数，对话数，模型路径，然后点击旁边确认按钮</h2>"
    )
    gr.Markdown(value="<h2>人数最多支持5人，对话数最多支持100条，超过限定则无效</h2>")
    with gr.Row():
        peo_num_bn = gr.Textbox(
            label="人数", scale=1, value=1, interactive=True, max_lines=1
        )
        chat_num_bn = gr.Textbox(
            label="对话数", scale=1, value=1, interactive=True, max_lines=1
        )
        model_path = gr.Textbox(
            label="模型路径",
            max_lines=1,
            placeholder="输入模型路径，不要带引号",
            value=os.path.join(os.path.dirname(__file__), "weights"),
        )
        confire_bn = gr.Button("确认", variant="primary")
    gr.Markdown(value="第二步：输入人名，选择模型。人名与模型不能重复")
    with gr.Row():
        for i in range(PEOPLE_NUM):
            with gr.Group():
                with gr.Column():
                    name_textbox.append(
                        gr.Textbox(
                            label="名称",
                            interactive=True,
                            placeholder="请输入人名",
                            visible=False,
                        )
                    )
                    model_dropdown.append(
                        gr.Dropdown(
                            label="模型",
                            choices=model_list,
                            interactive=True,
                            visible=False,
                        )
                    )
                    language_dropdown.append(
                        gr.Dropdown(
                            scale=1,
                            value=i18n("中文"),
                            label="",
                            choices=[
                                i18n("中文"),
                                i18n("英文"),
                                i18n("日文"),
                                i18n("中英混合"),
                                i18n("日英混合"),
                                i18n("多语种混合"),
                            ],
                            interactive=True,
                            visible=False,
                        )
                    )
    # 刷新模型列表
    model_path.blur(get_model, inputs=model_path, outputs=model_dropdown)
    gr.Markdown(value="<h2>第三步：选择人物，输入对话内容，选择语言<h2>")

    # with gr.Group():
    with gr.Group():
        for i in range(CHAT_NUM):
            with gr.Row():
                num_label.append(gr.Textbox(value=i + 1, scale=1, visible=False))
                name_dropdown.append(
                    gr.Dropdown(
                        scale=2, label="", choices=[], interactive=True, visible=False
                    )
                )
                text_content.append(
                    gr.Textbox(scale=10, label="", max_lines=1, visible=False)
                )
    peo_num_bn.change(check_peo_num, inputs=peo_num_bn, outputs=peo_num_bn)
    chat_num_bn.change(check_chat_num, inputs=chat_num_bn, outputs=chat_num_bn)
    confire_bn.click(
        show_gr,
        inputs=[peo_num_bn, chat_num_bn],
        outputs=name_textbox
        + model_dropdown
        + language_dropdown
        + num_label
        + name_dropdown
        + text_content,
    )
    for nt in name_textbox:
        # 检查人物名称十分重复
        nt.blur(
            check_name_textbox_repeat,
            inputs=[name for name in name_textbox],
            outputs=name_textbox,
        )
        # 刷新人物列表
        nt.blur(get_name, inputs=[name for name in name_textbox], outputs=name_dropdown)
    for md in model_dropdown:
        md.change(
            check_model_repeat,
            inputs=[i for i in model_dropdown],
            outputs=model_dropdown,
        )
    # 检查文本长度,检查人物选择
    for i in range(len(text_content)):
        text_content[i].blur(
            check_text, inputs=text_content[i], outputs=text_content[i]
        )
        text_content[i].blur(
            check_name_dropdown,
            inputs=[text_content[i], name_dropdown[i]],
            outputs=name_dropdown[i],
        )
        name_dropdown[i].change(
            cancel_name_label, inputs=name_dropdown[i], outputs=name_dropdown[i]
        )
    params_bn = gr.Button("参数确认", variant="primary")
    params_box = gr.Textbox(label="最终参数", interactive=False)
    start_bn = gr.Button("点击生成", visible=False, variant="primary")
    params_bn.click(
        check_params,
        inputs=[model_path]
        + name_textbox
        + model_dropdown
        + language_dropdown
        + name_dropdown
        + text_content,
        outputs=[params_box, start_bn],
    )
    gr.Markdown("<h2>生成效果,播放框右上角有下载按钮<h2>")
    audio = gr.Audio()
    start_bn.click(
        process_func,
        inputs=params_box,
        outputs=audio,
    )


app.queue().launch(
    server_name="0.0.0.0",
    inbrowser=True,
    share=False,
    server_port=3345,
    quiet=True,
)
