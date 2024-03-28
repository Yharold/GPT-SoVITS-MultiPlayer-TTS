"""
按中英混合识别
按日英混合识别
多语种启动切分识别语种
全部按中文识别
全部按英文识别
全部按日文识别
"""

import os, re, logging
import shutil
import subprocess
import time
import zipfile
import LangSegment

logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
import torch


cnhubert_base_path = os.environ.get(
    "cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base"
)
bert_path = os.environ.get(
    "bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
)
is_share = os.environ.get("is_share", "False")
is_share = eval(is_share)
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
import gradio as gr
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import librosa
from feature_extractor import cnhubert

cnhubert.cnhubert_base_path = cnhubert_base_path

from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from time import time as ttime
from module.mel_processing import spectrogram_torch
from my_utils import load_audio
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half == True:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)


def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


ssl_model = cnhubert.get_model()
if is_half == True:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)


def change_sovits_weights(sovits_path):
    global vq_model, hps
    dict_s2 = torch.load(sovits_path, map_location="cpu")
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )
    if "pretrained" not in sovits_path:
        del vq_model.enc_q
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    print(vq_model.load_state_dict(dict_s2["weight"], strict=False))


def change_gpt_weights(gpt_path):
    global hz, max_sec, t2s_model, config
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half == True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    total = sum([param.nelement() for param in t2s_model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))


def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec


dict_language = {
    i18n("中文"): "all_zh",  # 全部按中文识别
    i18n("英文"): "en",  # 全部按英文识别#######不变
    i18n("日文"): "all_ja",  # 全部按日文识别
    i18n("中英混合"): "zh",  # 按中英混合识别####不变
    i18n("日英混合"): "ja",  # 按日英混合识别####不变
    i18n("多语种混合"): "auto",  # 多语种启动切分识别语种
}


def clean_text_inf(text, language):
    phones, word2ph, norm_text = clean_text(text, language)
    phones = cleaned_text_to_sequence(phones)
    return phones, word2ph, norm_text


dtype = torch.float16 if is_half == True else torch.float32


def get_bert_inf(phones, word2ph, norm_text, language):
    language = language.replace("all_", "")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)  # .to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    return bert


splits = {
    "，",
    "。",
    "？",
    "！",
    ",",
    ".",
    "?",
    "!",
    "~",
    ":",
    "：",
    "—",
    "…",
}


def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text


def get_phones_and_bert(text, language):
    if language in {"en", "all_zh", "all_ja"}:
        language = language.replace("all_", "")
        if language == "en":
            LangSegment.setLangfilters(["en"])
            formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
        else:
            # 因无法区别中日文汉字,以用户输入为准
            formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        phones, word2ph, norm_text = clean_text_inf(formattext, language)
        if language == "zh":
            bert = get_bert_feature(norm_text, word2ph).to(device)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if is_half == True else torch.float32,
            ).to(device)
    elif language in {"zh", "ja", "auto"}:
        textlist = []
        langlist = []
        LangSegment.setLangfilters(["zh", "ja", "en", "ko"])
        if language == "auto":
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "ko":
                    langlist.append("zh")
                    textlist.append(tmp["text"])
                else:
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
        else:
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    # 因无法区别中日文汉字,以用户输入为准
                    langlist.append(language)
                textlist.append(tmp["text"])
        print(textlist)
        print(langlist)
        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
            bert = get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = "".join(norm_text_list)

    return phones, bert.to(dtype), norm_text


def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if len(text) > 0:
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result


def get_tts_wav(
    ref_wav_path,
    prompt_text,
    prompt_language,
    text,
    text_language,
    top_k=20,
    top_p=0.6,
    temperature=0.6,
    ref_free=False,
):
    if prompt_text is None or len(prompt_text) == 0:
        ref_free = True
    t0 = ttime()
    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]
    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if prompt_text[-1] not in splits:
            prompt_text += "。" if prompt_language != "en" else "."
        print(i18n("实际输入的参考文本:"), prompt_text)
    text = text.strip("\n")
    if text[0] not in splits and len(get_first(text)) < 4:
        text = "。" + text if text_language != "en" else "." + text

    print(i18n("实际输入的目标文本:"), text)
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half == True else np.float32,
    )
    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
            raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)
        if is_half == True:
            wav16k = wav16k.half().to(device)
            zero_wav_torch = zero_wav_torch.half().to(device)
        else:
            wav16k = wav16k.to(device)
            zero_wav_torch = zero_wav_torch.to(device)
        wav16k = torch.cat([wav16k, zero_wav_torch])
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
            "last_hidden_state"
        ].transpose(
            1, 2
        )  # .float()
        codes = vq_model.extract_latent(ssl_content)

        prompt_semantic = codes[0, 0]
    t1 = ttime()

    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    print(i18n("实际输入的目标文本(切句后):"), text)
    texts = text.split("\n")
    texts = merge_short_text_in_array(texts, 5)
    audio_opt = []
    if not ref_free:
        phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language)

    for text in texts:
        # 解决输入目标文本的空行导致报错的问题
        if len(text.strip()) == 0:
            continue
        if text[-1] not in splits:
            text += "。" if text_language != "en" else "."
        print(i18n("实际输入的目标文本(每句):"), text)
        phones2, bert2, norm_text2 = get_phones_and_bert(text, text_language)
        print(i18n("前端处理后的文本(每句):"), norm_text2)
        if not ref_free:
            bert = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = (
                torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
            )
        else:
            bert = bert2
            all_phoneme_ids = torch.LongTensor(phones2).to(device).unsqueeze(0)

        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
        prompt = prompt_semantic.unsqueeze(0).to(device)
        t2 = ttime()
        with torch.no_grad():
            # pred_semantic = t2s_model.model.infer(
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                None if ref_free else prompt,
                bert,
                # prompt_phone_len=ph_offset,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                early_stop_num=hz * max_sec,
            )
        t3 = ttime()
        # print(pred_semantic.shape,idx)
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(
            0
        )  # .unsqueeze(0)#mq要多unsqueeze一次
        refer = get_spepc(hps, ref_wav_path)  # .to(device)
        if is_half == True:
            refer = refer.half().to(device)
        else:
            refer = refer.to(device)
        # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
        audio = (
            vq_model.decode(
                pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refer
            )
            .detach()
            .cpu()
            .numpy()[0, 0]
        )  ###试试重建不带上prompt部分
        max_audio = np.abs(audio).max()  # 简单防止16bit爆音
        if max_audio > 1:
            audio /= max_audio
        audio_opt.append(audio)
        audio_opt.append(zero_wav)
        t4 = ttime()
    print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
    yield hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(
        np.int16
    )


def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


default_path = os.path.join(os.path.dirname(__file__), "weights")
num_people = 5
num_chat = 100
names = []
models = []
language_list = [
    "中文",
    "英文",
    "日文",
    "中英混合",
    "日英混合",
    "多语种混合",
]
languages = []
chat_num = []
chat_names = []
chat_contents = []
model_list = [item for item in os.listdir(default_path) if item.endswith(".bin")] + [
    "空"
]
params_info = ""


# 添加删除有问题,号要加上限，不能超过num_people。。
# 参数传入是个tuple，不是list

people_no = 0
chat_no = 0


def add_20_chat(*args):
    global chat_no, num_chat
    chat_names_c = args[0:num_chat]
    chat_contents_c = args[num_chat:]
    if chat_no < num_chat:
        chat_no = chat_no + 20 if chat_no + 20 < num_chat else num_chat
        r1 = [
            {"value": i + 1, "visible": False, "__type__": "update"}
            for i in range(num_chat)
        ]
        r2 = [
            {"value": item, "visible": False, "__type__": "update"}
            for item in chat_names_c
        ]
        r3 = [
            {"value": item, "visible": False, "__type__": "update"}
            for item in chat_contents_c
        ]
        for i in range(num_chat):
            if i < chat_no:
                r1[i]["visible"] = True
                r2[i]["visible"] = True
                r3[i]["visible"] = True
            else:
                r1[i]["visible"] = False
                r2[i]["visible"] = False
                r3[i]["visible"] = False
        return r1 + r2 + r3
    else:
        r1 = [
            {"value": i + 1, "visible": True, "__type__": "update"}
            for i in range(num_chat)
        ]
        return r1 + list(args)


def add_row(*args):
    global people_no, chat_no, num_people, num_chat
    if len(args) == 3 * num_people:
        r1 = args[0:num_people]
        r2 = args[num_people : 2 * num_people]
        r3 = args[2 * num_people : 3 * num_people]
        if people_no >= num_people:
            return r1 + r2 + r3
        else:
            people_no += 1
            r1 = [
                {"value": item, "visible": False, "__type__": "update"} for item in r1
            ]
            r2 = [
                {"value": item, "visible": False, "__type__": "update"} for item in r2
            ]
            r3 = [
                {"value": item, "visible": False, "__type__": "update"} for item in r3
            ]
            for i in range(num_people):
                if i < people_no:
                    r1[i]["visible"] = True
                    r2[i]["visible"] = True
                    r3[i]["visible"] = True
                else:
                    r1[i]["visible"] = False
                    r2[i]["visible"] = False
                    r3[i]["visible"] = False
            return r1 + r2 + r3
    if len(args) == 3 * num_chat:
        r1 = args[0:num_chat]
        r2 = args[num_chat : 2 * num_chat]
        r3 = args[2 * num_chat : 3 * num_chat]
        if chat_no >= num_chat:
            return r1 + r2 + r3
        else:
            chat_no += 1
            r1 = [
                {"value": item, "visible": False, "__type__": "update"} for item in r1
            ]
            r2 = [
                {"value": item, "visible": False, "__type__": "update"} for item in r2
            ]
            r3 = [
                {"value": item, "visible": False, "__type__": "update"} for item in r3
            ]
            for i in range(num_chat):
                if i < chat_no:
                    r1[i]["visible"] = True
                    r2[i]["visible"] = True
                    r3[i]["visible"] = True

                else:
                    r1[i]["visible"] = False
                    r2[i]["visible"] = False
                    r3[i]["visible"] = False

            return r1 + r2 + r3


def del_row(*args):
    global people_no, chat_no, num_people, num_chat
    if len(args) == 3 * num_people:
        r1 = args[0:num_people]
        r2 = args[num_people : 2 * num_people]
        r3 = args[2 * num_people : 3 * num_people]

        if people_no <= 0:
            return r1 + r2 + r3
        else:
            people_no -= 1
            r1 = [
                {"value": item, "visible": False, "__type__": "update"} for item in r1
            ]
            r2 = [
                {"value": item, "visible": False, "__type__": "update"} for item in r2
            ]
            r3 = [
                {"value": item, "visible": False, "__type__": "update"} for item in r3
            ]
            for i in range(num_people):
                if i < people_no:
                    r1[i]["visible"] = True
                    r2[i]["visible"] = True
                    r3[i]["visible"] = True
                else:
                    r1[i]["visible"] = False
                    r2[i]["visible"] = False
                    r3[i]["visible"] = False
                    r1[i]["value"] = ""
                    r2[i]["value"] = ""
                    r3[i]["value"] = ""
            return r1 + r2 + r3
    if len(args) == 3 * num_chat:
        r1 = args[0:num_chat]
        r2 = args[num_chat : 2 * num_chat]
        r3 = args[2 * num_chat : 3 * num_chat]
        if chat_no <= 0:
            return r1 + r2 + r3
        else:
            chat_no -= 1
            r1 = [
                {"value": item, "visible": False, "__type__": "update"} for item in r1
            ]
            r2 = [
                {"value": item, "visible": False, "__type__": "update"} for item in r2
            ]
            r3 = [
                {"value": item, "visible": False, "__type__": "update"} for item in r3
            ]
            for i in range(num_chat):
                if i < chat_no:
                    r1[i]["visible"] = True
                    r2[i]["visible"] = True
                    r3[i]["visible"] = True

                else:
                    r1[i]["visible"] = False
                    r2[i]["visible"] = False
                    r3[i]["visible"] = False
                    r2[i]["value"] = ""
                    r3[i]["value"] = ""
            return r1 + r2 + r3


# 转化白菜工厂文件
def translate_zip2bin():
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


def refresh_model_list(model_path):
    global model_list, num_people

    if not os.path.exists(model_path):
        r = [
            {"value": "", "choices": [], "__type__": "update"}
            for _ in range(num_people)
        ]
        return [{"label": "文件夹不存在", "__type__": "update"}] + r
    model_list = [item for item in os.listdir(model_path) if item.endswith(".bin")]
    model_list = model_list + ["空"]
    r = [{"choices": model_list, "__type__": "update"} for _ in range(num_people)]
    return [{"label": "", "__type__": "update"}] + r


def check_people_info(*args):
    global num_people
    names_c = args[0:num_people]
    models_c = args[num_people : 2 * num_people]
    languages_c = args[2 * num_people :]
    r1 = [{"value": item, "label": "", "__type__": "update"} for item in names_c]
    r2 = [{"value": item, "label": "", "__type__": "update"} for item in models_c]
    r3 = [{"value": item, "label": "", "__type__": "update"} for item in languages_c]
    # 确认检查范围，以人名为准
    n = num_people
    while len(names_c[n - 1]) == 0:
        if n > 0:
            n -= 1
        else:
            r1[0]["label"] = "人名为空"
            r2[0]["label"] = "人名为空"
            r3[0]["label"] = "人名为空"
            return r1 + r2 + r3
    # 检查人名，模型，语言是否为空
    for i in range(n):
        if len(names_c[i]) == 0:
            r1[i]["label"] = "人名为空"
        if len(models_c[i]) == 0 or models_c[i] == "空":
            r2[i]["label"] = "模型为空"
        if len(languages_c[i]) == 0:
            r3[i]["label"] = "语言为空"
    # 检查人名，模型是否重复
    for i in range(n):
        for j in range(n):
            if i != j:
                if names_c[i] == names_c[j]:
                    r1[i]["label"] = "人名重复"
                if models_c[i] == models_c[j]:
                    r2[i]["label"] = "模型重复"
    return r1 + r2 + r3


def refresh_name_list(*args):
    global num_chat, num_people
    names_c = args[0:num_people]
    chat_names_c = args[num_people:]

    # 拿到所有不重复不为空的人名
    name_list = list(set([item for item in names_c if len(item) > 0]))
    r = [
        {"choices": name_list, "label": "", "__type__": "update"}
        for _ in range(num_chat)
    ]
    # 确定判断范围
    n = num_chat
    while len(chat_names_c[n - 1]) == 0:
        if n > 0:
            n -= 1
        else:
            r[0]["label"] = "人名为空"
            return r
    # 检查当前名称是否在列表中
    for i in range(n):
        if chat_names_c[i] not in name_list:
            r[i]["label"] = "人名不在列表中"
        else:
            r[i]["label"] = ""
    return r


def check_chat(name, content):
    r1 = {"value": name, "label": "", "__type__": "update"}
    r2 = {"value": content, "label": "", "__type__": "update"}
    if len(name) == 0:
        r1["label"] = "人名为空"
    if len(name) != 0 and len(content) == 0:
        r2["label"] = "内容为空"
    if len(name) != 0 and len(content) > 50:
        r2["label"] = "长度大于50"
    return r1, r2


def check_params(*args):
    time.sleep(0.5)
    global params, num_chat, num_people
    r_info = ""
    r_bn = {"visible": False, "__type__": "update"}
    # 取出数据
    names_c = args[0:num_people]
    models_c = args[num_people : 2 * num_people]
    languages_c = args[2 * num_people : 3 * num_people]
    chat_names_c = args[3 * num_people : 3 * num_people + num_chat]
    chat_contents_c = args[3 * num_people + num_chat :]
    # 判断人名，模型，语言是否为空，以人名为准
    np = num_people
    while len(names_c[np - 1]) == 0:
        if np > 0:
            np -= 1
        else:
            r_info = "所有人名都为空"
            return r_info, r_bn
    for i in range(np):
        if len(names_c[i]) == 0 or len(models_c[i]) == 0 or len(languages_c[i]) == 0:
            r_info = f"第{i+1}个名称/模型/语言为空"
            return r_info, r_bn
    # 检查模型是否正确，解压模型
    pass
    # 确定聊天数据量，以人名为准
    nc = num_chat
    while len(chat_names_c[nc - 1]) == 0:
        if nc > 0:
            nc -= 1
        else:
            r_info = "所有人名都为空"
            return r_info, r_bn

    # 判断聊天人物是否在输入的人名中,人名，内容是否合适，确定几人参与聊天
    name_list = list(set(names_c[0:np]))
    idx = set()
    for i in range(nc):
        # 判断聊天人物是否在输入的人名中
        if chat_names_c[i] not in name_list:
            idx.add(i + 1)
        if len(chat_contents_c[i]) == 0 or len(chat_contents_c[i]) > 50:
            idx.add(i + 1)
    if len(idx) != 0:
        r_info = f"对话第{idx}行错误：人名不在列表/人名为空/内容为空/内容长的大于50"
        return r_info, r_bn
    # 确定参数
    params = [
        {"name": item, "model": "", "language": "", "text": {}}
        for item in list(set(chat_names_c[0:nc]))
    ]
    for item in params:
        for i in range(np):
            if item["name"] == names_c[i]:
                item["model"] = models_c[i]
                item["language"] = languages_c[i]
    for item in params:
        for i in range(nc):
            if item["name"] == chat_names_c[i]:
                item["text"].update({i: chat_contents_c[i]})
    r_info = "参数正确！"
    r_bn["visible"] = True
    return r_info, r_bn


def set_invisible():
    return {"visible": False, "__type__": "update"}


def generate_audio(model_path):
    global params
    audio_opt = {}
    for p in params:
        # 解压模型到tmp文件夹
        file_path = os.path.join(model_path, p["model"])
        tmp_dir = os.path.join(
            os.path.dirname(__file__),
            "temp",
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

    shutil.rmtree(os.path.join(os.path.dirname(__file__), "temp"))
    # 连接audio
    audio = np.zeros(int(sr * 0.5)).astype(np.int16)
    audio_quite = np.zeros(int(sr * 0.5)).astype(np.int16)
    sorted_items = sorted(audio_opt.items())
    for k, v in sorted_items:
        audio = np.concatenate((audio, v, audio_quite))
    r_bn = {"visible": False, "__type__": "update"}
    return [r_bn, (sr, audio)]


with gr.Blocks(title="多人对话TTS") as app:
    gr.Markdown("<h1>感谢“花儿不哭”大佬的无私开源</h1>")
    gr.Markdown("<h4>大佬开源地址：https://github.com/RVC-Boss/GPT-SoVITS<h6>")

    gr.Markdown(
        "<h4>模型来源于“白菜工厂”，十分感谢白菜工厂的无私奉献。地址：https://huggingface.co/baicai1145/GPT-SoVITS-STAR</h6>"
    )
    gr.Markdown(
        "<h4>此界面主要目标就是简化操作，因此模型参数都是默认参数无法调整。白菜工厂模型也无法直接使用，但提供了转换方法。</h6>"
    )
    gr.Markdown(
        "<h4>只需要将白菜工厂下载的zip文件放到一体包中weights文件夹下，然后点击转换按钮即可。</h6>"
    )
    gr.Markdown("<hr/>")

    gr.Markdown("<h2>第一步：输入模型文件夹地址</h2>")
    gr.Markdown("注意文件夹地址不带引号</h2>")
    with gr.Row():
        model_path = gr.Textbox(
            label="模型文件夹",
            scale=4,
            interactive=True,
            value=default_path,
            max_lines=1,
        )
        t_bn = gr.Button("转换", variant="primary")
        t_bn.click(translate_zip2bin, [], [])
    gr.Markdown(
        "<h2>第二步:点击按钮添加人物信息，上限5人。每行依次是人名，模型，语言</h2>"
    )
    with gr.Group():
        with gr.Column():
            for i in range(num_people):
                with gr.Row():
                    names.append(
                        gr.Textbox(
                            label="",
                            scale=1,
                            visible=False,
                            max_lines=1,
                            interactive=True,
                        )
                    )
                    models.append(
                        gr.Dropdown(
                            label="",
                            scale=1,
                            choices=model_list,
                            visible=False,
                            interactive=True,
                        )
                    )
                    languages.append(
                        gr.Dropdown(
                            label="",
                            scale=1,
                            choices=language_list,
                            value=language_list[0],
                            visible=False,
                            interactive=True,
                        )
                    )
    with gr.Row():
        add_people_bn = gr.Button("添加人物", variant="primary")
        del_people_bn = gr.Button("删除人物", variant="primary")

    gr.Markdown("<h2>第三步: 填写对话内容，上限100条，对话字数不能超过50</h2>")

    with gr.Group():
        with gr.Column():
            for i in range(num_chat):
                with gr.Row():
                    chat_num.append(
                        gr.Textbox(
                            scale=1,
                            label="",
                            value=i + 1,
                            visible=False,
                            min_width=50,
                            max_lines=1,
                            interactive=False,
                        )
                    )
                    chat_names.append(
                        gr.Dropdown(
                            scale=2,
                            label="",
                            choices=[],
                            visible=False,
                        )
                    )
                    chat_contents.append(
                        gr.Textbox(
                            scale=10,
                            label="",
                            visible=False,
                            max_lines=1,
                        )
                    )
    with gr.Row():
        add_chat_bn = gr.Button("添加", variant="primary")
        del_chat_bn = gr.Button("删除", variant="primary")
        add_20 = gr.Button("添加20条对话", variant="primary")
    gr.Markdown(
        "<h2>第四步: 检查对话信息。每次更改信息都要再次点击确认，否则数据还是未改变的数据"
    )
    confirm_bn = gr.Button("确认", variant="primary")
    info = gr.Textbox(label="", value=params_info, interactive=False)
    gr.Markdown(
        "<h2>第五步: 开始生成。生成按钮默认隐藏，只有点击确认按钮才会出现。</h2>"
    )
    ok_bn = gr.Button("开始生成", variant="primary", visible=False)
    gr.Markdown("第六步: 生成效果。预览框右上角有下载按钮")
    audio = gr.Audio(label="", interactive=False)
    # 每点一次添加20条对话
    add_20.click(
        add_20_chat,
        inputs=chat_names + chat_contents,
        outputs=chat_num + chat_names + chat_contents,
    )

    # 检查文件夹路径是否存在，存在就刷新model_list(需要 “” 空字符串)和models，若不存在，则后面组件都设为不可见，改变后，确认按钮不可见
    model_path.blur(
        refresh_model_list, inputs=model_path, outputs=[model_path] + models
    )
    # 添加 删除 人物按钮
    add_people_bn.click(
        add_row,
        inputs=names + models + languages,
        outputs=names + models + languages,
    )
    del_people_bn.click(
        del_row,
        inputs=names + models + languages,
        outputs=names + models + languages,
    )
    # 添加 删除 对话按钮
    add_chat_bn.click(
        add_row,
        inputs=chat_num + chat_names + chat_contents,
        outputs=chat_num + chat_names + chat_contents,
    )
    del_chat_bn.click(
        del_row,
        inputs=chat_num + chat_names + chat_contents,
        outputs=chat_num + chat_names + chat_contents,
    )

    # 要判断人名，模型，语言是否为空或重复
    for i in range(num_people):
        names[i].blur(
            check_people_info,
            inputs=names + models + languages,
            outputs=names + models + languages,
        )
        models[i].change(
            check_people_info,
            inputs=names + models + languages,
            outputs=names + models + languages,
        )
        languages[i].change(
            check_people_info,
            inputs=names + models + languages,
            outputs=names + models + languages,
        )
    # 刷新聊天框中人名列表,检查当前人名是否在输入人名中
    for item in names:
        item.blur(refresh_name_list, inputs=names + chat_names, outputs=chat_names)

    # 判断聊天内容，人名，文字是否为空，文字大于50或者小于0
    for i in range(num_chat):
        chat_names[i].change(
            check_chat,
            inputs=[chat_names[i], chat_contents[i]],
            outputs=[chat_names[i], chat_contents[i]],
        )
        chat_contents[i].blur(
            check_chat,
            inputs=[chat_names[i], chat_contents[i]],
            outputs=[chat_names[i], chat_contents[i]],
        )
    # 最后一次判断参数是否合格，合格则输出
    confirm_bn.click(
        check_params,
        inputs=names + models + languages + chat_names + chat_contents,
        outputs=[info, ok_bn],
    )
    # # 开始生成按钮，点击后生成音频
    ok_bn.click(set_invisible, inputs=[], outputs=ok_bn)
    ok_bn.click(generate_audio, inputs=model_path, outputs=[ok_bn, audio])

# app.queue(concurrency_count=511, max_size=1022).launch(
#     server_name="0.0.0.0", server_port=9527, inbrowser=True
# )
app.launch(server_name="0.0.0.0", server_port=9527, inbrowser=True)
