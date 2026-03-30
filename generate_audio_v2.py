#!/usr/bin/env python3
"""Batch-generate flashcard audio using sentence-bookend format.

Format: [Chinese sentence]。[English. 字. pinyin. ...]。[Chinese sentence]。
Theme: "A Day in China" — practical tourist sentences, loosely connected story.
"""
import os
import numpy as np
import soundfile as sf
import subprocess
from qwen_tts import Qwen3TTSModel

# All 31 characters with example sentences — "A Day in China" story
# Format: (char, pinyin, chinese_sentence, breakdown)
# breakdown = "English. 字. pinyin. English. 字. pinyin. ..."
CARDS = [
    # --- Arriving ---
    ("好", "hǎo",
     "你好！",
     "you. 你. nǐ. good. 好. hǎo"),

    ("中", "zhōng",
     "我来中国了。",
     "I. 我. wǒ. come to. 来. lái. middle. 中. zhōng. country. 国. guó"),

    ("大", "dà",
     "这个城市很大。",
     "this. 这个. zhège. city. 城市. chéngshì. very. 很. hěn. big. 大. dà"),

    ("人", "rén",
     "街上人很多。",
     "street. 街上. jiēshàng. people. 人. rén. very. 很. hěn. many. 多. duō"),

    ("门", "mén",
     "我走进大门。",
     "I. 我. wǒ. walk into. 走进. zǒujìn. big. 大. dà. door. 门. mén"),

    # --- Exploring ---
    ("天", "tiān",
     "今天天气很好。",
     "today. 今天. jīntiān. weather. 天气. tiānqì. very. 很. hěn. good. 好. hǎo"),

    ("日", "rì",
     "今日是个好日子。",
     "today. 今日. jīnrì. is. 是. shì. a good. 好. hǎo. day. 日子. rìzi"),

    ("目", "mù",
     "我目不转睛地看。",
     "I. 我. wǒ. eye. 目. mù. not. 不. bù. turn. 转. zhuǎn. stare. 睛. jīng"),

    ("小", "xiǎo",
     "我找到一个小店。",
     "I. 我. wǒ. find. 找到. zhǎodào. one. 一个. yígè. small. 小. xiǎo. shop. 店. diàn"),

    ("口", "kǒu",
     "门口有人在等。",
     "doorway. 门口. ménkǒu. have. 有. yǒu. people. 人. rén. waiting. 在等. zàiděng"),

    # --- Eating ---
    ("火", "huǒ",
     "我们去吃火锅。",
     "we. 我们. wǒmen. go. 去. qù. eat. 吃. chī. fire. 火. huǒ. pot. 锅. guō"),

    ("水", "shuǐ",
     "请给我一杯水。",
     "please. 请. qǐng. give me. 给我. gěiwǒ. one. 一. yī. cup. 杯. bēi. water. 水. shuǐ"),

    ("手", "shǒu",
     "我用手拿筷子。",
     "I. 我. wǒ. use. 用. yòng. hand. 手. shǒu. hold. 拿. ná. chopsticks. 筷子. kuàizi"),

    # --- Nature trip ---
    ("山", "shān",
     "我喜欢爬山。",
     "I. 我. wǒ. like. 喜欢. xǐhuān. climbing. 爬. pá. mountains. 山. shān"),

    ("木", "mù",
     "山上有很多树木。",
     "mountain. 山上. shānshàng. have. 有. yǒu. very many. 很多. hěnduō. trees. 树木. shùmù"),

    ("石", "shí",
     "路上有大石头。",
     "road. 路上. lùshàng. have. 有. yǒu. big. 大. dà. stone. 石头. shítou"),

    ("田", "tián",
     "山下是绿色的田。",
     "below the mountain. 山下. shānxià. is. 是. shì. green. 绿色的. lǜsède. field. 田. tián"),

    ("风", "fēng",
     "山上的风很大。",
     "mountaintop. 山上的. shānshàngde. wind. 风. fēng. very. 很. hěn. strong. 大. dà"),

    ("雨", "yǔ",
     "突然下雨了。",
     "suddenly. 突然. tūrán. start. 下. xià. rain. 雨. yǔ"),

    ("月", "yuè",
     "雨停了，我看到月亮。",
     "rain. 雨. yǔ. stopped. 停了. tíngle. I. 我. wǒ. see. 看到. kàndào. moon. 月亮. yuèliàng"),

    # --- Meeting people ---
    ("王", "wáng",
     "他姓王。",
     "he. 他. tā. surname. 姓. xìng. king. 王. wáng"),

    ("女", "nǚ",
     "那个女人是导游。",
     "that. 那个. nàgè. woman. 女人. nǚrén. is. 是. shì. tour guide. 导游. dǎoyóu"),

    ("子", "zǐ",
     "她带着孩子。",
     "she. 她. tā. brings. 带着. dàizhe. child. 孩子. háizi"),

    ("心", "xīn",
     "我很开心。",
     "I. 我. wǒ. very. 很. hěn. happy. 开心. kāixīn"),

    ("马", "mǎ",
     "我们骑马回去。",
     "we. 我们. wǒmen. ride. 骑. qí. horse. 马. mǎ. go back. 回去. huíqù"),

    ("力", "lì",
     "爬山需要力气。",
     "climbing. 爬山. páshān. needs. 需要. xūyào. strength. 力气. lìqì"),

    # --- Numbers (shopping) ---
    ("一", "yī",
     "我买了一个苹果。",
     "I. 我. wǒ. bought. 买了. mǎile. one. 一. yī. measure word. 个. gè. apple. 苹果. píngguǒ"),

    ("二", "èr",
     "还要二杯茶。",
     "also want. 还要. háiyào. two. 二. èr. cups. 杯. bēi. tea. 茶. chá"),

    ("三", "sān",
     "一共三十块钱。",
     "total. 一共. yígòng. three. 三. sān. ten. 十. shí. dollars. 块钱. kuàiqián"),

    ("四", "sì",
     "我走了四条街。",
     "I. 我. wǒ. walked. 走了. zǒule. four. 四. sì. measure word. 条. tiáo. streets. 街. jiē"),

    ("五", "wǔ",
     "今天走了五公里。",
     "today. 今天. jīntiān. walked. 走了. zǒule. five. 五. wǔ. kilometers. 公里. gōnglǐ"),
]

OUT_DIR = os.path.join(os.path.dirname(__file__), "audio")
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading Qwen3-TTS model...", flush=True)
tts = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="mps",
    dtype="float32",
)

success = 0
failed = []

for char, pinyin, chinese, breakdown in CARDS:
    # Build the bookend format: Chinese sentence → breakdown → Chinese sentence
    text = f"{chinese} {breakdown}. {chinese}"
    mp3_path = os.path.join(OUT_DIR, f"{char}.mp3")
    wav_path = f"/tmp/tts_card_{char}.wav"

    print(f"\nGenerating: {char} ({pinyin})", flush=True)
    print(f"  Text: {text}", flush=True)
    try:
        wavs, sr = tts.generate_custom_voice(
            text=text,
            language="Auto",
            speaker="serena",
            instruct="Speak naturally.",
        )

        wav = np.asarray(wavs[0], dtype=np.float32)
        dur = len(wav) / sr
        print(f"  Duration: {dur:.1f}s", flush=True)

        sf.write(wav_path, wav, sr)
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path,
             "-af", "loudnorm=I=-14:TP=-1:LRA=11",
             "-b:a", "192k", "-ar", "24000", mp3_path],
            capture_output=True,
        )

        if os.path.exists(mp3_path):
            os.remove(wav_path)
            success += 1
        else:
            failed.append(char)
            print(f"  ⚠️ ffmpeg failed", flush=True)

    except Exception as e:
        failed.append(char)
        print(f"  ❌ Failed: {e}", flush=True)

print(f"\n{'='*40}")
print(f"Done! {success}/{len(CARDS)} generated successfully.", flush=True)
if failed:
    print(f"Failed: {', '.join(failed)}", flush=True)
