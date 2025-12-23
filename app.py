# -*- coding: utf-8 -*-
"""
Quantum Encoder (HF v1)
Deterministic Kryon Encoder calculations + library tools.
"""

import os
import re
import io
import csv
import json
import math
import glob
import time
import datetime
import threading
import hashlib
from typing import Tuple, Optional, List, Dict, Any

import pandas as pd
import gradio as gr
from huggingface_hub import HfApi, CommitOperationAdd

# =========================
#  CSS для пользовательских стилей
# =========================
CUSTOM_CSS = """
/* Основной контейнер для отчёта. Увеличиваем базовый шрифт, чтобы текст было легче читать. */
.report-body {
    font-size: 150%;
    line-height: 1.4;
    color: #ffffff;
}
/* Заголовки разделов: белый цвет, чуть крупнее основного шрифта и жирный */
.section-heading {
    color: #ffffff;
    font-size: 160%;
    font-weight: bold;
    margin-top: 1.2em;
}
/* Примечания: более мелкий серый текст */
.small-note {
    font-size: 80%;
    color: #888888;
    line-height: 1.2;
}
"""

# =========================
#  Env / Версии / Глобалы
# =========================
SPACE_REPO_ID   = os.getenv("SPACE_REPO_ID", "")
HF_TOKEN        = os.getenv("HF_TOKEN", "")
ENCODER_VERSION = "v1.2"
CALC_VERSION    = "calc@2025-11-05"

MUTEX = threading.Lock()
LIB_DF: Optional[pd.DataFrame] = None

# Индексы для быстрого поиска по L1/L2C
INDEX_L1: Dict[int, List[str]] = {}
INDEX_L2C: Dict[int, List[str]] = {}
INDEX_READY: bool = False
LAST_RESULT: Dict[str, Any] = {}

# =========================
#  Контракт колонок библиотеки (SOURCE OF TRUTH)
# =========================
LIB_COLS = [
    "word",
    "sphere",
    "tone",
    "allowed",
    "field",
    "role",
    "notes",
    "l1",
    "l2c",
    "w",
    "C",
    "Hm",
    "Z",
]

# =========================
#  Конфиг ядра (config.json)
# =========================
CONFIG_PATH = "config.json"
DEFAULT_CONFIG = {
    "sigma_Z": 0.80,                 # ширина колокола для Z по умолчанию
    "resonator_threshold": 0.75,     # порог силы резонансной пары
    "cluster_bounds": {              # (сейчас не используем; храним для будущего)
        "phi": [1.00, 1.60],
        "e":   [1.60, 2.70],
        "e-pi":[2.70, 3.20],
        "pi":  [3.20, 99.00],
        "rt2": [1.214, 1.614]
    }
}

def load_config() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # заполняем отсутствующие поля дефолтами
        for k, v in DEFAULT_CONFIG.items():
            if k not in cfg:
                cfg[k] = v
        return cfg
    except Exception:
        return DEFAULT_CONFIG.copy()

def save_config(cfg: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

# Глобальный конфиг (читает твой текущий config.json)
APP_CFG = load_config()

def set_cfg_values(sigma: float, reson_thr: float, bounds: dict) -> Tuple[bool, str]:
    """Обновить APP_CFG в памяти и сохранить локально (без коммита)."""
    global APP_CFG
    try:
        new_cfg = APP_CFG.copy()
        new_cfg["sigma_Z"] = float(sigma)
        new_cfg["resonator_threshold"] = float(reson_thr)
        if isinstance(bounds, dict):
            new_cfg["cluster_bounds"] = bounds
        save_config(new_cfg)
        # перечитываем в глобальную переменную
        APP_CFG = load_config()
        return True, f"Сохранено локально: sigma_Z={APP_CFG['sigma_Z']:.2f}, resonator_threshold={APP_CFG['resonator_threshold']:.2f}"
    except Exception as e:
        return False, f"Ошибка сохранения: {type(e).__name__}: {e}"

def commit_config(message: str = "Update config.json") -> str:
    """Закоммитить config.json в репозиторий (если заданы HF_TOKEN/SPACE_REPO_ID)."""
    if not (HF_TOKEN and SPACE_REPO_ID):
        return "ℹ️ Автокоммит отключён (нет HF_TOKEN/SPACE_REPO_ID)."
    try:
        return commit_ops([CONFIG_PATH], message)
    except Exception as e:
        return f"⚠️ Commit error: {type(e).__name__}: {e}"

def reset_to_defaults() -> Tuple[bool, str]:
    """Откатить к дефолтным значениям и сохранить локально."""
    global APP_CFG
    try:
        APP_CFG = DEFAULT_CONFIG.copy()
        save_config(APP_CFG)
        return True, "Откат к дефолту выполнен."
    except Exception as e:
        return False, f"Ошибка отката: {type(e).__name__}: {e}"

# =========================
#  Kryon-33: базовые вещи
# =========================
KRYON_MAP = {
    "А":1,"Б":2,"В":3,"Г":4,"Д":5,"Е":6,"Ё":7,"Ж":8,"З":9,"И":10,"Й":11,
    "К":12,"Л":13,"М":14,"Н":15,"О":16,"П":17,"Р":18,"С":19,"Т":20,"У":21,
    "Ф":22,"Х":23,"Ц":24,"Ч":25,"Ш":26,"Щ":27,"Ь":28,"Ы":29,"Ъ":30,"Э":31,"Ю":32,"Я":33
}

HUNDS = ["","СТО","ДВЕСТИ","ТРИСТА","ЧЕТЫРЕСТА","ПЯТЬСОТ","ШЕСТЬСОТ","СЕМЬСОТ","ВОСЕМЬСОТ","ДЕВЯТЬСОТ"]
TENS  = ["","ДЕСЯТЬ","ДВАДЦАТЬ","ТРИДЦАТЬ","СОРОК","ПЯТЬДЕСЯТ","ШЕСТЬДЕСЯТ","СЕМЬДЕСЯТ","ВОСЕМЬДЕСЯТ","ДЕВЯНОСТО"]
UNITS = ["","ОДИН","ДВА","ТРИ","ЧЕТЫРЕ","ПЯТЬ","ШЕСТЬ","СЕМЬ","ВОСЕМЬ","ДЕВЯТЬ"]
UNITS_FEM = ["","ОДНА","ДВЕ","ТРИ","ЧЕТЫРЕ","ПЯТЬ","ШЕСТЬ","СЕМЬ","ВОСЕМЬ","ДЕВЯТЬ"]
TEENS = ["ДЕСЯТЬ","ОДИННАДЦАТЬ","ДВЕНАДЦАТЬ","ТРИНАДЦАТЬ","ЧЕТЫРНАДЦАТЬ",
         "ПЯТНАДЦАТЬ","ШЕСТНАДЦАТЬ","СЕМНАДЦАТЬ","ВОСЕМНАДЦАТЬ","ДЕВЯТНАДЦАТЬ"]

def normalize(t: str) -> str:
    """Удаляет все символы, кроме кириллических букв, и приводит к верхнему регистру."""
    return re.sub(r"[^А-ЯЁ]", "", (t or "").upper())

def _number_to_words_0_999(n: int, feminine: bool = False) -> str:
    """
    Преобразует число 0-999 в русские слова (верхний регистр).
    feminine: True для женского рода (ОДНА, ДВЕ), False для мужского (ОДИН, ДВА).
    """
    n = int(n)
    if n == 0:
        return "НОЛЬ"
    
    units_arr = UNITS_FEM if feminine else UNITS
    
    h = n // 100
    t = (n % 100) // 10
    u = n % 10
    
    out: List[str] = []
    
    if h:
        out.append(HUNDS[h])
    
    if t == 1:
        out.append(TEENS[u])
    else:
        if t:
            out.append(TENS[t])
        if u:
            out.append(units_arr[u])
    
    return " ".join(out)

def number_to_words_ru_0_999999(n: int) -> str:
    """
    Каноническое преобразование числа в русские слова (верхний регистр).
    Поддерживает диапазон: 0...999999 (включая тысячи).
    Падеж: именительный, верхний регистр.
    Возвращает строку с пробелами между словами.
    """
    n = int(n)
    if n == 0:
        return "НОЛЬ"
    
    if n < 1000:
        return _number_to_words_0_999(n, feminine=False)
    
    # Обработка тысяч
    T = n // 1000  # количество тысяч (1..999)
    R = n % 1000   # остаток
    
    # Определение формы слова "ТЫСЯЧА"
    T_mod_100 = T % 100
    T_mod_10 = T % 10
    
    if T_mod_100 in (11, 12, 13, 14):
        thousand_word = "ТЫСЯЧ"
    elif T_mod_10 == 1:
        thousand_word = "ТЫСЯЧА"
    elif T_mod_10 in (2, 3, 4):
        thousand_word = "ТЫСЯЧИ"
    else:
        thousand_word = "ТЫСЯЧ"
    
    # Преобразование количества тысяч (женский род для тысяч)
    thousands_str = _number_to_words_0_999(T, feminine=True)
    
    parts = [thousands_str, thousand_word]
    
    # Добавляем остаток, если есть
    if R > 0:
        rest_str = _number_to_words_0_999(R, feminine=False)
        parts.append(rest_str)
    
    return " ".join(parts)

def calc_l2c_from_l1(l1: int):
    """
    Вычисляет L2C, текстовое представление L1 и склеенную строку.
    Возвращает: (l2c, words, glued, out_of_range)
    Если l1 < 0 или l1 > 999999, возвращает (None, None, None, True).
    """
    l1 = int(l1)
    
    # Проверка диапазона
    if l1 < 0 or l1 > 999999:
        return (None, None, None, True)
    
    words = number_to_words_ru_0_999999(l1)
    # Удаляем все пробелы, табы, переносы строк и дефисы
    glued = re.sub(r"[\s\t\n\r\-]+", "", words)
    # Нормализуем: только кириллические буквы
    glued = normalize(glued)
    l2c = sum(KRYON_MAP.get(ch, 0) for ch in glued)
    return (l2c, words, glued, False)

# =========================
#  Даты → «официальные» формы
# =========================
ORD_DAY = {
    1:"ПЕРВОЕ",2:"ВТОРОЕ",3:"ТРЕТЬЕ",4:"ЧЕТВЁРТОЕ",5:"ПЯТОЕ",6:"ШЕСТОЕ",7:"СЕДЬМОЕ",8:"ВОСЬМОЕ",9:"ДЕВЯТОЕ",
    10:"ДЕСЯТОЕ",11:"ОДИННАДЦАТОЕ",12:"ДВЕНАДЦАТОЕ",13:"ТРИНАДЦАТОЕ",14:"ЧЕТЫРНАДЦАТОЕ",15:"ПЯТНАДЦАТОЕ",
    16:"ШЕСТНАДЦАТОЕ",17:"СЕМНАДЦАТОЕ",18:"ВОСЕМНАДЦАТОЕ",19:"ДЕВЯТНАДЦАТОЕ",20:"ДВАДЦАТОЕ",
    21:"ДВАДЦАТЬ ПЕРВОЕ",22:"ДВАДЦАТЬ ВТОРОЕ",23:"ДВАДЦАТЬ ТРЕТЬЕ",24:"ДВАДЦАТЬ ЧЕТВЁРТОЕ",
    25:"ДВАДЦАТЬ ПЯТОЕ",26:"ДВАДЦАТЬ ШЕСТОЕ",27:"ДВАДЦАТЬ СЕДЬМОЕ",28:"ДВАДЦАТЬ ВОСЬМОЕ",
    29:"ДВАДЦАТЬ ДЕВЯТОЕ",30:"ТРИДЦАТОЕ",31:"ТРИДЦАТЬ ПЕРВОЕ"
}
MONTHS_GEN = {1:"ЯНВАРЯ",2:"ФЕВРАЛЯ",3:"МАРТА",4:"АПРЕЛЯ",5:"МАЯ",6:"ИЮНЯ",7:"ИЮЛЯ",8:"АВГУСТА",9:"СЕНТЯБРЯ",10:"ОКТЯБРЯ",11:"НОЯБРЯ",12:"ДЕКАБРЯ"}
ORD_UNIT_GEN_M = {1:"ПЕРВОГО",2:"ВТОРОГО",3:"ТРЕТЬЕГО",4:"ЧЕТВЁРТОГО",5:"ПЯТОГО",6:"ШЕСТОГО",7:"СЕДЬМОГО",8:"ВОСЬМОГО",9:"ДЕВЯТОГО"}
ORD_TEEN_GEN_M = {10:"ДЕСЯТОГО",11:"ОДИННАДЦАТОГО",12:"ДВЕНАДЦАТОГО",13:"ТРИНАДЦАТОГО",14:"ЧЕТЫРНАДЦАТОГО",
                  15:"ПЯТНАДЦАТОГО",16:"ШЕСТНАДЦАТОГО",17:"СЕМНАДЦАТОГО",18:"ВОСЕМНАДЦАТОГО",19:"ДЕВЯТНАДЦАТОГО"}
TENS_CARD      = {2:"ДВАДЦАТЬ",3:"ТРИДЦАТЬ",4:"СОРОК",5:"ПЯТЬДЕСЯТ",6:"ШЕСТЬДЕСЯТ",7:"СЕМЬДЕСЯТ",8:"ВОСЕМЬДЕСЯТ",9:"ДЕВЯНОСТО"}
TENS_ORD_GEN_M = {2:"ДВАДЦАТОГО",3:"ТРИДЦАТОГО",4:"СОРОКОВОГО",5:"ПЯТИДЕСЯТОГО",6:"ШЕСТИДЕСЯТОГО",7:"СЕМИДЕСЯТОГО",8:"ВОСЬМИДЕСЯТОГО",9:"ДЕВЯНОСТОГО"}
HUND_ORD_GEN_M = {1:"СОТОГО",2:"ДВУХСОТОГО",3:"ТРЁХСОТОГО",4:"ЧЕТЫРЁХСОТОГО",5:"ПЯТИСОТОГО",6:"ШЕСТИСОТОГО",7:"СЕМИСОТОГО",8:"ВОСЬМИСОТОГО",9:"ДЕВЯТИСОТОГО"}

def is_leap(y:int)->bool:
    return (y%400==0) or (y%4==0 and y%100!=0)

def days_in_month(m:int,y:int)->int:
    return 31 if m in (1,3,5,7,8,10,12) else 30 if m in (4,6,9,11) else 29 if is_leap(y) else 28

DATE_RE = re.compile(r"^\s*(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})\s*$")

def last2_to_ordinal_gen_m(n:int)->str:
    if 10 <= n <= 19:
        return ORD_TEEN_GEN_M[n]
    t=n//10; u=n%10
    if t==0:
        return ORD_UNIT_GEN_M.get(u, "")
    if u==0:
        return TENS_ORD_GEN_M.get(t, "")
    return f"{TENS_CARD[t]} {ORD_UNIT_GEN_M[u]}"

def thousands_phrase(th:int)->str:
    if th==1:
        return "ОДНА ТЫСЯЧА"
    if th==2:
        return "ДВЕ ТЫСЯЧИ"
    base = ["","ОДНА","ДВЕ","ТРИ","ЧЕТЫРЕ","ПЯТЬ","ШЕСТЬ","СЕМЬ","ВОСЕМЬ","ДЕВЯТЬ"][th]
    tail = "ТЫСЯЧИ" if th in (3,4) else "ТЫСЯЧ"
    return f"{base} {tail}"

def date_to_phrase_official(d:int,m:int,y:int)->str:
    day = ORD_DAY[d]; month = MONTHS_GEN[m]
    th = y // 1000; h = (y % 1000) // 100; last2 = y % 100
    if y == 2000:
        return f"{day} {month} ДВУХТЫСЯЧНОГО ГОДА"
    parts = [thousands_phrase(th)]
    if last2 == 0:
        if h:
            parts = [thousands_phrase(th), HUND_ORD_GEN_M[h]]
        return f"{day} {month} {' '.join(parts)} ГОДА"
    if h:
        parts.append(HUND_ORD_GEN_M[h])
    parts.append(last2_to_ordinal_gen_m(last2))
    return f"{day} {month} {' '.join(parts)} ГОДА"

def parse_date_phrase(text:str)->Tuple[Optional[str], Optional[str]]:
    m = DATE_RE.match(text or "")
    if not m:
        return None, None
    d,mo,y = map(int, m.groups())
    if not (1000 <= y <= 9999 and 1 <= mo <= 12 and 1 <= d <= days_in_month(mo,y)):
        return None, None
    return date_to_phrase_official(d,mo,y), f"{y:04d}{mo:02d}{d:02d}"

# =========================
#  Метрики (с клиппингом)
# =========================
def calc_l1_from_string(s:str):
    w = normalize(s)
    if not w:
        return None, 0
    return w, sum(KRYON_MAP.get(ch,0) for ch in w)

def metrics(l1:int,l2c:int):
    w = l2c / l1
    ratio = abs(l2c - l1) / (l2c + l1)
    C  = math.cos(math.pi/2 * ratio)**2
    targets = [1, 1.25, 1.33, 1.5, 2, 3]
    Hm_raw = 1 - min(abs(w - t) / t for t in targets)
    Hm = max(0.0, min(1.0, Hm_raw))
    sigma = float(APP_CFG.get("sigma_Z", 0.8))
    Z_raw = (C * Hm) * math.exp(-((w - 2) / sigma)**2)
    Z = max(0.0, min(1.0, Z_raw))
    return w, C, Hm, Z

# =========================
#  Классификация начального импульса
# =========================
def classify_initial(v: Optional[int]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Классифицирует значение первой буквы (код Kryon‑33) по пяти типам:
    Активная (1–7), Гармоничная (8–16), Переходная (17–22), Турбулентная (23–25), Инверсная (26–33).
    Возвращает тройку (тип, действие, описание). Если v None или вне диапазона — возвращает (None,None,None).
    """
    if not v:
        return (None, None, None)
    try:
        v_int = int(v)
    except Exception:
        return (None, None, None)
    if 1 <= v_int <= 7:
        return ("Активная", "импульс наружу", "запуск")
    if 8 <= v_int <= 16:
        return ("Гармоничная", "равновесие", "стабилизация")
    if 17 <= v_int <= 22:
        return ("Переходная", "сдвиг", "адаптация")
    if 23 <= v_int <= 25:
        return ("Турбулентная", "напряжение", "пик")
    if 26 <= v_int <= 33:
        return ("Инверсная", "внутренняя работа", "возврат к ядру")
    return (None, None, None)

# =========================
#  Дополнительные расчёты для расширенного анализа
# =========================
def cluster_by_w(w: float) -> Tuple[str, str]:
    if 1.0 <= w < 1.6:
        return "phi", "φ-ядро"
    if 1.6 <= w < 2.7:
        return "e", "e"
    if 2.7 <= w < 3.2:
        return "e-pi", "e–π"
    return "pi", "π"

CLUSTER_ADVICES = {
    "phi": ("φ-ядро", "Гармония и стабильность. Совет: добавить e-слово (движение)."),
    "e":   ("e", "Рост и импульс. Совет: добавить φ-слово (покой)."),
    "e-pi":("e–π", "Прорыв, интенсивность. Совет: внести равновесие φ или √2."),
    "pi":  ("π", "Турбулентность. Совет: успокоить через φ и Z.")
}

RESONANCE_PAIRS = {
    ("phi","rt2"): ("φ–√2", "Harmony ↔ Duality", "Единство через различие — гармония, рождающаяся из двух полюсов."),
    ("phi","e"):   ("φ–e",  "Harmony ↔ Growth", "Переход гармонии в движение"),
    ("e","pi"):    ("e–π",  "Growth ↔ Cycle", "Прорыв и кульминация"),
    ("pi","rt2"):  ("π–√2", "Cycle ↔ Transition", "Завершение и тишина"),
    ("phi","pi"):  ("φ–π",  "Harmony ↔ Cycle", "Покой и полнота"),
    ("e","rt2"):   ("e–√2", "Growth ↔ Threshold", "Метаморфоза")
}

def resonance_pair(w: float, threshold: float = 0.0) -> Tuple[str, str, str, float]:
    r_phi = math.exp(-abs(w - 1.618))
    r_e   = math.exp(-abs(w - 2.718))
    r_pi  = math.exp(-abs(w - 3.142))
    r_rt2 = math.exp(-abs(w - 1.414))
    values = {"phi": r_phi, "e": r_e, "pi": r_pi, "rt2": r_rt2}
    top = sorted(values.items(), key=lambda x: x[1], reverse=True)[:2]
    (k1, v1), (k2, v2) = top[0], top[1]
    pair_key = tuple(sorted([k1, k2], key=lambda x: ["phi","e","pi","rt2"].index(x)))
    r_pair = math.sqrt(v1 * v2)
    if r_pair < threshold:
        return ("", "", "", r_pair)
    name, en, ru = RESONANCE_PAIRS.get(pair_key, ("", "", ""))
    return (name, en, ru, r_pair)

def fractal_unfold(l1: int) -> Tuple[str, int, int, float, str]:
    w_values = []
    curr_l1 = l1
    for _ in range(12):
        l2c, _, _, out_of_range = calc_l2c_from_l1(curr_l1)
        if out_of_range or l2c is None:
            # Остановка при выходе за диапазон
            break
        w, _, _, _ = metrics(curr_l1, l2c)
        w_values.append(w)
        curr_l1 = l2c
    
    # Обработка случая out_of_range
    if not w_values:
        return "—", 0, 0, 0.0, "out_of_range"
    
    pattern_chars = []
    inhale = 0
    exhale = 0
    for i in range(len(w_values)-1):
        if w_values[i+1] > w_values[i]:
            pattern_chars.append('●')
            inhale += 1
        else:
            pattern_chars.append('○')
            exhale += 1
    pattern_chars.append('→')
    
    if exhale == 0:
        R = float('inf') if inhale > 0 else 1.0
    else:
        R = inhale / exhale
    
    if R > 1.05:
        interp = "слово раскрывается"
    elif R < 0.95:
        interp = "слово стабилизирует поле"
    else:
        interp = "слово в равновесии"
    
    return ''.join(pattern_chars), inhale, exhale, R, interp

def fii_bar(fii: float) -> Tuple[str, str]:
    segments = max(0, min(10, round((fii + 10) / 2)))
    bar = '▰' * segments + '▱' * (10 - segments)
    if fii <= -6:
        cat = "🔴 Разрушитель — создаёт напряжение, дестабилизирует поле"
    elif fii <= -2:
        cat = "🟠 Ослабитель — рассеивает энергию, снижает фокус"
    elif fii < 2:
        cat = "⚪ Нейтральное — сбалансированное, не влияет заметно"
    elif fii < 6:
        cat = "🟢 Гармонизатор — усиливает гармонию и согласие"
    else:
        cat = "🔵 Резонатор — максимально усиливает поле, световой пик"
    return bar, cat

def q_bar(q: float, length: int = 10) -> str:
    filled = max(0, min(length, round(q * length)))
    return '▰' * filled + '▱' * (length - filled)

# =========================
#  Глубинная интерпретация
# =========================
def generate_deep_interpretation(res: Dict[str, Any]) -> str:
    """
    Формирует текстовую интерпретацию слова на основе ключевых показателей.

    Используются:
      - кластер W, который определяет «зону» (покой, рост, прорыв, турбулентность);
      - индекс FII, задающий влияние слова на общее поле;
      - показатели Z, C и Hm (гармония, согласованность и музыкальность) с грубой шкалой «высокий/средний/низкий»;
      - рекомендации по добавлению дополнительных слов в зависимости от кластера.

    Возвращает связный текст из нескольких предложений, понятный пользователю.
    """
    # Описание зон в зависимости от кластера W
    cluster_phrases = {
        'phi': 'зона покоя и устойчивости',
        'e':   'зона роста и движения',
        'e-pi':'зона прорыва и пика активности',
        'pi':  'зона напряжения и турбулентности'
    }
    # Описание влияния FII
    fii = res.get('fii', 0.0)
    if fii <= -6:
        fii_desc = 'может вызывать сильный внутренний дискомфорт и напряжение'
    elif fii <= -2:
        fii_desc = 'рассеивает внимание и ослабляет фокус'
    elif fii < 2:
        fii_desc = 'не вносит заметных изменений в ваше состояние'
    elif fii < 6:
        fii_desc = 'усиливает гармонию и поддерживает согласие'
    else:
        fii_desc = 'максимально усиливает ваше состояние'
    # Функция для грубого уровня показателей
    def level(value: float) -> str:
        if value > 0.6:
            return 'высокие'
        elif value >= 0.3:
            return 'средние'
        return 'низкие'
    z_level = level(res.get('Z', 0.0))
    c_level = level(res.get('C', 0.0))
    hm_level = level(res.get('Hm', 0.0))
    # Формируем список описаний
    parts: List[str] = []
    # Музыкальность / Hm
    if hm_level == 'высокие':
        parts.append('плавное звучание')
    elif hm_level == 'средние':
        parts.append('умеренная музыкальность')
    else:
        parts.append('несогласованная ритмика')
    # Когерентность / C
    if c_level == 'высокие':
        parts.append('внутренняя согласованность')
    elif c_level == 'средние':
        parts.append('частичная согласованность')
    else:
        parts.append('конфликтность частей')
    # Интегральная гармония / Z
    if z_level == 'высокие':
        parts.append('целостность и гармония')
    elif z_level == 'средние':
        parts.append('некоторая собранность')
    else:
        parts.append('разрозненность')
    metrics_desc = ', '.join(parts)
    # Советы по подбору слов
    suggestions = {
        'phi': 'Попробуйте дополнить его словами движения: «путь», «рост», «развитие».',
        'e':   'Добавьте слова покоя и устойчивости: «спокойствие», «равновесие».',
        'e-pi':'Уравновесьте его сочетанием с более стабильными и симметричными словами.',
        'pi':  'Снизьте напряжение, используя слова гармонии и симметрии.'
    }
    cluster_code = res.get('cluster_code', '')
    cluster_desc = cluster_phrases.get(cluster_code, 'зона неопределённого состояния')
    suggestion = suggestions.get(cluster_code, '')
    # Формируем итоговый текст
    text = f"Это слово относится к {cluster_desc}, {fii_desc}. "
    text += f"Его показатели отражают {metrics_desc}. "
    text += suggestion
    return text

# =========================
#  Вспомогательная функция: парсинг категории FII
# =========================
def parse_fii_category_str(cat: str) -> Tuple[str, str]:
    """
    Разбирает строку категории FII, отделяя название от описания.
    Строка имеет формат «🟠 Ослабитель — рассеивает энергию, снижает фокус».
    Возвращает кортеж (название без эмодзи, описание).
    """
    if not cat:
        return "", ""
    # отрезаем эмодзи и пробел после него
    s = cat.strip()
    # находим первую пробел после emoji
    first_space = s.find(' ')
    if first_space != -1:
        s_no_emoji = s[first_space + 1:].strip()
    else:
        s_no_emoji = s
    # разделяем по длинному тире (—) если есть
    if '—' in s_no_emoji:
        label, desc = s_no_emoji.split('—', 1)
    elif '-' in s_no_emoji:
        label, desc = s_no_emoji.split('-', 1)
    else:
        label, desc = s_no_emoji, ''
    return label.strip(), desc.strip()

def analyze_word(raw_input: str) -> Dict[str, Any]:
    phrase, _ = parse_date_phrase(raw_input or "")
    src = phrase if phrase else raw_input
    norm, l1 = calc_l1_from_string(src)
    if not l1:
        return {}
    l2c, words, _, out_of_range = calc_l2c_from_l1(l1)
    if out_of_range or l2c is None:
        return {}
    w, C, Hm, Z = metrics(l1, l2c)
    q_total = (Z + C + Hm) / 3.0
    fii = 10 * (0.4 * Z + 0.3 * q_total + 0.2 * C + 0.1 * Hm - 0.5)
    cluster_code, cluster_ru = cluster_by_w(w)
    cluster_name, advice = CLUSTER_ADVICES[cluster_code]
    th = float(APP_CFG.get("resonator_threshold", 0.75))
    pair_code, pair_en, pair_ru, r_pair = resonance_pair(w, threshold=th)
    pattern, inh, exh, r_coef, r_interp = fractal_unfold(l1)
    fii_b, fii_cat = fii_bar(fii)
    q_b = q_bar(q_total)
    first_char = norm[:1] if norm else ""
    first_val = KRYON_MAP.get(first_char, None)
    t,d,m = classify_initial(first_val)
    R_phi = math.exp(-abs(w - 1.618))
    R_e   = math.exp(-abs(w - 2.718))
    R_pi  = math.exp(-abs(w - 3.142))
    R_rt2 = math.exp(-abs(w - 1.414))
    values = {"φ": R_phi, "e": R_e, "π": R_pi, "√2": R_rt2}
    r_max_label = max(values.items(), key=lambda x: x[1])[0]
    r_max_val = values[r_max_label]
    return {
        'raw': raw_input,
        'phrase_used': src,
        'norm': norm,
        'l1': l1,
        'l2c': l2c,
        'w': w,
        'C': C,
        'Hm': Hm,
        'Z': Z,
        'q_total': q_total,
        'fii': fii,
        'cluster_code': cluster_code,
        'cluster_ru': cluster_ru,
        'cluster_name': cluster_name,
        'cluster_advice': advice,
        'res_pair_code': pair_code,
        'res_pair_en': pair_en,
        'res_pair_ru': pair_ru,
        'res_pair_value': r_pair,
        'fractal_pattern': pattern,
        'fractal_inhale': inh,
        'fractal_exhale': exh,
        'fractal_R': r_coef,
        'fractal_interp': r_interp,
        'fii_bar': fii_b,
        'fii_category': fii_cat,
        'q_bar': q_b,
        'first_char': first_char,
        'first_val': first_val,
        'first_impulse': (t,d,m),
        'R_phi': R_phi,
        'R_e': R_e,
        'R_pi': R_pi,
        'R_rt2': R_rt2,
        'resonator_max': (r_max_label, r_max_val)
    }

# >>> PATCH: autopick L1/L2C for FA by W-neighborhood
def _autopick_l1_l2c_for_fa(W_target: float,
                             eps_steps=(0.005, 0.01, 0.02, 0.05)) -> Tuple[Optional[int], Optional[int], float, int]:
    """
    Подбирает L1 и L2C по окрестности W в библиотеке:
    - ищем слова с |w - W_target| <= eps (по возрастанию eps),
    - берём моды (самые частые значения) для l1 и l2c,
    - возвращаем (l1, l2c, eps_used, hits).
    Если не нашли — (None, None, 0.0, 0).
    """
    if LIB_DF is None or (isinstance(LIB_DF, pd.DataFrame) and LIB_DF.empty):
        return None, None, 0.0, 0

    try:
        df = LIB_DF.copy()
        df["w"] = pd.to_numeric(df["w"], errors="coerce")
        df["l1"] = pd.to_numeric(df["l1"], errors="coerce")
        df["l2c"] = pd.to_numeric(df["l2c"], errors="coerce")
        df = df.dropna(subset=["w", "l1", "l2c"])
    except Exception:
        return None, None, 0.0, 0

    for eps in eps_steps:
        cand = df[(df["w"].sub(W_target).abs() <= eps)]
        if len(cand) == 0:
            continue
        try:
            l1_mode = int(cand["l1"].value_counts().index[0])
        except Exception:
            l1_mode = None
        try:
            l2c_mode = int(cand["l2c"].value_counts().index[0])
        except Exception:
            l2c_mode = None

        if l1_mode is not None and l2c_mode is not None:
            return l1_mode, l2c_mode, float(eps), int(len(cand))

    return None, None, 0.0, 0
# <<< PATCH

# >>> PATCH: FA analyzer (build result from given W,C,Hm,Z,Φ)
def analyze_from_fa(raw_label: str,
                    W_in: float, C_in: float, Hm_in: float, Z_in: float, Phi_in: Optional[float]) -> Dict[str, Any]:
    """
    Строит полный результат отчёта из интегрального FractalAvatar-профиля:
    W, C, Hm, Z, Φ (без пересчёта через L1/L2C). L1/L2C синтезируем только как служебные.
    """
    cluster_code, cluster_ru = cluster_by_w(W_in)
    cluster_name, advice = CLUSTER_ADVICES[cluster_code]

    # служебные L1/L2C → сначала пробуем автоподбор по окрестности W
    l1_pick, l2c_pick, eps_used, hits = _autopick_l1_l2c_for_fa(float(W_in))
    if l1_pick is None or l2c_pick is None:
        # fallback: синтетические коды
        l1 = 1000
        l2c = int(round(float(W_in) * l1))
        fa_autopick = {"used": False, "eps": None, "hits": 0}
    else:
        l1 = int(l1_pick)
        l2c = int(l2c_pick)
        fa_autopick = {"used": True, "eps": eps_used, "hits": hits}

    q_total = (Z_in + C_in + Hm_in) / 3.0
    fii = 10 * (0.4 * Z_in + 0.3 * q_total + 0.2 * C_in + 0.1 * Hm_in - 0.5)
    fii_b, fii_cat = fii_bar(fii)
    q_b = q_bar(q_total)

    th = float(APP_CFG.get("resonator_threshold", 0.75))
    pair_code, pair_en, pair_ru, r_pair = resonance_pair(W_in, threshold=th)

    # резонансное пространство
    R_phi = math.exp(-abs(W_in - 1.618))
    R_e   = math.exp(-abs(W_in - 2.718))
    R_pi  = math.exp(-abs(W_in - 3.142))
    R_rt2 = math.exp(-abs(W_in - 1.414))
    values = {"φ": R_phi, "e": R_e, "π": R_pi, "√2": R_rt2}
    r_max_label = max(values.items(), key=lambda x: x[1])[0]
    r_max_val = values[r_max_label]

    return {
        'raw': raw_label,
        'phrase_used': raw_label,
        'norm': raw_label,
        'l1': l1,
        'l2c': l2c,
        'w': float(W_in),
        'C': float(C_in),
        'Hm': float(Hm_in),
        'Z': float(Z_in),
        'Phi_align': None if Phi_in is None else float(Phi_in),
        'q_total': q_total,
        'fii': fii,
        'cluster_code': cluster_code,
        'cluster_ru': cluster_ru,
        'cluster_name': cluster_name,
        'cluster_advice': advice,
        'res_pair_code': pair_code,
        'res_pair_en': pair_en,
        'res_pair_ru': pair_ru,
        'res_pair_value': r_pair,
        'fractal_pattern': None,
        'fractal_inhale': 0,
        'fractal_exhale': 0,
        'fractal_R': 0.0,
        'fractal_interp': "FA-mode: без L1-развёртки",
        'fii_bar': fii_b,
        'fii_category': fii_cat,
        'q_bar': q_b,
        'first_char': None,
        'first_val': None,
        'first_impulse': (None, None, None),
        'R_phi': R_phi,
        'R_e': R_e,
        'R_pi': R_pi,
        'R_rt2': R_rt2,
        'resonator_max': (r_max_label, r_max_val),
        'fa_mode': True,
        'fa_autopick': fa_autopick
    }
# <<< PATCH

# =========================
#  JSON экспорт отчётов
# =========================
def build_json_report(res: Dict[str, Any]) -> str:
    if not res:
        return ""
    data = {
        'meta': {
            'encoder': 'Kryon-33',
            'version': ENCODER_VERSION,
            'calc_version': CALC_VERSION,
            'generated_at': datetime.datetime.utcnow().isoformat() + 'Z',
            'lang': 'ru'
        },
        'input': res['raw'],
        'phrase_used': res['phrase_used'],
        'metrics': {
            'L1': res['l1'],
            'L2C': res['l2c'],
            'W': round(res['w'], 3),
            'C': round(res['C'], 3),
            'Hm': round(res['Hm'], 3),
            'Z': round(res['Z'], 3),
            'Q_total': round(res['q_total'], 3),
            'FII': round(res['fii'], 3)
        },
        'cluster': res['cluster_code'],
        'resonance_pair': {
            'code': res['res_pair_code'],
            'en': res['res_pair_en'],
            'ru': res['res_pair_ru']
        },
        'fractal': {
            'pattern': res['fractal_pattern'],
            'inhale': res['fractal_inhale'],
            'exhale': res['fractal_exhale'],
            'R': res['fractal_R'],
            'interpretation': res['fractal_interp']
        }
    }
    path = f"/tmp/report_{int(time.time())}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path

# >>> PATCH: full JSON report with related words
def build_full_json_report(res: Dict[str, Any],
                           limit_l1: int = 500,
                           limit_l2c: int = 500,
                           limit_near: int = 50,
                           limit_contrast: int = 50) -> str:
    """
    Формирует расширенный JSON: метрики слова + связанные списки:
    - совпадения по L1 и L2C (из LIB_DF и персональной)
    - созвучные (near) и контрастные (contrast) по D
    """
    if not res:
        return ""
    l1_list, l2c_list = _collect_matches_by_code(res, limit_l1=limit_l1, limit_l2c=limit_l2c)
    near, contrast = _collect_near_contrast(res, limit_near=limit_near, limit_contrast=limit_contrast)

    # >>> PATCH: FA fallback — если by_L1/by_L2C пустые, подставим окрестность W
    if res.get("fa_mode", False):
        eps_fallback = res.get("fa_autopick", {}).get("eps", 0.02) or 0.02
        if not l1_list:
            l1_list = _fa_neighborhood_words(res, eps=eps_fallback, limit=limit_l1)
        if not l2c_list:
            l2c_list = _fa_neighborhood_words(res, eps=eps_fallback, limit=limit_l2c)

    data = {
        'meta': {
            'encoder': 'Kryon-33',
            'version': ENCODER_VERSION,
            'calc_version': CALC_VERSION,
            'generated_at': datetime.datetime.utcnow().isoformat() + 'Z',
            'lang': 'ru'
        },
        'input': res['raw'],
        'phrase_used': res['phrase_used'],
        'metrics': {
            'L1': res['l1'],
            'L2C': res['l2c'],
            'W': round(res['w'], 3),
            'C': round(res['C'], 3),
            'Hm': round(res['Hm'], 3),
            'Z': round(res['Z'], 3),
            'Q_total': round(res['q_total'], 3),
            'FII': round(res['fii'], 3)
        },
        'cluster': res.get('cluster_code', ''),
        'resonance_pair': {
            'code': res.get('res_pair_code', ''),
            'en': res.get('res_pair_en', ''),
            'ru': res.get('res_pair_ru', ''),
            'value': round(float(res.get('res_pair_value', 0.0)), 3) if res.get('res_pair_value') is not None else None
        },
        'related': {
            'by_L1': l1_list,
            'by_L2C': l2c_list,
            'near': near,
            'contrast': contrast
        }
    }
    path = f"/tmp/full_report_{int(time.time())}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path
# <<< PATCH

# =========================
#  Фразовый анализ и экспорт
# =========================
def analyze_phrase(text: str):
    tokens = re.split(r"[\s,;]+", text or "")
    items = []
    valid_count = 0
    limit = 5000
    
    for tok in tokens:
        if not tok:
            continue
        if valid_count >= limit:
            break
        res = analyze_word(tok)
        if res:
            items.append({
                "word": res['norm'],
                "phrase_used": res['phrase_used'],
                "L1": res['l1'],
                "L2C": res['l2c'],
                "W": round(res['w'], 3),
                "C": round(res['C'], 3),
                "Hm": round(res['Hm'], 3),
                "Z": round(res['Z'], 3)
            })
            valid_count += 1
    
    df = pd.DataFrame(items)
    if not df.empty:
        total_processed = len(df)
        limit_note = " (обрезано до 5000)" if valid_count >= limit else ""
        summary = f"Всего слов: {total_processed}{limit_note} (лимит 5000) | ⌀W = {df['W'].mean():.2f} | ⌀Z = {df['Z'].mean():.2f}"
    else:
        summary = "Нет валидных слов."
        
    data_json = {
        "meta": {
            "encoder": "Kryon-33",
            "version": ENCODER_VERSION,
            "generated_at": datetime.datetime.utcnow().isoformat() + 'Z'
        },
        "words": items
    }
    return df, summary, data_json

def export_phrase_json(data_json: Dict[str, Any]):
    if not data_json:
        return ("", "Нет данных для экспорта.")
    path = f"/tmp/phrase_report_{int(time.time())}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data_json, f, ensure_ascii=False, indent=2)
    return path, "Файл phrase_report.json сформирован."

# =========================
#  Интеграция с персональной библиотекой и экспортом
# =========================
def commit_ops(paths: List[str], message: str) -> str:
    if not (HF_TOKEN and SPACE_REPO_ID):
        return "ℹ️ Автокоммит отключён (нет HF_TOKEN/SPACE_REPO_ID)."
    ops=[]
    for p in paths:
        with open(p, "rb") as f:
            ops.append(CommitOperationAdd(path_in_repo=os.path.relpath(p,"."), path_or_fileobj=io.BytesIO(f.read())))
    api = HfApi(token=HF_TOKEN)
    last_exc = None
    for attempt in range(4):
        try:
            api.create_commit(repo_id=SPACE_REPO_ID, repo_type="space", operations=ops,
                              commit_message=f"{message} | {datetime.datetime.utcnow().isoformat(timespec='seconds')}Z")
            return "✅ Закоммичено в репозиторий."
        except Exception as e:
            last_exc = e
            time.sleep(2 ** attempt)
    return f"⚠️ Commit error: {type(last_exc).__name__}: {last_exc}"

def atomic_write_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)

PERSONAL_DIR="./personal"
PERSONAL_CSV=os.path.join(PERSONAL_DIR,"personal.csv")

def ensure_personal_csv():
    os.makedirs(PERSONAL_DIR,exist_ok=True)
    if not os.path.exists(PERSONAL_CSV):
        atomic_write_csv(pd.DataFrame(columns=["text","phrase_used","l1","l2c","w","C","Hm","Z","created_at"]),
                         PERSONAL_CSV)
    keep_path = os.path.join(PERSONAL_DIR, ".keep")
    if not os.path.exists(keep_path):
        with open(keep_path, "wb") as k: k.write(b"keep")

def already_in_personal(text,phrase)->bool:
    if not os.path.exists(PERSONAL_CSV):
        return False
    with open(PERSONAL_CSV,"r",encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            if r.get("text") == text and r.get("phrase_used") == phrase:
                return True
    return False

def add_to_personal():
    if not LAST_RESULT:
        return ("Сначала сделайте расчёт.", gr.update(value=compute_base_indicator()))
    text=LAST_RESULT["input"]
    phrase=LAST_RESULT["phrase_used"]
    l1=LAST_RESULT["l1"]
    l2c=LAST_RESULT["l2c"]
    w=LAST_RESULT["w"]
    C=LAST_RESULT["C"]
    Hm=LAST_RESULT["Hm"]
    Z=LAST_RESULT["Z"]
    ensure_personal_csv()
    if already_in_personal(text,phrase):
        with MUTEX:
            msg = commit_ops([PERSONAL_CSV, os.path.join(PERSONAL_DIR,".keep")], "Ensure personal in repo")
        return (f"Уже в персональной. {msg}", gr.update(value=compute_base_indicator()))
    df = pd.read_csv(PERSONAL_CSV, encoding="utf-8")
    df.loc[len(df)] = [text, phrase, int(l1), int(l2c), float(f"{w:.6f}"),
                       float(f"{C:.6f}"), float(f"{Hm:.6f}"), float(f"{Z:.6f}"),
                       datetime.datetime.utcnow().isoformat()+"Z"]
    atomic_write_csv(df, PERSONAL_CSV)
    with MUTEX:
        msg = commit_ops([PERSONAL_CSV, os.path.join(PERSONAL_DIR,".keep")], "Update personal.csv")
    return (f"Добавлено: «{text}». {msg}", gr.update(value=compute_base_indicator()))

def slugify(title:str)->str:
    m = {"А":"A","Б":"B","В":"V","Г":"G","Д":"D","Е":"E","Ё":"E","Ж":"Zh","З":"Z","И":"I","Й":"Y",
         "К":"K","Л":"L","М":"M","Н":"N","О":"O","П":"P","Р":"R","С":"S","Т":"T","У":"U","Ф":"F",
         "Х":"H","Ц":"C","Ч":"Ch","Ш":"Sh","Щ":"Sch","Ы":"Y","Э":"E","Ю":"Yu","Я":"Ya","Ь":"","Ъ":""}
    t="".join(m.get(ch.upper(),ch) for ch in title)
    t=re.sub(r"[^A-Za-z0-9]+","-",t).strip("-").lower()
    return t or "sphere"

def parse_bool(x) -> bool:
    """
    Парсит значение в булево по канону:
    - если x bool -> вернуть x
    - если x int/float -> вернуть bool(x)
    - если x None -> False
    - если x str -> strip/lower и True только для: ("true","1","yes","y","да")
    - для всего остального -> False
    """
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if x is None:
        return False
    if isinstance(x, str):
        s = x.strip().lower()
        return s in ("true", "1", "yes", "y", "да")
    return False

def force_recalc_row(
    word: str,
    sphere: str,
    tone: str,
    allowed,
    notes,
    l1: Optional[int] = None,
    l2c: Optional[int] = None,
    field: Optional[str] = None,
    role: Optional[str] = None
):
    if l1 is None:
        _, l1 = calc_l1_from_string(word)
    if l2c is None:
        l2c, _, _, out_of_range = calc_l2c_from_l1(int(l1))
        if out_of_range or l2c is None:
            # Возвращаем минимальный результат при out_of_range
            return {
                'word': word,
                'sphere': sphere,
                'tone': tone,
                'allowed': allowed,
                'notes': notes or "",
                'l1': None,
                'l2c': None,
                'w': None,
                'C': None,
                'Hm': None,
                'Z': None,
                'field': field or "",
                'role': role or ""
            }

    w, C, Hm, Z = metrics(int(l1), int(l2c))

    notes_str = "" if notes is None else str(notes).strip()
    if notes_str.lower() == "nan":
        notes_str = ""

    return {
        "word": word,
        "sphere": sphere,
        "tone": tone,
        "allowed": parse_bool(allowed),
        "field": (field or "").strip(),
        "role": (role or "").strip(),
        "notes": notes_str,
        "l1": int(l1),
        "l2c": int(l2c),
        "w": float(w),
        "C": float(C),
        "Hm": float(Hm),
        "Z": float(Z),
    }


def soft_dedup(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    # гарантируем контракт колонок (LIB_COLS должен быть объявлен выше)
    for c in LIB_COLS:
        if c not in df.columns:
            df[c] = ""

    def uniq_notes(series, sep=" | "):
        """Собирает уникальные непустые значения и склеивает через sep."""
        out = []
        for x in series:
            s = "" if pd.isna(x) else str(x).strip()
            if s and s not in out:
                out.append(s)
        return sep.join(out)

    def pick_first_nonempty(series):
        """Берёт первое непустое значение из серии."""
        for x in series:
            s = "" if pd.isna(x) else str(x).strip()
            if s:
                return s
        return ""

    # Группируем по ключу дедупликации
    grouped = df.groupby(["word", "sphere", "tone"], as_index=False)
    
    deduped_rows = []
    
    for (word, sphere, tone), group in grouped:
        # Нормализуем word и пропускаем пустые
        word_norm = normalize(str(word).strip().upper()) if word else ""
        if not word_norm:
            continue
        
        # Собираем метаданные из группы
        allowed_vals = [parse_bool(x) for x in group["allowed"]]
        allowed = any(allowed_vals)  # True если хотя бы один True
        
        field = pick_first_nonempty(group["field"])
        role = pick_first_nonempty(group["role"])
        notes = uniq_notes(group["notes"], sep=" | ")
        
        # Пересчитываем метрики детерминированно из word
        row = force_recalc_row(
            word=word_norm,
            sphere=str(sphere).strip() if sphere else "прочее",
            tone=str(tone).strip() if tone else "neutral",
            allowed=allowed,
            notes=notes,
            field=field if field else None,
            role=role if role else None
        )
        
        deduped_rows.append(row)
    
    if not deduped_rows:
        return pd.DataFrame(columns=LIB_COLS)
    
    result_df = pd.DataFrame(deduped_rows)
    return result_df[LIB_COLS]


def import_json_library(file_obj):
    global LIB_DF
    try:
        # 1. Читаем JSON как раньше
        data = json.load(open(file_obj.name, "r", encoding="utf-8"))
        items = data["library"] if "library" in data else data

        # 2. Собираем новые строки
        rows = []
        for it in items:
            word = (it.get("word") or it.get("text") or "").strip().upper()
            if not word:
                continue
            sphere = (it.get("sphere") or "прочее").strip()
            tone   = (it.get("tone") or "neutral").strip()
            allowed = parse_bool(it.get("allowed", True))
            notes  = it.get("notes", "")
            field  = (it.get("field") or "").strip()
            role   = (it.get("role") or "").strip()
            l1     = it.get("l1", None)
            l2c    = it.get("l2c", None)

            row = force_recalc_row(
                word, sphere, tone, allowed, notes, l1, l2c,
                field=field, role=role
            )
            rows.append(row)


        df_new = pd.DataFrame(rows)
        df_new = df_new[LIB_COLS]
        df_new = soft_dedup(df_new)

        # 3. Подгружаем то, что уже есть (если надо)
        _ensure_lib_loaded()  # подгрузит ./spheres/sphere_*.csv в LIB_DF, если она пустая

        # 4. МЕРДЖ: если библиотека пуста → просто новые данные;
        #    если нет → конкатенируем и снова dedup
        if LIB_DF is None or LIB_DF.empty:
            merged = df_new
        else:
            merged = pd.concat([LIB_DF, df_new], ignore_index=True)
            merged = soft_dedup(merged)

        LIB_DF = merged.copy()
        rebuild_indexes(LIB_DF)

        if LIB_DF.empty:
            return "Файл прочитан, но записей не найдено.", gr.Dataframe(), ""

        summary = quality_summary(LIB_DF)
        return f"Импортировано новых слов: {len(df_new)}  |  Всего в библиотеке: {len(LIB_DF)}", summary, "Готово к сохранению в /spheres/ как CSV."
    except Exception as e:
        return f"Ошибка импорта: {e}", gr.Dataframe(), ""


def load_spheres_into_memory():
    global LIB_DF
    rows = []
    for path in glob.glob("./spheres/sphere_*.csv"):
        try:
            df = pd.read_csv(path, encoding="utf-8")
            for _, r in df.iterrows():
                rows.append(force_recalc_row(
                    word=str(r.get("word", "")).upper(),
                    sphere=str(r.get("sphere", "прочее")),
                    tone=str(r.get("tone", "neutral")),
                    allowed=parse_bool(r.get("allowed", True)),
                    notes=r.get("notes", ""),
                    l1=int(r.get("l1", 0)) if not pd.isna(r.get("l1", 0)) else None,
                    l2c=int(r.get("l2c", 0)) if not pd.isna(r.get("l2c", 0)) else None,
                    field=str(r.get("field", "")).strip(),
                    role=str(r.get("role", "")).strip(),
                ))

        except Exception:
            continue
    df = pd.DataFrame(rows)
    df = soft_dedup(df)
    LIB_DF = df.copy()
    rebuild_indexes(LIB_DF)
    if LIB_DF.empty:
        return "В папке /spheres/ данных не найдено.", gr.Dataframe()
    s = quality_summary(LIB_DF)
    return f"Загружено из spheres/: {len(LIB_DF)} слов", s

def get_all_spheres() -> List[str]:
    spheres = set()
    if LIB_DF is not None and not LIB_DF.empty:
        for s in LIB_DF["sphere"].astype(str).fillna("прочее"):
            for part in str(s).split(";"):
                part = part.strip()
                if part:
                    spheres.add(part)
    else:
        for path in glob.glob("./spheres/sphere_*.csv"):
            try:
                df = pd.read_csv(path, encoding="utf-8", usecols=["sphere"])
                for s in df["sphere"].astype(str).fillna("прочее"):
                    s = s.strip()
                    if s:
                        spheres.add(s)
            except Exception:
                continue
    out = sorted(spheres) if spheres else ["прочее"]
    return out

def _sphere_exact_match(cell: str, query: str) -> bool:
    """
    Строгое совпадение сферы:
    - делим значение по ';'
    - чистим пробелы
    - сравниваем по нижнему регистру
    """
    if not query:
        return False
    q = query.strip().lower()
    parts = [p.strip().lower() for p in str(cell).split(";")]
    return q in parts


def resolve_sphere(sphere_choice: str, create_new: bool, new_sphere: str) -> str:
    if create_new and new_sphere and new_sphere.strip():
        return new_sphere.strip()
    if sphere_choice and str(sphere_choice).strip():
        return str(sphere_choice).strip()
    return "прочее"

def _ensure_lib_loaded() -> Tuple[bool, str]:
    global LIB_DF
    if LIB_DF is None or (isinstance(LIB_DF, pd.DataFrame) and LIB_DF.empty):
        try:
            msg, _ = load_spheres_into_memory()
            return True, f"Auto-merge: {msg}"
        except Exception as e:
            return False, f"Auto-merge: не удалось загрузить из spheres/ ({type(e).__name__})"
    return False, "Auto-merge: в памяти уже есть библиотека"

def add_words_to_library(raw_text: str, sphere_choice: str, create_new: bool, new_sphere: str):
    global LIB_DF
    was_loaded, auto_msg = _ensure_lib_loaded()
    if not raw_text.strip():
        return "Вставьте слова.", gr.Dataframe(), gr.Markdown.update(value="")
    items = [normalize(x) for x in re.split(r"[\n,;]+", raw_text) if normalize(x)]
    if not items:
        return "Не найдено валидных слов (кириллица).", gr.Dataframe(), gr.Markdown.update(value="")
    sphere_name = resolve_sphere(sphere_choice, create_new, new_sphere)
    rows = []
    for w in items:
        rows.append(
            force_recalc_row(
                word=w,
                sphere=sphere_name,
                tone="neutral",
                allowed=True,
                notes=""
            )
        )
    df_new = pd.DataFrame(rows)
    if LIB_DF is None or LIB_DF.empty:
        merged = df_new
    else:
        merged = pd.concat([LIB_DF, df_new], ignore_index=True)
    merged = soft_dedup(merged)
    LIB_DF = merged.copy()
    rebuild_indexes(LIB_DF)
    base_msg = f"Добавлено слов: {len(rows)} (после dedup: {len(LIB_DF)})"
    if was_loaded:
        base_msg = f"{base_msg}\n{auto_msg}"
    return base_msg, df_new, quality_summary(LIB_DF)

def clusters_from_w(w: float) -> str:
    if 1.214 <= w <= 1.614:
        return "rt2"
    if 1.0   <= w < 1.6:
        return "phi"
    if 1.6   <= w < 2.7:
        return "e"
    if 2.7   <= w < 3.2:
        return "e-pi"
    if w >= 3.2:
        return "pi"
    return "phi"

def quality_summary(df: pd.DataFrame):
    if df is None or df.empty:
        return gr.Dataframe()
    d = df.copy()
    d["cluster"] = d["w"].apply(clusters_from_w)
    total = len(d)
    zone = d[(d["w"]>=1.6) & (d["w"]<=2.4)]
    edge = d[(d["w"]>4.0) | (d["w"]<0.7)]
    tmp = d.assign(sphere1=d["sphere"].str.split(";")).explode("sphere1")
    _top = (tmp.groupby("sphere1")
              .agg(cnt=("word","count"), Z_mean=("Z","mean"))
              .reset_index()
              .query("cnt >= 30")
              .sort_values(["Z_mean","cnt"], ascending=[False,False])
              .head(10))
    clusters = d["cluster"].value_counts().reindex(["phi","e","e-pi","pi","rt2"], fill_value=0)
    summary_tbl = pd.DataFrame({
        "metric": ["Всего","Зона 2±0.4 (%)","Edge (W<0.7 или >4.0)","phi","e","e–pi","pi","√2"],
        "value":  [total, round(len(zone)/total*100,1) if total else 0, len(edge),
                   clusters.get("phi",0), clusters.get("e",0), clusters.get("e-pi",0), clusters.get("pi",0), clusters.get("rt2",0)]
    })
    return summary_tbl

def filter_library_view(sphere_query:str, cluster_query:str, search:str):
    if LIB_DF is None or LIB_DF.empty:
        return gr.Dataframe()
    d = LIB_DF.copy()
    d["cluster"] = d["w"].apply(clusters_from_w)
    if sphere_query and sphere_query.strip():
        sq = sphere_query.strip().lower()
        d = d[d["sphere"].str.lower().str.contains(sq)]
    if cluster_query in {"phi","e","e-pi","pi","rt2"}:
        d = d[d["cluster"]==cluster_query]
    if search and search.strip():
        q = normalize(search)
        if q:
            d = d[d["word"].str.contains(q)]
    cols = ["word","sphere","tone","l1","l2c","w","C","Hm","Z","cluster","notes"]
    return d[cols].sort_values(["Z","w"], ascending=[False,True]).reset_index(drop=True)

def sha256_of_sources():
    hasher = hashlib.sha256()
    files = sorted(glob.glob("./spheres/sphere_*.csv"))
    if os.path.exists(PERSONAL_CSV):
        files.append(PERSONAL_CSV)
    for p in files:
        try:
            with open(p, "rb") as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    hasher.update(chunk)
        except Exception:
            continue
    return hasher.hexdigest()

def save_as_sphere_csvs():
    global LIB_DF
    try:
        if LIB_DF is None or LIB_DF.empty:
            return "Сначала импортируйте библиотеку (JSON) или «Загрузите из spheres/»."
        os.makedirs("./spheres", exist_ok=True)
        expanded_rows = []
        for _, r in LIB_DF.iterrows():
            spheres = [s for s in str(r.get("sphere", "")).split(";") if s.strip()]
            if not spheres:
                spheres = ["прочее"]
            for sph in spheres:
                rr = {
                    "word":   str(r.get("word", "")).upper(),
                    "sphere": sph,
                    "tone":   str(r.get("tone", "neutral")),
                    "allowed": parse_bool(r.get("allowed", True)),
                    "field":  str(r.get("field", "")).strip(),
                    "role":   str(r.get("role", "")).strip(),
                    "notes":  ("" if pd.isna(r.get("notes", "")) else str(r.get("notes", ""))),
                    "l1":     int(r.get("l1", 0)),
                    "l2c":    int(r.get("l2c", 0)),
                    "w":      float(r.get("w", 0.0)),
                    "C":      float(r.get("C", 0.0)),
                    "Hm":     float(r.get("Hm", 0.0)),
                    "Z":      float(r.get("Z", 0.0)),
                }
                expanded_rows.append(rr)
        df_expanded = pd.DataFrame(expanded_rows)
        saved_paths = []
        for sph, df in df_expanded.groupby("sphere"):
            slug = slugify(sph)
            path = f"./spheres/sphere_{slug}.csv"
            cols = ["word","sphere","tone","allowed","field","role","notes","l1","l2c","w","C","Hm","Z"]
            for c in cols:
                if c not in df.columns:
                    df[c] = None
            atomic_write_csv(df[cols], path)

            saved_paths.append(path)
        keep_path = "spheres/.keep"
        if not os.path.exists(keep_path):
            with open(keep_path, "wb") as k: k.write(b"keep")
        else:
            open(keep_path, "ab").close()
        saved_paths.append(keep_path)
        msg_commit = commit_ops(saved_paths, "Save sphere CSVs")
        return "✅ Сохранено в ./spheres/:\n" + "\n".join(f"- {p}" for p in saved_paths) + f"\n{msg_commit}"
    except Exception as e:
        return f"⚠️ Ошибка сохранения: {type(e).__name__}: {e}"

def compute_base_indicator() -> str:
    global LIB_DF
    if LIB_DF is None or LIB_DF.empty:
        return "📦 База пуста."
    total = len(LIB_DF)
    spheres_set = set()
    for s in LIB_DF["sphere"].astype(str).fillna(""):
        for part in s.split(";"):
            part = part.strip()
            if part:
                spheres_set.add(part)
    mean_w = LIB_DF["w"].mean() if not LIB_DF.empty else 0.0
    mean_z = LIB_DF["Z"].mean() if not LIB_DF.empty else 0.0
    return f"📦 Загружено слов: {total}  |  Сфер: {len(spheres_set)}  |  ⌀W = {mean_w:.2f}  |  ⌀Z = {mean_z:.2f}"

def compute_library_stats(df: Optional[pd.DataFrame]) -> dict:
    if df is None or df.empty:
        return {"count_words": 0, "num_spheres": 0, "w_mean": 0.0, "z_mean": 0.0}
    count_words = len(df)
    spheres = set()
    for s in df["sphere"].astype(str).fillna(""):
        for part in s.split(";"):
            part = part.strip()
            if part:
                spheres.add(part)
    w_mean = df["w"].astype(float).mean() if not df.empty else 0.0
    z_mean = df["Z"].astype(float).mean() if not df.empty else 0.0
    return {
        "count_words": count_words,
        "num_spheres": len(spheres),
        "w_mean": w_mean,
        "z_mean": z_mean
    }

def fmt_bar(value: float, n: int = 10) -> str:
    v = max(0.0, min(1.0, value))
    filled = int(round(v * n))
    return '▰' * filled + '▱' * (n - filled)

def axis_line_for_w(w: float) -> str:
    min_w, max_w = 1.0, 3.6
    width = 40
    w_clamped = min(max(w, min_w), max_w)
    pos = int(round((w_clamped - min_w) / (max_w - min_w) * width))
    def mark(val):
        return int(round((val - min_w) / (max_w - min_w) * width))
    m_phi = mark(1.6)
    m_e   = mark(2.7)
    m_epi = mark(3.2)
    axis_chars = ['─'] * (width + 1)
    for idx in (m_phi, m_e, m_epi):
        if 0 <= idx <= width:
            axis_chars[idx] = '│'
    if 0 <= pos <= width:
        axis_chars[pos] = '●'
    return ''.join(axis_chars)

def fmt_fractal_series(pattern: str) -> str:
    return pattern

def rebuild_indexes(df: pd.DataFrame):
    """
    Перестраивает индексы INDEX_L1 и INDEX_L2C из DataFrame.
    Вызывается после любых изменений LIB_DF.
    """
    global INDEX_L1, INDEX_L2C, INDEX_READY
    
    INDEX_L1.clear()
    INDEX_L2C.clear()
    
    if df is None or df.empty:
        INDEX_READY = False
        return
    
    try:
        for _, r in df.iterrows():
            word = str(r.get('word', '')).upper()
            if not word:
                continue
            
            # L1 индекс
            try:
                l1_val = r.get('l1')
                if l1_val is not None and not pd.isna(l1_val):
                    l1 = int(float(l1_val))
                    if l1 not in INDEX_L1:
                        INDEX_L1[l1] = []
                    if word not in INDEX_L1[l1]:
                        INDEX_L1[l1].append(word)
            except Exception:
                pass
            
            # L2C индекс
            try:
                l2c_val = r.get('l2c')
                if l2c_val is not None and not pd.isna(l2c_val):
                    l2c = int(float(l2c_val))
                    if l2c not in INDEX_L2C:
                        INDEX_L2C[l2c] = []
                    if word not in INDEX_L2C[l2c]:
                        INDEX_L2C[l2c].append(word)
            except Exception:
                pass
        
        INDEX_READY = True
    except Exception:
        INDEX_READY = False

def fmt_matches_by_code(l1: int, l2c: int, current_word: str, limit: int = 30) -> Tuple[str, str]:
    matches_l1 = []
    matches_l2c = []
    current_upper = current_word.upper()
    
    # Используем индексы для быстрого поиска
    if INDEX_READY:
        try:
            # L1 совпадения из индекса
            if l1 in INDEX_L1:
                matches_l1.extend([w for w in INDEX_L1[l1] if w != current_upper])
            
            # L2C совпадения из индекса
            if l2c in INDEX_L2C:
                matches_l2c.extend([w for w in INDEX_L2C[l2c] if w != current_upper])
        except Exception:
            pass
    
    # Персональная библиотека (как раньше)
    try:
        if os.path.exists(PERSONAL_CSV):
            with open(PERSONAL_CSV, 'r', encoding='utf-8') as f:
                rdr = csv.DictReader(f)
                for r in rdr:
                    word = str(r.get('text', '')).upper()
                    if word == current_upper:
                        continue
                    try:
                        if int(float(r.get('l1', 0))) == l1:
                            matches_l1.append(word)
                        if int(float(r.get('l2c', 0))) == l2c:
                            matches_l2c.append(word)
                    except Exception:
                        continue
    except Exception:
        pass
    
    def fmt_list(lst: List[str]) -> List[str]:
        unique = []
        for w in lst:
            if w not in unique:
                unique.append(w)
        return unique[:limit]
    
    unique_l1 = fmt_list(matches_l1)
    # Исключаем дубли между l1 и l2c
    unique_l2c = [w for w in fmt_list(matches_l2c) if w not in {uw.upper() for uw in unique_l1}]
    
    def to_str(lst: List[str]) -> str:
        return ' · '.join(lst) if lst else '—'
    
    return to_str(unique_l1), to_str(unique_l2c)

# -------------------------
#  Поиск совпадений по кодам в персональной библиотеке
# -------------------------
def fmt_matches_personal_by_code(l1: int, l2c: int, current_word: str, limit: int = 30) -> Tuple[str, str]:
    """
    Ищет совпадения по L1 и L2C только в персональной CSV-базе.
    Возвращает списки слов для L1 и L2C (до limit элементов).
    current_word исключается из результатов.
    """
    matches_l1: List[str] = []
    matches_l2c: List[str] = []
    try:
        if os.path.exists(PERSONAL_CSV):
            with open(PERSONAL_CSV, 'r', encoding='utf-8') as f:
                rdr = csv.DictReader(f)
                for r in rdr:
                    word = str(r.get('text', '')).upper()
                    # пропускаем текущее слово
                    if word == current_word.upper():
                        continue
                    try:
                        # сравниваем l1 и l2c через float/ int, поскольку значения в CSV могут быть float-подобные
                        if int(float(r.get('l1', 0))) == l1:
                            matches_l1.append(word)
                        if int(float(r.get('l2c', 0))) == l2c:
                            matches_l2c.append(word)
                    except Exception:
                        continue
    except Exception:
        pass
    # уникализируем и ограничиваем
    def fmt_list(lst: List[str]) -> List[str]:
        out: List[str] = []
        for w in lst:
            if w not in out:
                out.append(w)
        return out[:limit]
    unique_l1 = fmt_list(matches_l1)
    # Исключаем дубли между l1 и l2c, оставляя уникальные для l2c
    unique_l2c = [w for w in fmt_list(matches_l2c) if w not in {uw.upper() for uw in unique_l1}]
    def to_str(lst: List[str]) -> str:
        return ' · '.join(lst) if lst else '—'
    return to_str(unique_l1), to_str(unique_l2c)

def fmt_near_far_words(res: Dict[str, Any], limit_near: int = 5, limit_contrast: int = 5) -> Tuple[str, str]:
    if LIB_DF is None or LIB_DF.empty:
        return '—', '—'
    
    try:
        # Векторизованный расчёт расстояний
        df = LIB_DF[['word', 'w', 'C', 'Z']].copy()
        df['word'] = df['word'].astype(str).str.upper()
        df['w'] = pd.to_numeric(df['w'], errors='coerce')
        df['C'] = pd.to_numeric(df['C'], errors='coerce')
        df['Z'] = pd.to_numeric(df['Z'], errors='coerce')
        df = df.dropna(subset=['w', 'C', 'Z', 'word'])
        
        # Исключаем текущее слово
        current = str(res['norm']).upper()
        df = df[df['word'] != current]
        
        if df.empty:
            return '—', '—'
        
        # Векторизованный расчёт расстояний
        w0, C0, Z0 = float(res['w']), float(res['C']), float(res['Z'])
        df['D'] = ((df['w'] - w0)**2 + (df['C'] - C0)**2 + (df['Z'] - Z0)**2).pow(0.5)
        
        # Сортируем по расстоянию и удаляем дубли по 'word' (сохраняем первое вхождение = ближайшее)
        df = df.sort_values('D').drop_duplicates(subset=['word'], keep='first')
        
        # Near: D <= 0.30
        near_df = df[df['D'] <= 0.30].head(limit_near)
        near_words = set(near_df['word'].tolist())
        near_list = [f"{w} ({d:.2f})" for w, d in zip(near_df['word'], near_df['D'])]
        
        # Contrast: 0.20 < D <= 1.00, исключаем слова из near
        contrast_df = df[(df['D'] > 0.20) & (df['D'] <= 1.00) & (~df['word'].isin(near_words))].head(limit_contrast)
        contrast_list = [f"{w} ({d:.2f})" for w, d in zip(contrast_df['word'], contrast_df['D'])]
        
        return (' · '.join(near_list) if near_list else '—', ' · '.join(contrast_list) if contrast_list else '—')
    except Exception:
        return '—', '—'

def _collect_matches_by_code(res: Dict[str, Any], limit_l1: int = 500, limit_l2c: int = 500) -> Tuple[List[str], List[str]]:
    """
    Возвращает списки слов (в верхнем регистре) с тем же L1 и тем же L2C
    из общей библиотеки (LIB_DF) и персональной, исключая текущее слово.
    Использует индексы для ускорения.
    """
    l1, l2c, current = int(res['l1']), int(res['l2c']), str(res['norm']).upper()
    seen_l1, seen_l2c = set(), set()
    out_l1, out_l2c = [], []

    # LIB - используем индексы для быстрого поиска
    if INDEX_READY:
        try:
            # L1 совпадения из индекса
            if l1 in INDEX_L1:
                for w in INDEX_L1[l1]:
                    if w != current and w not in seen_l1:
                        out_l1.append(w)
                        seen_l1.add(w)
                        if len(out_l1) >= limit_l1:
                            break
            
            # L2C совпадения из индекса
            if l2c in INDEX_L2C:
                for w in INDEX_L2C[l2c]:
                    if w != current and w not in seen_l2c:
                        out_l2c.append(w)
                        seen_l2c.add(w)
                        if len(out_l2c) >= limit_l2c:
                            break
        except Exception:
            pass

    # PERSONAL
    try:
        if os.path.exists(PERSONAL_CSV):
            with open(PERSONAL_CSV, 'r', encoding='utf-8') as f:
                rdr = csv.DictReader(f)
                for r in rdr:
                    w = str(r.get('text', '')).upper()
                    if not w or w == current:
                        continue
                    try:
                        if int(float(r.get('l1', 0))) == l1 and w not in seen_l1:
                            out_l1.append(w); seen_l1.add(w)
                        if int(float(r.get('l2c', 0))) == l2c and w not in seen_l2c:
                            out_l2c.append(w); seen_l2c.add(w)
                    except Exception:
                        continue
    except Exception:
        pass

    return out_l1[:limit_l1], out_l2c[:limit_l2c]


def _collect_near_contrast(res: Dict[str, Any],
                           limit_near: int = 50,
                           limit_contrast: int = 50) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Возвращает списки слов из LIB_DF со структурой:
    [{'word': '...', 'D': 0.23, 'W': 1.88, 'C': 0.73, 'Z': 0.41}, ...]
    Векторизованная версия для ускорения.
    """
    if LIB_DF is None or LIB_DF.empty:
        return [], []
    
    try:
        # Векторизованный расчёт расстояний
        df = LIB_DF[['word', 'w', 'C', 'Z']].copy()
        df['word'] = df['word'].astype(str).str.upper()
        df['w'] = pd.to_numeric(df['w'], errors='coerce')
        df['C'] = pd.to_numeric(df['C'], errors='coerce')
        df['Z'] = pd.to_numeric(df['Z'], errors='coerce')
        df = df.dropna(subset=['w', 'C', 'Z', 'word'])
        
        # Исключаем текущее слово
        current = str(res['norm']).upper()
        df = df[df['word'] != current]
        
        if df.empty:
            return [], []
        
        # Векторизованный расчёт расстояний
        w0, C0, Z0 = float(res['w']), float(res['C']), float(res['Z'])
        df['D'] = ((df['w'] - w0)**2 + (df['C'] - C0)**2 + (df['Z'] - Z0)**2).pow(0.5)
        
        # Сортируем по расстоянию и удаляем дубли по 'word' (сохраняем первое вхождение = ближайшее)
        df = df.sort_values('D').drop_duplicates(subset=['word'], keep='first')
        
        # Near: D <= 0.30
        near_df = df[df['D'] <= 0.30].head(limit_near)
        near_words = set(near_df['word'].tolist())
        near = [
            {
                'word': w,
                'D': round(d, 3),
                'W': round(w_val, 3),
                'C': round(c_val, 3),
                'Z': round(z_val, 3)
            }
            for w, d, w_val, c_val, z_val in zip(
                near_df['word'], near_df['D'], near_df['w'], near_df['C'], near_df['Z']
            )
        ]
        
        # Contrast: 0.20 < D <= 1.00, исключаем слова из near
        contrast_df = df[(df['D'] > 0.20) & (df['D'] <= 1.00) & (~df['word'].isin(near_words))].head(limit_contrast)
        contrast = [
            {
                'word': w,
                'D': round(d, 3),
                'W': round(w_val, 3),
                'C': round(c_val, 3),
                'Z': round(z_val, 3)
            }
            for w, d, w_val, c_val, z_val in zip(
                contrast_df['word'], contrast_df['D'], contrast_df['w'], contrast_df['C'], contrast_df['Z']
            )
        ]
        
        return near, contrast
    except Exception:
        return [], []
# <<< PATCH

# >>> PATCH: fallback for FA — neighborhood-as-codes
def _fa_neighborhood_words(res: Dict[str, Any], eps: float = 0.02, limit: int = 50) -> List[str]:
    """
    Возвращает до limit слов из LIB_DF с |W - W0| <= eps.
    Используется как "мягкая" подстановка в by_L1/by_L2C для FA-режима,
    если точных совпадений по кодам нет.
    """
    if LIB_DF is None or (isinstance(LIB_DF, pd.DataFrame) and LIB_DF.empty):
        return []
    try:
        df = LIB_DF.copy()
        df["w"] = pd.to_numeric(df["w"], errors="coerce")
        df = df.dropna(subset=["w"])
    except Exception:
        return []

    w0 = float(res.get("w", 0.0))
    current = str(res.get("norm", "")).upper()
    cand = df[(df["w"].sub(w0).abs() <= eps)]

    words_raw: List[str] = []
    for _, r in cand.iterrows():
        w = str(r.get("word", "")).upper()
        if w and w != current:
            words_raw.append(w)

    # Уникальные, с сохранением порядка
    seen = set()
    out: List[str] = []
    for w in words_raw:
        if w not in seen:
            out.append(w); seen.add(w)
        if len(out) >= limit:
            break
    return out
# <<< PATCH

# =========================
#  Интерфейс Gradio
# =========================
with gr.Blocks(css=CUSTOM_CSS) as demo:
    gr.Markdown("# Quantum Encoder")
    status_env = gr.Markdown(value=f"**Repo:** `{SPACE_REPO_ID or '—'}`  |  **HF_TOKEN:** {'✅' if HF_TOKEN else '—'}  |  **Contract:** {ENCODER_VERSION}  |  **Calc:** {CALC_VERSION}")
    with gr.Tabs():
        # ---- Расчёт слова ----
        with gr.Tab("Расчёт слова"):
            gr.Markdown("## Анализ слова")
            with gr.Row():
                inp1 = gr.Textbox(label="Слово", placeholder="например: ГАРМОНИЯ", lines=1)
            with gr.Row():
                mode = gr.Radio(choices=["Слово", "FractalAvatar"], value="Слово", label="Режим ввода")

            with gr.Row(visible=False) as fa_row:
                fa_W   = gr.Number(label="W",  precision=3)
                fa_C   = gr.Number(label="C",  precision=3)
                fa_Hm  = gr.Number(label="Hm", precision=3)
                fa_Z   = gr.Number(label="Z",  precision=3)
                fa_Phi = gr.Number(label="Φ (опционально)", precision=3)

            def _toggle_fa(r):
                return gr.update(visible=(r=="FractalAvatar"))
            mode.change(_toggle_fa, inputs=mode, outputs=fa_row)

            with gr.Row():
                btn_calc  = gr.Button("Рассчитать", variant="primary")
            ver_info = gr.Markdown(value=f"Версия ядра Kryon Encoder {ENCODER_VERSION} | Формулы {CALC_VERSION}")
            # выводы анализа
            passport_md = gr.HTML()
            visual_md   = gr.Markdown()
            fractal_md  = gr.Markdown()
            resonance_md= gr.Markdown()
            advice_md   = gr.Markdown()
            dl_btn      = gr.DownloadButton(label="📦 Скачать JSON расчёта", value=None)
            dl_btn_full = gr.DownloadButton(label="📦 Скачать FULL JSON (+связанные)", value=None)
            add_btn_an  = gr.Button("➕ Добавить в персональную библиотеку")
            personal_status = gr.Markdown()
            base_indicator = gr.Markdown()

            # функция расчёта одного слова / FA
            def on_calc(w1, mode_val, W_in, C_in, Hm_in, Z_in, Phi_in):
                _ensure_lib_loaded()
                if mode_val == "FractalAvatar":
                    try:
                        Wv = float(W_in); Cv = float(C_in); Hmv = float(Hm_in); Zv = float(Z_in)
                    except Exception:
                        return ("Укажи корректные W, C, Hm, Z для FA.", "", "", "", "", None, None)
                    res1 = analyze_from_fa("FA input", Wv, Cv, Hmv, Zv, Phi_in)
                else:
                    res1 = analyze_word(w1 or "")
                if not res1:
                    return ("Введите слово для расчёта.", "", "", "", "", None, None)

                # сохраняем для добавления в библиотеку
                LAST_RESULT.clear()
                LAST_RESULT.update({
                    "input": (res1.get("raw") if res1.get("fa_mode") else w1),
                    "phrase_used": res1['phrase_used'],
                    "l1": res1['l1'],
                    "l2c": res1['l2c'],
                    "w": res1['w'],
                    "C": res1['C'],
                    "Hm": res1['Hm'],
                    "Z": res1['Z']
                })
                # начальный импульс
                fi_t, fi_d, fi_m = res1['first_impulse']
                first_char = res1['first_char'] or "—"
                first_val = res1['first_val'] or "—"
                impulse_section = []
                if not res1.get('fa_mode'):
                    fi_t, fi_d, fi_m = res1['first_impulse']
                    first_char = res1['first_char'] or "—"
                    first_val = res1['first_val'] or "—"
                    impulse_section.append('<div class="section-heading">&gt; Начальный импульс</div>')
                    impulse_section.append(f"<b>Первая буква:</b> {first_char} → код: {first_val}")
                    impulse_section.append(f"Тип: <b>{fi_t or '—'}</b> — {fi_d or '—'} · {fi_m or '—'}")


                # основные показатели
                cluster_label = {
                    'phi': 'φ-ядро',
                    'e': 'e',
                    'e-pi': 'e–π',
                    'pi': 'π (турбулентность)'
                }.get(res1['cluster_code'], res1['cluster_ru'])
                basics = []
                basics.append('<div class="section-heading">&gt; Основные показатели</div>')
                # используем абзацы <p> для каждой строки, чтобы обеспечить устойчивые отступы
                basics.append(
                    f"<p><b>L1 = {res1['l1']}</b>  <b>L2C = {res1['l2c']}</b>  <b>W = {res1['w']:.3f}</b> — {cluster_label}</p>"
                )
                basics.append(
                    f"<p><b>Q_total = {res1['q_total']:.2f}</b> — согласованность формы и смысла</p>"
                )
                basics.append(
                    f"<p><b>FII = {res1['fii']:+.1f}</b> {res1['fii_bar']}</p>"
                )
                # FA-бейдж (для явного признака режима)
                if res1.get('fa_mode'):
                    phi_str = f"{res1.get('Phi_align'):.2f}" if res1.get('Phi_align') is not None else "—"
                    basics.append(
                        f"<p><b>FA-mode:</b> W={res1['w']:.3f}, C={res1['C']:.3f}, Hm={res1['Hm']:.3f}, Z={res1['Z']:.3f}, Φ={phi_str}</p>"
                    )

                # категория FII: извлекаем название и описание
                fii_label, fii_desc_full = parse_fii_category_str(res1['fii_category'])
                if fii_label:
                    basics.append(f"<p><b>{fii_label}</b> — {fii_desc_full}</p>")

                # Удаляем ось кластеров и пояснение, чтобы отчёт был короче

                # особые метрики (бывшие визуальные)
                visuals = []
                visuals.append('<div class="section-heading">&gt; Особые метрики</div>')
                visuals.append(
                    f"<p><b>Музыкальность (Hm):</b> {fmt_bar(res1['Hm'])} ({res1['Hm']:.2f})</p>"
                )
                visuals.append(
                    f"<p><b>Когерентность (C):</b> {fmt_bar(res1['C'])} ({res1['C']:.2f})</p>"
                )
                visuals.append(
                    f"<p><b>Интегральная гармония (Z):</b> {fmt_bar(res1['Z'])} ({res1['Z']:.2f})</p>"
                )
                visuals.append(
                    f"<p><b>Согласованность формы и смысла (Q_total):</b> {fmt_bar(res1['q_total'])} ({res1['q_total']:.2f})</p>"
                )
                visuals.append('<div class="small-note">Музыкальность — насколько слово «поёт», его ритмика и плавность звучания.<br>Когерентность — согласованность частей слова, отсутствие внутренних конфликтов.<br>Интегральная гармония — общая собранность и энергетическая цельность слова.<br>Согласованность формы и смысла (Q_total) — насколько форма слова отражает его внутренний смысл.</div>')

                # фрактальная развёртка (только в режиме Слова)
                fractal_lines = []
                if not res1.get('fa_mode'):
                    pattern = fmt_fractal_series(res1['fractal_pattern'])
                    inh_bars = '■' * res1['fractal_inhale'] + '□' * (12 - res1['fractal_inhale'])
                    exh_bars = '■' * res1['fractal_exhale'] + '□' * (12 - res1['fractal_exhale'])
                    fractal_lines.append('<div class="section-heading">&gt; Фрактальная развёртка (12 измерений)</div>')
                    fractal_lines.append(f"<p>φ ядро π турбулентность |{pattern}|</p>")
                    fractal_lines.append("<br>")
                    fractal_lines.append(f"<p><b>Вдох</b> [{inh_bars}] {res1['fractal_inhale']}</p>")
                    fractal_lines.append(f"<p><b>Выдох</b> [{exh_bars}] {res1['fractal_exhale']}</p>")
                    fractal_lines.append("<br>")
                    fractal_lines.append(f"<p><b>R = {res1['fractal_R']:.2f}</b> → {res1['fractal_interp']}</p>")

                resonance_lines = []
                if res1.get('res_pair_code'):
                    resonance_lines.append('<div class="section-heading">&gt; Резонансная пара</div>')
                    resonance_lines.append(f"<b>Resonance Pair:</b> {res1['res_pair_code']} ({res1['res_pair_en']})")
                    # resonance_lines.append(f"R_pair = {res1.get('res_pair_value', 0):.3f}")  # опционально
                    resonance_lines.append(res1['res_pair_ru'])


                # резонансное пространство
                res_space = []
                res_space.append('<div class="section-heading">&gt; Резонансное пространство</div>')
                res_space.append(f"<b>Rφ</b> = {res1['R_phi']:.3f} · <b>Re</b> = {res1['R_e']:.3f} · <b>Rπ</b> = {res1['R_pi']:.3f} · <b>R√2</b> = {res1['R_rt2']:.3f}")
                label, val = res1['resonator_max']
                res_space.append(f"Максимум: <b>{label}</b> ({val:.3f})")

                # совпадения по кодам (отображаем только в режиме Слова)
                pers_matches_lines = []
                matches_lines = []
                if not res1.get('fa_mode'):
                    # персональная база
                    pers_match_l1, pers_match_l2c = fmt_matches_personal_by_code(res1['l1'], res1['l2c'], res1['norm'], limit=30)
                    pers_matches_lines.append('<div class="section-heading">&gt; Совпадения по персональной библиотеке</div>')
                    pers_matches_lines.append(f"<p><b>L1</b> = {res1['l1']} → {pers_match_l1}</p>")
                    pers_matches_lines.append("<br>")
                    pers_matches_lines.append(f"<p><b>L2C</b> = {res1['l2c']} → {pers_match_l2c}</p>")

                    # общая библиотека (+персональная)
                    match_l1, match_l2c = fmt_matches_by_code(res1['l1'], res1['l2c'], res1['norm'], limit=30)
                    matches_lines.append('<div class="section-heading">&gt; Совпадения по кодам</div>')
                    matches_lines.append(f"<p><b>L1</b> = {res1['l1']} → {match_l1}</p>")
                    matches_lines.append("<br>")
                    matches_lines.append(f"<p><b>L2C</b> = {res1['l2c']} → {match_l2c}</p>")


                # совпадения по кодам (общая библиотека + персональная)
                match_l1, match_l2c = fmt_matches_by_code(res1['l1'], res1['l2c'], res1['norm'], limit=30)
                matches_lines = []
                matches_lines.append('<div class="section-heading">&gt; Совпадения по кодам</div>')
                matches_lines.append(f"<p><b>L1</b> = {res1['l1']} → {match_l1}</p>")
                matches_lines.append("<br>")
                matches_lines.append(f"<p><b>L2C</b> = {res1['l2c']} → {match_l2c}</p>")

                # созвучия/контрасты
                near, contrast = fmt_near_far_words(res1)
                harmony_lines = []
                harmony_lines.append('<div class="section-heading">&gt; Созвучные и контрастные</div>')
                harmony_lines.append(f"<p><b>СОЗВУЧИЯ:</b> {near}</p>")
                harmony_lines.append("<br>")
                harmony_lines.append(f"<p><b>КОНТРАСТЫ:</b> {contrast}</p>")

                # глубинная интерпретация (отключено — секция не используется)
                interp_lines = []

                # совет
                advice_lines = []
                advice_lines.append('<div class="section-heading">&gt; Психогеометрический совет</div>')
                advice_map2 = {
                    'phi': "Добавь e-слова (ПУТЬ, ДВИЖЕНИЕ, ПРОЦЕСС). Для углубления — √2-слова (ЗЕРКАЛО, ОТРАЖЕНИЕ).",
                    'e':   "Дополни φ-словами (СПОКОЙ, РАВНОВЕСИЕ)…",
                    'e-pi':"Сильный всплеск. Соедини с φ/√2, чтобы не увести в турбулентность…",
                    'pi':  "Снизь напряжение через φ и повышение Z: комбинации со словами покоя/симметрии…"
                }
                advice_text2 = advice_map2.get(res1['cluster_code'], '—')
                if res1['fii'] <= -6:
                    advice_text2 += " Повышенная дестабилизация поля — использовать с осторожностью."
                advice_lines.append(advice_text2)

                # цифровая подпись
                signature_lines = []
                signature_lines.append('<div class="section-heading">&gt; Цифровая подпись слова</div>')
                signature_lines.append(
                    f"Φ {res1['w']:.3f} | C {res1['C']:.2f} | Hm {res1['Hm']:.2f} | Z {res1['Z']:.2f} | Q {res1['q_total']:.2f} | FII {res1['fii']:+.1f}"
                )

                # формируем итоговый отчёт. Порядок разделов: основные, импульс, особые метрики, резонансная пара, резонансное пространство, фрактал, совет, совпадения, созвучия/контрасты, подпись
                sections = [
                    "\n".join(basics),
                    "\n".join(impulse_section),
                    "\n".join(visuals),
                    "\n".join(resonance_lines),
                    "\n".join(res_space),
                    "\n".join(fractal_lines),
                    "\n".join(advice_lines),
                    "\n".join(pers_matches_lines),
                    "\n".join(matches_lines),
                    "\n".join(harmony_lines),
                    "\n".join(signature_lines)
                ]
                full_report = '<div class="report-body">' + "\n\n".join(sections) + '</div>'
                path_json = build_json_report(res1)
                path_full = build_full_json_report(res1)
                return (full_report, "", "", "", "", path_json, path_full)
            # обработчики
            btn_calc.click(
                on_calc,
                inputs=[inp1, mode, fa_W, fa_C, fa_Hm, fa_Z, fa_Phi],
                outputs=[passport_md, visual_md, fractal_md, resonance_md, advice_md, dl_btn, dl_btn_full]
            )
            inp1.submit(
                on_calc,
                inputs=[inp1, mode, fa_W, fa_C, fa_Hm, fa_Z, fa_Phi],
                outputs=[passport_md, visual_md, fractal_md, resonance_md, advice_md, dl_btn, dl_btn_full]
            )
            add_btn_an.click(
                add_to_personal,
                inputs=None,
                outputs=[personal_status, base_indicator]
            )
        # ---- Фраза ----
        with gr.Tab("Фраза"):
            gr.Markdown("### Фразовый анализатор (визуал)\nВставь фразу (50–120 слов) **или** список слов столбиком. Даты `ДД.ММ.ГГГГ` распознаются.")
            phrase_inp = gr.Textbox(label="Фраза / слова столбиком", lines=8, placeholder="Пример: Свет по воде, горизонт ровный, внутри — покой и ожидание.\n21.06.1992")
            run_phrase = gr.Button("Рассчитать фразу", variant="primary")
            phrase_summary = gr.Markdown()
            phrase_table = gr.Dataframe(interactive=False)
            export_btn = gr.Button("Export JSON")
            export_status = gr.Markdown()
            dl_phrase = gr.DownloadButton(label="Скачать phrase_report.json", value=None)
            phrase_state = gr.State(value=None)
            def _on_phrase_calc(text):
                df, summary_md, data_json = analyze_phrase(text or "")
                return summary_md, df, data_json
            def _on_phrase_export(data_json):
                path, msg = export_phrase_json(data_json) if data_json else export_phrase_json(
                    {"error":"Нет данных для экспорта. Сначала рассчитайте фразу.",
                     "meta":{"encoder":"Kryon-33","version":ENCODER_VERSION,"generated_at":datetime.datetime.utcnow().isoformat()+"Z"}})
                return msg, path
            run_phrase.click(_on_phrase_calc, inputs=[phrase_inp], outputs=[phrase_summary, phrase_table, phrase_state])
            export_btn.click(_on_phrase_export, inputs=[phrase_state], outputs=[export_status, dl_phrase])
            phrase_base_indicator = gr.Markdown()
        # ---- Библиотека ----
        with gr.Tab("Библиотека"):
            gr.Markdown("### Вставить слова (CSV/список) → в библиотеку")
            words_box = gr.Textbox(label="Слова (по одной на строку или через запятую)", lines=8, placeholder="пример:\nГАРМОНИЯ\nРАВНОВЕСИЕ\nТИШИНА")
            sphere_dd = gr.Dropdown(label="Сфера", choices=[], value=None)
            refresh_spheres_btn = gr.Button("Обновить список сфер", variant="secondary")
            create_new_cb = gr.Checkbox(label="Создать новую сферу", value=False)
            new_sphere_tb = gr.Textbox(label="Новая сфера", visible=False)
            add_words_btn = gr.Button("Добавить в библиотеку")
            def _toggle_new(checked: bool):
                return gr.update(visible=checked)
            create_new_cb.change(_toggle_new, inputs=create_new_cb, outputs=new_sphere_tb)
            def _ui_get_spheres():
                return gr.update(choices=get_all_spheres(), value=None)
            refresh_spheres_btn.click(_ui_get_spheres, inputs=None, outputs=[sphere_dd])
            add_words_status = gr.Markdown()
            add_words_table = gr.Dataframe(interactive=False)
            quality_tbl = gr.Dataframe(label="Диагностика качества", interactive=False)
            gr.Markdown("### Импорт / Загрузка / Сохранение")
            with gr.Row():
                up = gr.File(label="Импорт JSON", file_types=[".json"])
                load_btn = gr.Button("Загрузить из spheres/")
                save_btn = gr.Button("Сохранить CSV по сферам в /spheres/")
            imp_status = gr.Markdown()
            save_status = gr.Markdown()
            gr.Markdown("### Быстрые фильтры / Поиск")
            with gr.Row():
                q_sphere = gr.Textbox(label="Фильтр по сфере (подстрока)", placeholder="например: эзотерика", scale=2)
                q_cluster = gr.Dropdown(choices=["","phi","e","e-pi","pi","rt2"], label="Кластер W", value="", scale=1)
                q_search = gr.Textbox(label="Поиск по слову", placeholder="введи слово или его часть", scale=2)
                refresh_view = gr.Button("Показать", variant="secondary", scale=1)
            lib_view = gr.Dataframe(interactive=False)
            def _handle_add_words(text, sphere_choice, create_new, new_sphere):
                msg, df_new, q = add_words_to_library(text, sphere_choice, create_new, new_sphere)
                return msg, df_new, q
            add_words_btn.click(
                _handle_add_words,
                inputs=[words_box, sphere_dd, create_new_cb, new_sphere_tb],
                outputs=[add_words_status, add_words_table, quality_tbl]
            )
            def handle_import(file):
                if file is None:
                    return "Выберите JSON-файл для импорта.", gr.Dataframe(), "", gr.update()
                try:
                    msg, q_tbl, save_hint = import_json_library(file)
                    return msg, q_tbl, save_hint, gr.update(choices=get_all_spheres(), value=None)
                except Exception as e:
                    return f"Ошибка импорта: {type(e).__name__}: {e}", gr.Dataframe(), "", gr.update()
            up.change(handle_import, inputs=up, outputs=[imp_status, quality_tbl, save_status, sphere_dd])
            def handle_load_spheres():
                msg, q_tbl = load_spheres_into_memory()
                return msg, q_tbl, gr.update(choices=get_all_spheres(), value=None)
            load_btn.click(handle_load_spheres, inputs=None, outputs=[imp_status, quality_tbl, sphere_dd])
            save_btn.click(
                lambda: (save_as_sphere_csvs(), gr.update(choices=get_all_spheres(), value=None)),
                inputs=None,
                outputs=[save_status, sphere_dd]
            )
            def handle_refresh(sph, cl, srch):
                return filter_library_view(sph, cl, srch)
            refresh_view.click(handle_refresh, inputs=[q_sphere, q_cluster, q_search], outputs=[lib_view])
            library_base_indicator = gr.Markdown()

        # ---- Рекалибровка гармонии ядра ----
            gr.Markdown("### 🔧 Рекалибровка гармонии ядра")

            with gr.Row():
                sigma_inp = gr.Number(label="sigma_Z", value=float(APP_CFG.get("sigma_Z", 0.8)), precision=2)
                thr_inp   = gr.Number(label="resonator_threshold", value=float(APP_CFG.get("resonator_threshold", 0.75)), precision=2)

            bounds_inp = gr.Textbox(
                label="cluster_bounds (JSON)",
                lines=4,
                value=json.dumps(APP_CFG.get("cluster_bounds", {}), ensure_ascii=False)
            )

            with gr.Row():
                btn_reload_cfg  = gr.Button("Перечитать config.json", variant="secondary")
                btn_save_cfg    = gr.Button("Сохранить локально (без коммита)", variant="secondary")
                btn_commit_cfg  = gr.Button("Сохранить и зафиксировать (commit)", variant="primary")
            with gr.Row():
                btn_recalc_personal = gr.Button("Пересчитать персональную базу по текущему конфигу", variant="secondary")
                btn_reset_defaults  = gr.Button("Откатить к дефолту", variant="secondary")

            cfg_status = gr.Markdown()

            def _ui_reload_cfg():
                global APP_CFG
                APP_CFG = load_config()
                return (
                float(APP_CFG.get("sigma_Z", 0.8)),
                float(APP_CFG.get("resonator_threshold", 0.75)),
                json.dumps(APP_CFG.get("cluster_bounds", {}), ensure_ascii=False),
                "🔄 Конфигурация перечитана."
                )

            def _ui_save_cfg(sigma, thr, bounds_text):
                try:
                    bounds = json.loads(bounds_text) if bounds_text.strip() else APP_CFG.get("cluster_bounds", {})
                except Exception as e:
                    return gr.update(), gr.update(), gr.update(), f"⚠️ Ошибка JSON в cluster_bounds: {e}"
                ok, msg = set_cfg_values(sigma, thr, bounds)
                return sigma, thr, json.dumps(APP_CFG.get("cluster_bounds", {}), ensure_ascii=False), msg

            def _ui_commit_cfg(sigma, thr, bounds_text):
                s, t, b, msg = _ui_save_cfg(sigma, thr, bounds_text)
                commit_msg = commit_config("Update config.json (UI)")
                return s, t, b, f"{msg}\n{commit_msg}"

            def _ui_recalc_personal():
                if not os.path.exists(PERSONAL_CSV):
                    return "Персональная база отсутствует — пересчитывать нечего."
                try:
                    df = pd.read_csv(PERSONAL_CSV, encoding="utf-8")
                    if df.empty:
                        return "Персональная база пуста."
                # пересчёт метрик на основе текущего APP_CFG (metrics уже берет sigma из APP_CFG)
                    for idx, r in df.iterrows():
                        try:
                            l1 = int(float(r.get("l1", 0)))
                            l2c = int(float(r.get("l2c", 0)))
                            w, C, Hm, Z = metrics(l1, l2c)
                            df.at[idx, "w"]  = float(f"{w:.6f}")
                            df.at[idx, "C"]  = float(f"{C:.6f}")
                            df.at[idx, "Hm"] = float(f"{Hm:.6f}")
                            df.at[idx, "Z"]  = float(f"{Z:.6f}")
                        except Exception:
                            continue
                    atomic_write_csv(df, PERSONAL_CSV)
                    return "✅ Пересчёт персональной базы выполнен (w, C, Hm, Z обновлены)."
                except Exception as e:
                    return f"⚠️ Ошибка пересчёта: {type(e).__name__}: {e}"

            def _ui_reset_defaults():
                ok, msg = reset_to_defaults()
                return (
                    float(APP_CFG.get("sigma_Z", 0.8)),
                    float(APP_CFG.get("resonator_threshold", 0.75)),
                    json.dumps(APP_CFG.get("cluster_bounds", {}), ensure_ascii=False),
                    msg
                )

            btn_reload_cfg.click(_ui_reload_cfg, inputs=None, outputs=[sigma_inp, thr_inp, bounds_inp, cfg_status])
            btn_save_cfg.click(_ui_save_cfg, inputs=[sigma_inp, thr_inp, bounds_inp], outputs=[sigma_inp, thr_inp, bounds_inp, cfg_status])
            btn_commit_cfg.click(_ui_commit_cfg, inputs=[sigma_inp, thr_inp, bounds_inp], outputs=[sigma_inp, thr_inp, bounds_inp, cfg_status])
            btn_recalc_personal.click(lambda: _ui_recalc_personal(), inputs=None, outputs=[cfg_status])
            btn_reset_defaults.click(_ui_reset_defaults, inputs=None, outputs=[sigma_inp, thr_inp, bounds_inp, cfg_status])

        # ---- Экспорт ---- (новая вкладка для выгрузки данных библиотеки)
        with gr.Tab("Экспорт"):
            gr.Markdown("### Экспорт библиотеки")
            # Выбор источника: вся библиотека (сферы+персональная) или только персональная база
            export_source = gr.Radio(
                choices=["Вся библиотека", "Только персональная библиотека"],
                value="Вся библиотека",
                label="Источник данных"
            )
            export_btn = gr.Button("Сформировать JSON", variant="primary")
            export_status = gr.Markdown()
            export_dl = gr.DownloadButton(label="Скачать JSON", value=None)

            # ----------- Экспорт по сфере -----------
            gr.Markdown("#### Экспорт по сфере")

            with gr.Row():
                sphere_export_dd = gr.Dropdown(
                     label="Сфера",
                     choices=[],
                     value=None,
                     scale=3
                )
                refresh_spheres_export = gr.Button(
                 "Обновить список сфер",
                 variant="secondary",
                 scale=1
                )

            def _ui_get_spheres_export():
                return gr.update(choices=get_all_spheres(), value=None)

            refresh_spheres_export.click(
                _ui_get_spheres_export,
                inputs=None,
                outputs=[sphere_export_dd]
            )


            sphere_btn = gr.Button("Сформировать JSON сферы", variant="primary")
            sphere_status = gr.Markdown()
            sphere_dl = gr.DownloadButton(label="Скачать JSON сферы", value=None)

            # CSV export controls
            sphere_csv_btn = gr.Button("Сформировать CSV сферы", variant="primary")
            sphere_csv_status = gr.Markdown()
            sphere_csv_dl = gr.DownloadButton(label="Скачать CSV сферы", value=None)


            def handle_export_sphere(sphere_query: str):
                """
                Экспортирует записи, относящиеся к заданной сфере (подстрока).
                Составляется общая таблица из библиотечного DataFrame (LIB_DF) и персональной базы,
                затем выбираются строки, где поле "sphere" содержит подстроку sphere_query.
                Возвращает статус и путь к JSON-файлу.
                """
                query = (sphere_query or "").strip().lower()
                if not query:
                    return "Введите название или часть названия сферы.", None
                    
                # загружаем библиотеку и персональную
                _ensure_lib_loaded()
                df = LIB_DF.copy() if LIB_DF is not None else pd.DataFrame()
                # добавляем персональную базу
                if os.path.exists(PERSONAL_CSV):
                    try:
                        df_p = pd.read_csv(PERSONAL_CSV, encoding="utf-8")
                        if not df_p.empty:
                            df_p = df_p.rename(columns={"text": "word"})
                            df_p["sphere"] = "прочее"
                            df_p["tone"] = "neutral"
                            df_p["allowed"] = True
                            df_p["notes"] = ""
                            df_p = df_p[["word", "sphere", "tone", "allowed", "notes", "l1", "l2c", "w", "C", "Hm", "Z"]]
                            df = pd.concat([df, df_p], ignore_index=True) if not df.empty else df_p.copy()
                    except Exception:
                        pass
                if df is None or df.empty:
                    return "Библиотека пуста.", None
                # фильтрация
                dff = df[df["sphere"].apply(lambda s: _sphere_exact_match(s, query))]
                if dff.empty:
                    return f"Не найдены записи для сферы '{sphere_query}'.", None
                # формируем JSON
                items = []
                for _, r in dff.iterrows():
                    try:
                        items.append({
                            "word": str(r.get("word", "")).strip(),
                            "sphere": str(r.get("sphere", "")).strip(),
                            "tone": str(r.get("tone", "")).strip(),
                            "allowed": parse_bool(r.get("allowed", True)),
                            "field": str(r.get("field", "")).strip(),
                            "role":  str(r.get("role", "")).strip(),
                            "notes": str(r.get("notes", "")).strip(),
                            "l1": int(float(r.get("l1", 0))) if not pd.isna(r.get("l1", 0)) else 0,
                            "l2c": int(float(r.get("l2c", 0))) if not pd.isna(r.get("l2c", 0)) else 0,
                            "w": float(r.get("w", 0.0)),
                            "C": float(r.get("C", 0.0)),
                            "Hm": float(r.get("Hm", 0.0)),
                            "Z": float(r.get("Z", 0.0))
                        })
                    except Exception:
                        continue
                data = {
                    "meta": {
                        "encoder": "Kryon-33",
                        "version": ENCODER_VERSION,
                        "calc_version": CALC_VERSION,
                        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
                        "lang": "ru",
                        "sphere_query": sphere_query
                    },
                    "library": items
                }
                # создаём временный файл
                path = f"/tmp/export_sphere_{int(time.time())}.json"
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    return f"Ошибка записи файла: {e}", None
                status = f"Экспортировано записей: {len(items)}."
                return status, path

            # подключаем кнопку сферы
            sphere_btn.click(
                handle_export_sphere,
                inputs=[sphere_export_dd],
                outputs=[sphere_status, sphere_dl]
            )

            # --- обработчик экспорта сферы в CSV ---
            def handle_export_sphere_csv(sphere_query: str):
                query = (sphere_query or "").strip().lower()
                if not query:
                    return "Введите название или часть названия сферы.", None
                _ensure_lib_loaded()
                df = LIB_DF.copy() if LIB_DF is not None else pd.DataFrame()
                # добавляем персональную базу
                if os.path.exists(PERSONAL_CSV):
                    try:
                        df_p = pd.read_csv(PERSONAL_CSV, encoding="utf-8")
                        if not df_p.empty:
                            df_p = df_p.rename(columns={"text": "word"})
                            df_p["sphere"] = "прочее"
                            df_p["tone"] = "neutral"
                            df_p["allowed"] = True
                            df_p["notes"] = ""
                            df_p = df_p[["word", "sphere", "tone", "allowed", "notes", "l1", "l2c", "w", "C", "Hm", "Z"]]
                            df = pd.concat([df, df_p], ignore_index=True) if not df.empty else df_p.copy()
                    except Exception:
                        pass
                if df is None or df.empty:
                    return "Библиотека пуста.", None
                dff = df[df["sphere"].apply(lambda s: _sphere_exact_match(s, query))]
                if dff.empty:
                    return f"Не найдены записи для сферы '{sphere_query}'.", None

                for c in ["field", "role"]:
                    if c not in dff.columns:
                        dff[c] = ""

                # сохраняем CSV
                path = f"/tmp/export_sphere_{int(time.time())}.csv"
                try:
                    dff.to_csv(path, index=False, encoding="utf-8")
                except Exception as e:
                    return f"Ошибка записи CSV: {e}", None
                status = f"Экспортировано записей: {len(dff)}."
                return status, path

            sphere_csv_btn.click(
                handle_export_sphere_csv,
                inputs=[sphere_export_dd],
                outputs=[sphere_csv_status, sphere_csv_dl]
            )


            def handle_export_json(source: str, sphere_query: str = ""):
                """
                Формирует JSON-экспорт библиотеки в зависимости от выбранного источника.
                - "Вся библиотека": LIB_DF + personal.csv
                - "Только персональная библиотека": только personal.csv
                - "По сфере": (опционально) если ты решишь использовать этот режим отдельно
                """
                df = None

                if source == "Только персональная библиотека":
                    if not os.path.exists(PERSONAL_CSV):
                        return "Персональная библиотека пуста.", None
                    try:
                        df_pers = pd.read_csv(PERSONAL_CSV, encoding="utf-8")
                    except Exception:
                        return "Не удалось прочитать персональную библиотеку.", None
                    if df_pers.empty:
                        return "Персональная библиотека пуста.", None

                    df_pers = df_pers.rename(columns={"text": "word"})
                    df_pers["sphere"] = "прочее"
                    df_pers["tone"] = "neutral"
                    df_pers["allowed"] = True
                    df_pers["notes"] = ""

                    df = df_pers[["word","sphere","tone","allowed","notes","l1","l2c","w","C","Hm","Z"]].copy()

                    # ✅ СТРАХОВКА КОНТРАКТА (нужна, потому что personal.csv не содержит field/role)
                    for c in ["field", "role"]:
                        if c not in df.columns:
                            df[c] = ""

                    # Формируем список элементов
                    items = []
                    try:
                        for _, r in df.iterrows():
                            items.append({
                                "word": str(r.get("word", "")).upper(),
                                "sphere": str(r.get("sphere", "прочее")),
                                "tone": str(r.get("tone", "neutral")),
                                "allowed": parse_bool(r.get("allowed", True)),
                                "field": str(r.get("field", "")).strip(),
                                "role":  str(r.get("role", "")).strip(),
                                "notes": "" if pd.isna(r.get("notes")) else str(r.get("notes")).strip(),
                                "l1": int(float(r.get("l1", 0))) if not pd.isna(r.get("l1")) else 0,
                                "l2c": int(float(r.get("l2c", 0))) if not pd.isna(r.get("l2c")) else 0,
                                "w": float(r.get("w", 0.0)) if not pd.isna(r.get("w")) else 0.0,
                                "C": float(r.get("C", 0.0)) if not pd.isna(r.get("C")) else 0.0,
                                "Hm": float(r.get("Hm", 0.0)) if not pd.isna(r.get("Hm")) else 0.0,
                                "Z": float(r.get("Z", 0.0)) if not pd.isna(r.get("Z")) else 0.0,
                            })
                    except Exception:
                        return "Ошибка преобразования данных.", None

                    data = {
                        "version": ENCODER_VERSION,
                        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
                        "library": items
                    }

                    path = f"/tmp/export_library_{int(time.time())}.json"
                    try:
                        with open(path, "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        return f"Ошибка записи файла: {e}", None

                    return f"Экспортировано слов: {len(items)}.", path

                # Вся библиотека
                _ensure_lib_loaded()
                df = LIB_DF.copy() if LIB_DF is not None else pd.DataFrame()

                # Добавляем personal.csv
                if os.path.exists(PERSONAL_CSV):
                    try:
                        df_pers = pd.read_csv(PERSONAL_CSV, encoding="utf-8")
                        if not df_pers.empty:
                            df_pers = df_pers.rename(columns={"text": "word"})
                            df_pers["sphere"] = "прочее"
                            df_pers["tone"] = "neutral"
                            df_pers["allowed"] = True
                            df_pers["notes"] = ""
                            df_pers = df_pers[["word","sphere","tone","allowed","notes","l1","l2c","w","C","Hm","Z"]]
                            df = pd.concat([df, df_pers], ignore_index=True)
                    except Exception:
                        pass

                if df is None or df.empty:
                    return "Библиотека пуста.", None

                # ✅ СТРАХОВКА КОНТРАКТА (нужна, потому что personal.csv не содержит field/role)
                for c in ["field", "role"]:
                    if c not in df.columns:
                        df[c] = ""

                # Формируем список элементов
                items = []
                try:
                    for _, r in df.iterrows():
                        items.append({
                            "word": str(r.get("word", "")).upper(),
                            "sphere": str(r.get("sphere", "прочее")),
                            "tone": str(r.get("tone", "neutral")),
                            "allowed": parse_bool(r.get("allowed", True)),
                            "field": str(r.get("field", "")).strip(),
                            "role":  str(r.get("role", "")).strip(),
                            "notes": "" if pd.isna(r.get("notes")) else str(r.get("notes")).strip(),
                            "l1": int(float(r.get("l1", 0))) if not pd.isna(r.get("l1")) else 0,
                            "l2c": int(float(r.get("l2c", 0))) if not pd.isna(r.get("l2c")) else 0,
                            "w": float(r.get("w", 0.0)) if not pd.isna(r.get("w")) else 0.0,
                            "C": float(r.get("C", 0.0)) if not pd.isna(r.get("C")) else 0.0,
                            "Hm": float(r.get("Hm", 0.0)) if not pd.isna(r.get("Hm")) else 0.0,
                            "Z": float(r.get("Z", 0.0)) if not pd.isna(r.get("Z")) else 0.0,
                        })
                except Exception:
                    return "Ошибка преобразования данных.", None

                data = {
                    "version": ENCODER_VERSION,
                    "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
                    "library": items
                }

                path = f"/tmp/export_library_{int(time.time())}.json"
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    return f"Ошибка записи файла: {e}", None

                return f"Экспортировано слов: {len(items)}.", path


            # обработчик кнопки экспорта библиотеки (вся/персональная)
            export_btn.click(handle_export_json, inputs=[export_source], outputs=[export_status, export_dl])
    # --- Инициализация при запуске ---
    def _init_controls():
        _ensure_lib_loaded()
        if LIB_DF is not None and not LIB_DF.empty:
            rebuild_indexes(LIB_DF)
        stats_str = compute_base_indicator()
        spheres_update = gr.update(choices=get_all_spheres(), value=None)
        return (
            spheres_update,   # для sphere_dd
            spheres_update,   # для sphere_export_dd
            stats_str,
            stats_str,
            stats_str
        )

    demo.load(
        _init_controls,
        inputs=None,
        outputs=[sphere_dd, sphere_export_dd, base_indicator, phrase_base_indicator, library_base_indicator]
    )

    demo.queue().launch()