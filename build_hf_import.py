#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сборка финального файла для импорта в HuggingFace для QPM
"""

import json
import re
from collections import Counter
from typing import Dict, Any, List, Tuple

def generate_time_tokens():
    """
    Генерация списка из 32 TIME-токенов для декабря
    """
    tokens = ["ДЕКАБРЬ"]
    
    # Дни декабря: ПЕРВОЕДЕКАБРЯ, ВТОРОЕДЕКАБРЯ, ..., ТРИДЦАТЬПЕРВОЕДЕКАБРЯ
    day_names = [
        "ПЕРВОЕ", "ВТОРОЕ", "ТРЕТЬЕ", "ЧЕТВЕРТОЕ", "ПЯТОЕ", "ШЕСТОЕ",
        "СЕДЬМОЕ", "ВОСЬМОЕ", "ДЕВЯТОЕ", "ДЕСЯТОЕ", "ОДИННАДЦАТОЕ",
        "ДВЕНАДЦАТОЕ", "ТРИНАДЦАТОЕ", "ЧЕТЫРНАДЦАТОЕ", "ПЯТНАДЦАТОЕ",
        "ШЕСТНАДЦАТОЕ", "СЕМНАДЦАТОЕ", "ВОСЕМНАДЦАТОЕ", "ДЕВЯТНАДЦАТОЕ",
        "ДВАДЦАТОЕ", "ДВАДЦАТЬПЕРВОЕ", "ДВАДЦАТЬВТОРОЕ", "ДВАДЦАТЬТРЕТЬЕ",
        "ДВАДЦАТЬЧЕТВЕРТОЕ", "ДВАДЦАТЬПЯТОЕ", "ДВАДЦАТЬШЕСТОЕ",
        "ДВАДЦАТЬСЕДЬМОЕ", "ДВАДЦАТЬВОСЬМОЕ", "ДВАДЦАТЬДЕВЯТОЕ",
        "ТРИДЦАТОЕ", "ТРИДЦАТЬПЕРВОЕ"
    ]
    
    for day in day_names:
        tokens.append(f"{day}ДЕКАБРЯ")
    
    return tokens

def normalize_time_word(word: str) -> str:
    """
    Нормализация TIME-слова: удаление хвоста года, приведение к month-only формату
    """
    cleaned = word
    
    # Удалить хвост "ДВЕТЫСЯЧИ..." если есть
    cleaned = re.sub(r"ДВЕТЫСЯЧИ[А-ЯЁ0-9]+", "", cleaned)
    
    # Удалить "ГОДА" или "ГОД" в конце
    cleaned = re.sub(r"ГОДА?$", "", cleaned)
    
    # Удалить пробелы и привести к верхнему регистру
    cleaned = cleaned.upper().replace(" ", "")
    
    # Если слово начинается с дня и заканчивается на "ДЕКАБРЯ" - оставить как есть
    # Если это просто "ДЕКАБРЬ" - оставить
    # Если есть "ДЕКАБРЯ" в конце - оставить
    # Иначе попробовать найти паттерн дня+ДЕКАБРЯ
    
    return cleaned

def clean_non_time_word(word: str) -> Tuple[str, bool]:
    """
    Очистка non-TIME слова от хвостов и обрубков
    
    Returns:
        (очищенное слово, изменилось ли)
    """
    original = word
    cleaned = word
    
    # Удалить хвосты "ДЕКАБРЯДВЕТЫСЯЧ", "ДВЕТЫСЯЧ"
    cleaned = re.sub(r"ДЕКАБРЯДВЕТЫСЯЧИ[А-ЯЁ0-9]+", "", cleaned)
    cleaned = re.sub(r"ДВЕТЫСЯЧИ[А-ЯЁ0-9]+", "", cleaned)
    
    # Удалить другие месяцы в конце
    other_months = ["ЯНВАРЯ", "ФЕВРАЛЯ", "МАРТА", "АПРЕЛЯ", "МАЯ", "ИЮНЯ", 
                    "ИЮЛЯ", "АВГУСТА", "СЕНТЯБРЯ", "ОКТЯБРЯ", "НОЯБРЯ",
                    "ЯНВАРЬ", "ФЕВРАЛЬ", "МАРТ", "АПРЕЛЬ", "МАЙ", "ИЮНЬ",
                    "ИЮЛЬ", "АВГУСТ", "СЕНТЯБРЬ", "ОКТЯБРЬ", "НОЯБРЬ"]
    
    for month in other_months:
        if cleaned.endswith(month):
            cleaned = cleaned[:-len(month)]
            break
    
    # Удалить обрубки "ДВАДЦАТЬ", "ТРИДЦАТЬ" в конце
    cleaned = re.sub(r"(ДВАДЦАТЬ|ТРИДЦАТЬ)$", "", cleaned)
    
    # Нормализация
    cleaned = cleaned.upper().replace(" ", "")
    
    changed = (cleaned != original)
    return cleaned, changed

def build_hf_import(input_file: str, output_file: str):
    """
    Сборка финального файла для импорта в HuggingFace
    
    Args:
        input_file: Путь к входному JSON-файлу
        output_file: Путь к выходному JSON-файлу
    """
    print(f"Чтение файла: {input_file}")
    
    # Чтение исходного JSON-файла
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    items = data.get('items', [])
    print(f"Исходное количество записей: {len(items)}")
    
    # Генерация стандартных TIME-токенов
    standard_time_tokens = generate_time_tokens()
    print(f"Стандартных TIME-токенов: {len(standard_time_tokens)}")
    
    # Разделение на TIME и non-TIME
    time_items = []
    non_time_items = []
    
    for item in items:
        if item.get('field') == 'TIME':
            time_items.append(item)
        else:
            non_time_items.append(item)
    
    print(f"TIME элементов в исходном файле: {len(time_items)}")
    print(f"Non-TIME элементов: {len(non_time_items)}")
    
    # 1) Нормализация TIME-токенов
    # Просто создаём стандартный набор из 32 токенов
    final_time_items = []
    time_missing_days = []
    
    for token in standard_time_tokens:
        final_time_items.append({
            'word': token,
            'sphere': 'Quantum Prognosis Monthly',
            'tone': '2025-12',
            'field': 'TIME',
            'role': 'temporal',
            'allowed': True,
            'notes': '2025-12 | TIME',
            'token_kind': 'EVENT'
        })
    
    # Проверяем, какие токены были в исходном файле (для отчёта)
    existing_time_words = set()
    for item in time_items:
        word = item.get('word', '')
        normalized = normalize_time_word(word)
        existing_time_words.add(normalized)
    
    # Определяем недостающие дни
    for token in standard_time_tokens:
        found = False
        for existing in existing_time_words:
            if token == existing or existing.endswith(token) or token in existing:
                found = True
                break
        if not found:
            time_missing_days.append(token)
    
    print(f"Финальных TIME-токенов: {len(final_time_items)}")
    
    # 2) Очистка non-TIME слов
    cleaned_non_time_items = []
    changed_count = 0
    
    for item in non_time_items:
        word = item.get('word', '')
        cleaned_word, changed = clean_non_time_word(word)
        
        if not cleaned_word or len(cleaned_word) < 3:
            continue
        
        if changed:
            changed_count += 1
        
        new_item = item.copy()
        new_item['word'] = cleaned_word
        cleaned_non_time_items.append(new_item)
    
    print(f"Изменено non-TIME слов: {changed_count}")
    
    # Дедуп non-TIME по ключу (word, tone, field, role)
    seen_keys = {}
    deduped_non_time = []
    
    for item in cleaned_non_time_items:
        key = (
            item.get('word', ''),
            item.get('tone', ''),
            item.get('field', ''),
            item.get('role', '')
        )
        
        if key not in seen_keys:
            seen_keys[key] = True
            deduped_non_time.append(item)
    
    print(f"Non-TIME после дедупа: {len(deduped_non_time)}")
    
    # Объединение TIME и non-TIME
    final_items = final_time_items + deduped_non_time
    
    # Sanity checks
    non_time_with_DVETYSYACHI = 0
    non_time_with_other_month = 0
    non_time_bad_tail_DVADCAT_TRIDCAT = 0
    
    other_months = ["ЯНВАРЯ", "ФЕВРАЛЯ", "МАРТА", "АПРЕЛЯ", "МАЯ", "ИЮНЯ",
                    "ИЮЛЯ", "АВГУСТА", "СЕНТЯБРЯ", "ОКТЯБРЯ", "НОЯБРЯ",
                    "ЯНВАРЬ", "ФЕВРАЛЬ", "МАРТ", "АПРЕЛЬ", "МАЙ", "ИЮНЬ",
                    "ИЮЛЬ", "АВГУСТ", "СЕНТЯБРЬ", "ОКТЯБРЬ", "НОЯБРЬ"]
    
    for item in final_items:
        if item.get('field') != 'TIME':
            word = item.get('word', '')
            
            if re.search(r"ДВЕТЫСЯЧИ[А-ЯЁ0-9]+", word):
                non_time_with_DVETYSYACHI += 1
            
            for month in other_months:
                if month in word:
                    non_time_with_other_month += 1
                    break
            
            if word.endswith("ДВАДЦАТЬ") or word.endswith("ТРИДЦАТЬ"):
                non_time_bad_tail_DVADCAT_TRIDCAT += 1
    
    # Статистика
    breakdown_by_field = Counter()
    breakdown_by_role = Counter()
    count_EVENT = 0
    count_LEXEME = 0
    
    for item in final_items:
        field = item.get('field', 'UNKNOWN')
        role = item.get('role', 'UNKNOWN')
        token_kind = item.get('token_kind', 'LEXEME')
        
        breakdown_by_field[field] += 1
        breakdown_by_role[role] += 1
        
        if token_kind == "EVENT":
            count_EVENT += 1
        else:
            count_LEXEME += 1
    
    # Формирование выходного файла (массив объектов без meta)
    output_data = final_items
    
    # Сохранение в файл
    print(f"\nСохранение в файл: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("Готово!")
    
    # Вывод отчёта
    print("\n" + "="*60)
    print("ОТЧЁТ:")
    print("="*60)
    print(f"\nОбщая статистика:")
    print(f"  Всего элементов: {len(final_items)}")
    print(f"  TIME элементов: {len(final_time_items)}")
    print(f"  Non-TIME элементов: {len(deduped_non_time)}")
    
    print(f"\nBreakdown по field:")
    for field, count in sorted(breakdown_by_field.items()):
        print(f"  {field}: {count}")
    
    print(f"\nBreakdown по role:")
    for role, count in sorted(breakdown_by_role.items()):
        print(f"  {role}: {count}")
    
    print(f"\nToken kind:")
    print(f"  EVENT: {count_EVENT}")
    print(f"  LEXEME: {count_LEXEME}")
    
    print(f"\nSanity checks:")
    print(f"  Non-TIME с хвостом ДВЕТЫСЯЧИ...: {non_time_with_DVETYSYACHI}")
    print(f"  Non-TIME с другими месяцами: {non_time_with_other_month}")
    print(f"  Non-TIME с обрубками ДВАДЦАТЬ/ТРИДЦАТЬ: {non_time_bad_tail_DVADCAT_TRIDCAT}")
    print(f"  TIME элементов всего: {len(final_time_items)}")
    if time_missing_days:
        print(f"  TIME недостающих дней: {len(time_missing_days)}")
        print(f"    {time_missing_days[:5]}...")
    else:
        print(f"  TIME недостающих дней: 0")
    
    print(f"\nСписок TIME-токенов ({len(final_time_items)}):")
    for i, item in enumerate(final_time_items, 1):
        print(f"  {i:2d}. {item['word']}")
    
    return output_data

if __name__ == '__main__':
    input_file = '/Users/airos/Desktop/AI ANGEL/Cursor Quantum/QuantumEncoder/qpm_2025-12_v2_ready2.json'
    output_file = '/Users/airos/Desktop/AI ANGEL/Cursor Quantum/QuantumEncoder/qpm_2025-12_HF_IMPORT.json'
    
    build_hf_import(input_file, output_file)

