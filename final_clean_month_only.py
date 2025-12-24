#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Финальная очистка month-only: удаление обрубков дней и остатков других месяцев
"""

import json
import re
from collections import Counter
from typing import Dict, Any, List, Tuple

def clean_word_final(word: str, field: str) -> Tuple[str, bool]:
    """
    Финальная очистка word: удаление обрубков дней и остатков других месяцев
    
    Args:
        word: Исходное слово
        field: Поле field элемента
    
    Returns:
        (очищенное слово, изменилось ли слово)
    """
    original_word = word
    
    # Для field == "TIME" не трогаем
    if field == "TIME":
        return word, False
    
    cleaned = word
    
    # 1) Удалить хвосты "ДВАДЦАТЬ" или "ТРИДЦАТЬ" в конце
    cleaned = re.sub(r"(ДВАДЦАТЬ|ТРИДЦАТЬ)$", "", cleaned)
    
    # 2) Удалить хвосты других месяцев в конце
    cleaned = re.sub(r"(ЯНВАРЬ|ФЕВРАЛЬ|МАРТ|АПРЕЛЬ|МАЙ|ИЮНЬ|ИЮЛЬ|АВГУСТ|СЕНТЯБРЬ|ОКТЯБРЬ|НОЯБРЬ)$", "", cleaned)
    
    # Нормализация
    cleaned = cleaned.upper()
    cleaned = cleaned.replace(" ", "")
    
    # Проверка на пустоту или слишком короткое слово
    if not cleaned or len(cleaned) < 3:
        return "", True
    
    changed = (cleaned != original_word)
    return cleaned, changed

def final_clean_month_only(input_file: str, output_file: str):
    """
    Финальная очистка month-only
    
    Args:
        input_file: Путь к входному JSON-файлу
        output_file: Путь к выходному JSON-файлу
    """
    print(f"Чтение файла: {input_file}")
    
    # Чтение исходного JSON-файла
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    items = data.get('items', [])
    input_total = len(items)
    print(f"Исходное количество записей: {input_total}")
    
    # Статистика
    changed_word_count = 0
    removed_empty_word = []
    examples_changed = []
    seen_keys = {}  # Для дедупа: (word, tone, field, role) -> индекс первого вхождения
    removed_dedup = []
    
    # Обработка элементов
    processed_items = []
    
    for idx, item in enumerate(items):
        word = item.get('word', '')
        field = item.get('field', '')
        tone = item.get('tone', '')
        role = item.get('role', '')
        
        # Очистка word
        cleaned_word, word_changed = clean_word_final(word, field)
        
        if not cleaned_word or len(cleaned_word) < 3:
            removed_empty_word.append(word)
            continue
        
        if word_changed:
            changed_word_count += 1
            if len(examples_changed) < 10:
                examples_changed.append({
                    'was': word,
                    'became': cleaned_word
                })
        
        # Дедуп по ключу (word, tone, field, role)
        dedup_key = (cleaned_word, tone, field, role)
        
        if dedup_key in seen_keys:
            # Дубликат - удаляем
            removed_dedup.append({
                'word': cleaned_word,
                'tone': tone,
                'field': field,
                'role': role
            })
            continue
        
        # Первое вхождение - сохраняем
        seen_keys[dedup_key] = idx
        
        # Создаём новый элемент с очищенным word
        new_item = item.copy()
        new_item['word'] = cleaned_word
        
        processed_items.append(new_item)
    
    output_total = len(processed_items)
    
    print(f"Обработано записей: {output_total}")
    print(f"Изменено слов: {changed_word_count}")
    print(f"Удалено пустых слов: {len(removed_empty_word)}")
    print(f"Удалено дублей: {len(removed_dedup)}")
    
    # Sanity checks
    count_non_time_words_ending_DVADCAT_TRIDCAT = 0
    count_non_time_words_ending_other_month = 0
    
    other_months = ["ЯНВАРЬ", "ФЕВРАЛЬ", "МАРТ", "АПРЕЛЬ", "МАЙ", "ИЮНЬ", "ИЮЛЬ", "АВГУСТ", "СЕНТЯБРЬ", "ОКТЯБРЬ", "НОЯБРЬ"]
    
    for item in processed_items:
        word = item.get('word', '')
        field = item.get('field', '')
        
        if field != "TIME":
            # Проверка на "ДВАДЦАТЬ" или "ТРИДЦАТЬ" в конце
            if word.endswith("ДВАДЦАТЬ") or word.endswith("ТРИДЦАТЬ"):
                count_non_time_words_ending_DVADCAT_TRIDCAT += 1
            
            # Проверка на другие месяцы в конце
            for month in other_months:
                if word.endswith(month):
                    count_non_time_words_ending_other_month += 1
                    break
    
    # Формирование meta
    meta = {
        'input_total': input_total,
        'output_total': output_total,
        'changed_word_count': changed_word_count,
        'removed_empty_word_count': len(removed_empty_word),
        'removed_dedup_count': len(removed_dedup),
        'sanity': {
            'count_non_time_words_ending_DVADCAT_TRIDCAT': count_non_time_words_ending_DVADCAT_TRIDCAT,
            'count_non_time_words_ending_other_month': count_non_time_words_ending_other_month
        }
    }
    
    # Примеры изменений
    if examples_changed:
        meta['examples_changed'] = [
            f"{ex['was']} -> {ex['became']}"
            for ex in examples_changed
        ]
    
    # Формирование результирующего объекта
    result = {
        'meta': meta,
        'items': processed_items
    }
    
    # Сохранение в файл
    print(f"\nСохранение в файл: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print("Готово!")
    
    # Вывод sanity checks
    print("\nSanity checks:")
    print(f"  Слова, оканчивающиеся на ДВАДЦАТЬ/ТРИДЦАТЬ (non-TIME): {count_non_time_words_ending_DVADCAT_TRIDCAT} (должно быть 0)")
    print(f"  Слова, оканчивающиеся на другие месяцы (non-TIME): {count_non_time_words_ending_other_month} (должно быть 0)")
    
    return result

if __name__ == '__main__':
    input_file = '/Users/airos/Desktop/AI ANGEL/Cursor Quantum/QuantumEncoder/qpm_2025-12_v2_ready.json'
    output_file = '/Users/airos/Desktop/AI ANGEL/Cursor Quantum/QuantumEncoder/qpm_2025-12_v2_ready2.json'
    
    result = final_clean_month_only(input_file, output_file)
    
    # Итоговая сводка
    meta = result['meta']
    print(f"\nИтоговая сводка:")
    print(f"  Было: {meta['input_total']}")
    print(f"  Стало: {meta['output_total']}")
    print(f"  Изменено слов: {meta['changed_word_count']}")
    print(f"  Удалено пустых: {meta['removed_empty_word_count']}")
    print(f"  Удалено дублей: {meta['removed_dedup_count']}")


