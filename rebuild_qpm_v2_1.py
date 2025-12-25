#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Доведение qpm_2025-12_v2_source.json до QPM v2.1 (month-only)
"""

import json
import re
from collections import Counter
from typing import Dict, Any, List, Tuple

def clean_word_v2_1(word: str, field: str) -> Tuple[str, bool]:
    """
    Очистка word по правилам QPM v2.1
    
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
    
    # 1.1) Удалить остатки года/времени
    # Regex для удаления хвоста "ДВЕТЫСЯЧИ[А-ЯЁ0-9]+$"
    cleaned = re.sub(r"ДВЕТЫСЯЧИ[А-ЯЁ0-9]+$", "", cleaned)
    
    # Удалить одиночные "ГОД", "ГОДА" в конце
    cleaned = re.sub(r"ГОД$", "", cleaned)
    cleaned = re.sub(r"ГОДА$", "", cleaned)
    
    # 1.2) Удалить суффиксы дневной привязки (только в конце слова)
    day_suffixes = [
        "ПЕРВОГО", "ВТОРОГО", "ТРЕТЬЕГО", "ЧЕТВЕРТОГО", "ПЯТОГО", "ШЕСТОГО",
        "СЕДЬМОГО", "ВОСЬМОГО", "ДЕВЯТОГО", "ДЕСЯТОГО", "ОДИННАДЦАТОГО",
        "ДВЕНАДЦАТОГО", "ТРИНАДЦАТОГО", "ЧЕТЫРНАДЦАТОГО", "ПЯТНАДЦАТОГО",
        "ШЕСТНАДЦАТОГО", "СЕМНАДЦАТОГО", "ВОСЕМНАДЦАТОГО", "ДЕВЯТНАДЦАТОГО",
        "ДВАДЦАТОГО", "ДВАДЦАТЬПЕРВОГО", "ДВАДЦАТЬВТОРОГО", "ДВАДЦАТЬТРЕТЬЕГО",
        "ДВАДЦАТЬЧЕТВЕРТОГО", "ДВАДЦАТЬПЯТОГО", "ДВАДЦАТЬШЕСТОГО",
        "ДВАДЦАТЬСЕДЬМОГО", "ДВАДЦАТЬВОСЬМОГО", "ДВАДЦАТЬДЕВЯТОГО",
        "ТРИДЦАТОГО", "ТРИДЦАТЬПЕРВОГО"
    ]
    
    for suffix in day_suffixes:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)]
            break
    
    # 1.3) Удалить артефакты "ДЕКАБРЬ" в конце
    if cleaned.endswith("ДЕКАБРЬ"):
        cleaned = cleaned[:-len("ДЕКАБРЬ")]
    
    # 1.4) Нормализация
    cleaned = cleaned.upper()
    cleaned = cleaned.replace(" ", "")
    
    # Проверка на пустоту или слишком короткое слово
    if not cleaned or len(cleaned) < 3:
        return "", True
    
    changed = (cleaned != original_word)
    return cleaned, changed

def rebuild_qpm_v2_1(input_file: str, output_file: str):
    """
    Доведение файла до QPM v2.1 (month-only)
    
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
    seen_keys = {}  # Для дедупа: (word, tone, field, role) -> индекс первого вхождения
    removed_dedup = []
    
    # Обработка элементов
    processed_items = []
    
    for idx, item in enumerate(items):
        word = item.get('word', '')
        field = item.get('field', '')
        tone = item.get('tone', '')
        role = item.get('role', '')
        
        # 1) Исправление word
        cleaned_word, word_changed = clean_word_v2_1(word, field)
        
        if not cleaned_word or len(cleaned_word) < 3:
            removed_empty_word.append(word)
            continue
        
        if word_changed:
            changed_word_count += 1
        
        # 2) Дедуп по ключу (word, tone, field, role)
        dedup_key = (cleaned_word, tone, field, role)
        
        if dedup_key in seen_keys:
            # Дубликат - удаляем
            removed_dedup.append({
                'word': cleaned_word,
                'tone': tone,
                'field': field,
                'role': role,
                'original_index': idx
            })
            continue
        
        # Первое вхождение - сохраняем
        seen_keys[dedup_key] = idx
        
        # Создаём новый элемент с очищенным word
        new_item = item.copy()
        new_item['word'] = cleaned_word
        
        # 3) Notes оставляем как есть
        # 4) token_kind оставляем как есть
        
        processed_items.append(new_item)
    
    output_total = len(processed_items)
    
    print(f"Обработано записей: {output_total}")
    print(f"Изменено слов: {changed_word_count}")
    print(f"Удалено пустых слов: {len(removed_empty_word)}")
    print(f"Удалено дублей: {len(removed_dedup)}")
    
    # Статистика по выходным данным
    breakdown_by_field_output = Counter()
    breakdown_by_role_output = Counter()
    count_EVENT_output = 0
    count_LEXEME_output = 0
    
    for item in processed_items:
        field = item.get('field', 'UNKNOWN')
        role = item.get('role', 'UNKNOWN')
        token_kind = item.get('token_kind', 'LEXEME')
        
        breakdown_by_field_output[field] += 1
        breakdown_by_role_output[role] += 1
        
        if token_kind == "EVENT":
            count_EVENT_output += 1
        else:
            count_LEXEME_output += 1
    
    # Sanity checks
    count_non_time_words_with_DECEMBER_suffix = 0
    count_non_time_words_with_DVETYSYACHI_tail = 0
    count_non_time_words_with_day_suffix = 0
    
    day_suffixes = [
        "ПЕРВОГО", "ВТОРОГО", "ТРЕТЬЕГО", "ЧЕТВЕРТОГО", "ПЯТОГО", "ШЕСТОГО",
        "СЕДЬМОГО", "ВОСЬМОГО", "ДЕВЯТОГО", "ДЕСЯТОГО", "ОДИННАДЦАТОГО",
        "ДВЕНАДЦАТОГО", "ТРИНАДЦАТОГО", "ЧЕТЫРНАДЦАТОГО", "ПЯТНАДЦАТОГО",
        "ШЕСТНАДЦАТОГО", "СЕМНАДЦАТОГО", "ВОСЕМНАДЦАТОГО", "ДЕВЯТНАДЦАТОГО",
        "ДВАДЦАТОГО", "ДВАДЦАТЬПЕРВОГО", "ДВАДЦАТЬВТОРОГО", "ДВАДЦАТЬТРЕТЬЕГО",
        "ДВАДЦАТЬЧЕТВЕРТОГО", "ДВАДЦАТЬПЯТОГО", "ДВАДЦАТЬШЕСТОГО",
        "ДВАДЦАТЬСЕДЬМОГО", "ДВАДЦАТЬВОСЬМОГО", "ДВАДЦАТЬДЕВЯТОГО",
        "ТРИДЦАТОГО", "ТРИДЦАТЬПЕРВОГО"
    ]
    
    for item in processed_items:
        word = item.get('word', '')
        field = item.get('field', '')
        
        if field != "TIME":
            # Проверка на "ДЕКАБРЬ" в конце
            if word.endswith("ДЕКАБРЬ"):
                count_non_time_words_with_DECEMBER_suffix += 1
            
            # Проверка на хвост "ДВЕТЫСЯЧИ..."
            if re.search(r"ДВЕТЫСЯЧИ[А-ЯЁ0-9]+$", word):
                count_non_time_words_with_DVETYSYACHI_tail += 1
            
            # Проверка на суффиксы дневной привязки
            for suffix in day_suffixes:
                if word.endswith(suffix):
                    count_non_time_words_with_day_suffix += 1
                    break
    
    # Формирование meta
    meta = {
        'input_total': input_total,
        'output_total': output_total,
        'changed_word_count': changed_word_count,
        'removed_empty_word_count': len(removed_empty_word),
        'removed_dedup_count': len(removed_dedup),
        'breakdown_by_field_output': dict(breakdown_by_field_output),
        'breakdown_by_role_output': dict(breakdown_by_role_output),
        'count_EVENT_output': count_EVENT_output,
        'count_LEXEME_output': count_LEXEME_output,
        'sanity_checks': {
            'count_non_time_words_with_DECEMBER_suffix': count_non_time_words_with_DECEMBER_suffix,
            'count_non_time_words_with_DVETYSYACHI_tail': count_non_time_words_with_DVETYSYACHI_tail,
            'count_non_time_words_with_day_suffix': count_non_time_words_with_day_suffix
        }
    }
    
    # Примеры удалённых элементов
    removed_examples = {}
    if removed_empty_word:
        removed_examples['removed_empty_word'] = removed_empty_word[:10]
    if removed_dedup:
        removed_examples['removed_dedup'] = [
            f"{item['word']} ({item['field']}, {item['role']})"
            for item in removed_dedup[:10]
        ]
    
    if removed_examples:
        meta['removed_examples'] = removed_examples
    
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
    
    return result

if __name__ == '__main__':
    input_file = '/Users/airos/Desktop/AI ANGEL/Cursor Quantum/QuantumEncoder/qpm_2025-12_v2_source.json'
    output_file = '/Users/airos/Desktop/AI ANGEL/Cursor Quantum/QuantumEncoder/qpm_2025-12_v2_ready.json'
    
    result = rebuild_qpm_v2_1(input_file, output_file)
    
    # Вывод итоговой сводки
    meta = result['meta']
    print(f"\nГотово: было {meta['input_total']}, стало {meta['output_total']}, "
          f"изменено word {meta['changed_word_count']}, "
          f"удалено пустых {meta['removed_empty_word_count']}, "
          f"удалено дублей {meta['removed_dedup_count']}. "
          f"Файл: qpm_2025-12_v2_ready.json")
    
    # Вывод sanity checks
    print("\nSanity checks:")
    sanity = meta['sanity_checks']
    print(f"  Слова с суффиксом ДЕКАБРЬ (non-TIME): {sanity['count_non_time_words_with_DECEMBER_suffix']} (должно быть 0)")
    print(f"  Слова с хвостом ДВЕТЫСЯЧИ... (non-TIME): {sanity['count_non_time_words_with_DVETYSYACHI_tail']} (должно быть 0)")
    print(f"  Слова с суффиксом дня (non-TIME): {sanity['count_non_time_words_with_day_suffix']} (должно быть 0)")



