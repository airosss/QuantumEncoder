#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пересборка месяца 2025-12 по канону QPM v2
"""

import json
import re
from collections import Counter
from typing import Dict, Any, List, Tuple

def normalize_word(word: str, field: str) -> Tuple[str, bool]:
    """
    Нормализация word по правилам QPM v2
    
    Args:
        word: Исходное слово
        field: Поле field элемента
    
    Returns:
        (нормализованное слово, изменилось ли слово)
    """
    original_word = word
    
    # Правило B1: Для field == "TIME" - не трогать
    if field == "TIME":
        return word, False
    
    # Правило B2: Для field != "TIME" - удалить хвосты даты
    cleaned = word
    
    # Удаляем подстроки (в порядке от длинных к коротким)
    substrings_to_remove = [
        "ДЕКАБРЯДВЕТЫСЯЧИДВАДЦАТЬПЯТОГОГОДА",
        "ДЕКАБРЯ",
        "ДВЕТЫСЯЧИДВАДЦАТЬПЯТОГОГОДА",
        "ГОДА"
    ]
    
    for substr in substrings_to_remove:
        cleaned = cleaned.replace(substr, "")
    
    # Удаление двойных повторяющихся частей (упрощённо: удаляем повторяющиеся символы подряд)
    # Но это может быть сложно, поэтому просто убираем лишние пробелы если есть
    cleaned = cleaned.replace(" ", "")
    
    # Верхний регистр
    cleaned = cleaned.upper()
    
    # Проверка на пустоту или слишком короткое слово
    if not cleaned or len(cleaned) < 3:
        return "", True  # Пустое слово, нужно удалить
    
    changed = (cleaned != original_word)
    return cleaned, changed

def normalize_token_kind(role: str, field: str) -> str:
    """
    Нормализация token_kind по правилам QPM v2
    
    Args:
        role: Роль элемента
        field: Поле field элемента
    
    Returns:
        token_kind: "EVENT" или "LEXEME"
    """
    # Правило E
    if role == "temporal":
        return "EVENT"
    elif role == "marker":
        return "EVENT"
    elif role == "state":
        return "LEXEME"
    elif role == "signal":
        event_fields = {"SOLAR", "LUNAR", "MAGNETIC", "METEOR", "SCHUMANN", "TIME"}
        if field in event_fields:
            return "EVENT"
        else:
            return "LEXEME"
    else:
        # По умолчанию LEXEME
        return "LEXEME"

def rebuild_qpm_v2(input_file: str, output_file: str):
    """
    Пересборка месяца 2025-12 по канону QPM v2
    
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
    removed_time_in_word = []
    removed_forbidden_psy = []
    
    # Запрещённые слова для PSY (правило C)
    forbidden_psy_words = {"ДЕКАБРЬ", "ДВЕТЫСЯЧИДВАДЦАТЬПЯТЫЙГОД", "ДВЕТЫСЯЧИДВАДЦАТЬПЯТОГОГОДА"}
    
    # Обработка элементов
    output_items = []
    
    for item in items:
        # Правило A: Проверка sphere и tone
        if (item.get('sphere') != 'Quantum Prognosis Monthly' or 
            item.get('tone') != '2025-12'):
            continue
        
        word = item.get('word', '')
        field = item.get('field', '')
        role = item.get('role', '')
        
        # Правило C: Удаление запрещённых PSY-слов
        if field == "PSY" and word in forbidden_psy_words:
            removed_forbidden_psy.append(word)
            continue
        
        # Правило B: Нормализация word
        normalized_word, word_changed = normalize_word(word, field)
        
        if not normalized_word or len(normalized_word) < 3:
            removed_empty_word.append(word)
            continue
        
        if word_changed:
            changed_word_count += 1
        
        # Правило C: Проверка на наличие даты в word (для не-TIME)
        if field != "TIME":
            if "ДВАДЦАТЬПЯТОГОГОДА" in normalized_word or "ДВАДЦАТЬГОДА" in normalized_word:
                removed_time_in_word.append(word)
                continue
        
        # Правило D: Нормализация notes
        notes = f"2025-12 | {field}"
        
        # Правило E: Нормализация token_kind
        token_kind = normalize_token_kind(role, field)
        
        # Правило F: Создание выходного объекта только с нужными полями
        output_item = {
            'word': normalized_word,
            'sphere': item.get('sphere'),
            'tone': item.get('tone'),
            'field': field,
            'role': role,
            'allowed': item.get('allowed'),
            'notes': notes,
            'token_kind': token_kind
        }
        
        output_items.append(output_item)
    
    output_total = len(output_items)
    
    print(f"Обработано записей: {output_total}")
    print(f"Изменено слов: {changed_word_count}")
    print(f"Удалено пустых слов: {len(removed_empty_word)}")
    print(f"Удалено с датой в word: {len(removed_time_in_word)}")
    print(f"Удалено запрещённых PSY: {len(removed_forbidden_psy)}")
    
    # Статистика по выходным данным
    breakdown_by_field_output = Counter()
    breakdown_by_role_output = Counter()
    count_EVENT_output = 0
    count_LEXEME_output = 0
    
    for item in output_items:
        field = item.get('field', 'UNKNOWN')
        role = item.get('role', 'UNKNOWN')
        token_kind = item.get('token_kind', 'LEXEME')
        
        breakdown_by_field_output[field] += 1
        breakdown_by_role_output[role] += 1
        
        if token_kind == "EVENT":
            count_EVENT_output += 1
        else:
            count_LEXEME_output += 1
    
    # Формирование meta
    meta = {
        'input_total': input_total,
        'output_total': output_total,
        'changed_word_count': changed_word_count,
        'removed_empty_word_count': len(removed_empty_word),
        'removed_time_in_word_count': len(removed_time_in_word),
        'breakdown_by_field_output': dict(breakdown_by_field_output),
        'breakdown_by_role_output': dict(breakdown_by_role_output),
        'count_EVENT_output': count_EVENT_output,
        'count_LEXEME_output': count_LEXEME_output
    }
    
    # Примеры удалённых элементов
    removed_examples = {}
    if removed_empty_word:
        removed_examples['removed_empty_word'] = removed_empty_word[:10]
    if removed_time_in_word:
        removed_examples['removed_time_in_word'] = removed_time_in_word[:10]
    if removed_forbidden_psy:
        removed_examples['removed_forbidden_psy'] = removed_forbidden_psy[:10]
    
    if removed_examples:
        meta['removed_examples'] = removed_examples
    
    # Формирование результирующего объекта
    result = {
        'meta': meta,
        'items': output_items
    }
    
    # Сохранение в файл
    print(f"\nСохранение в файл: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print("Готово!")
    
    return result

if __name__ == '__main__':
    input_file = '/Users/airos/Desktop/AI ANGEL/Cursor Quantum/QuantumEncoder/qpm_2025-12_clean.json'
    output_file = '/Users/airos/Desktop/AI ANGEL/Cursor Quantum/QuantumEncoder/qpm_2025-12_v2_source.json'
    
    result = rebuild_qpm_v2(input_file, output_file)
    
    # Вывод итоговой сводки meta
    print("\n" + "="*60)
    print("ИТОГОВАЯ СВОДКА META:")
    print("="*60)
    meta = result['meta']
    print(f"Входных записей: {meta['input_total']}")
    print(f"Выходных записей: {meta['output_total']}")
    print(f"Изменено слов: {meta['changed_word_count']}")
    print(f"Удалено пустых слов: {meta['removed_empty_word_count']}")
    print(f"Удалено с датой в word: {meta['removed_time_in_word_count']}")
    print(f"\nBreakdown по field (выход):")
    for field, count in sorted(meta['breakdown_by_field_output'].items()):
        print(f"  {field}: {count}")
    print(f"\nBreakdown по role (выход):")
    for role, count in sorted(meta['breakdown_by_role_output'].items()):
        print(f"  {role}: {count}")
    print(f"\nToken kind (выход):")
    print(f"  EVENT: {meta['count_EVENT_output']}")
    print(f"  LEXEME: {meta['count_LEXEME_output']}")
    if 'removed_examples' in meta:
        print(f"\nПримеры удалённых элементов:")
        for reason, examples in meta['removed_examples'].items():
            print(f"  {reason}: {len(examples)} примеров")
            for word in examples[:3]:
                print(f"    - {word}")



