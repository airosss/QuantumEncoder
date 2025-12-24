#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Извлечение данных по указанному месяцу из JSON-файла библиотеки Quantum Prognosis Monthly
"""

import json
from collections import defaultdict, Counter
from typing import List, Dict, Any, Set, Tuple

def extract_month_data(input_file: str, output_file: str, target_tone: str = "2025-12"):
    """
    Извлечение данных по указанному месяцу
    
    Args:
        input_file: Путь к входному JSON-файлу
        output_file: Путь к выходному JSON-файлу
        target_tone: Целевой месяц (формат: "2025-12")
    """
    print(f"Чтение файла: {input_file}")
    
    # Чтение исходного JSON-файла
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Получение массива library
    library = data.get('library', [])
    print(f"Всего записей: {len(library)}")
    
    # Фильтрация объектов по условию
    filtered_items = []
    for item in library:
        if (item.get('sphere') == 'Quantum Prognosis Monthly' and 
            item.get('tone') == target_tone):
            filtered_items.append(item)
    
    print(f"Записей, соответствующих условию: {len(filtered_items)}")
    
    # Статистика
    total_words = len(filtered_items)
    
    # breakdown_by_field
    breakdown_by_field = Counter()
    for item in filtered_items:
        field = item.get('field', 'UNKNOWN')
        breakdown_by_field[field] += 1
    
    # breakdown_by_role
    breakdown_by_role = Counter()
    for item in filtered_items:
        role = item.get('role', 'UNKNOWN')
        breakdown_by_role[role] += 1
    
    # count_EVENT и count_LEXEME
    count_EVENT = 0
    count_LEXEME = 0
    for item in filtered_items:
        token_kind = item.get('token_kind')
        if token_kind == 'EVENT':
            count_EVENT += 1
        elif token_kind == 'LEXEME':
            count_LEXEME += 1
    
    # Поиск дубликатов (word, tone, field, role)
    duplicates = []
    seen_keys: Dict[Tuple[str, str, str, str], List[Dict]] = defaultdict(list)
    
    for idx, item in enumerate(filtered_items):
        key = (
            item.get('word', ''),
            item.get('tone', ''),
            item.get('field', ''),
            item.get('role', '')
        )
        seen_keys[key].append((idx, item))
    
    for key, occurrences in seen_keys.items():
        if len(occurrences) > 1:
            duplicates.append({
                'key': {
                    'word': key[0],
                    'tone': key[1],
                    'field': key[2],
                    'role': key[3]
                },
                'count': len(occurrences),
                'indices': [idx for idx, _ in occurrences]
            })
    
    # Поиск конфликтов (одинаковые word+tone, но разные field)
    conflicts = []
    word_tone_map: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    
    for idx, item in enumerate(filtered_items):
        key = (item.get('word', ''), item.get('tone', ''))
        word_tone_map[key].append((idx, item))
    
    for key, occurrences in word_tone_map.items():
        fields = set(item.get('field', '') for _, item in occurrences)
        if len(fields) > 1:
            conflicts.append({
                'word': key[0],
                'tone': key[1],
                'fields': sorted(list(fields)),
                'count': len(occurrences),
                'indices': [idx for idx, _ in occurrences]
            })
    
    # Формирование объекта meta
    meta = {
        'total_words': total_words,
        'breakdown_by_field': dict(breakdown_by_field),
        'breakdown_by_role': dict(breakdown_by_role),
    }
    
    if count_EVENT > 0 or count_LEXEME > 0:
        meta['count_EVENT'] = count_EVENT
        meta['count_LEXEME'] = count_LEXEME
    
    if duplicates:
        meta['duplicates'] = duplicates
    
    if conflicts:
        meta['conflicts'] = conflicts
    
    # Формирование результирующего объекта
    result = {
        'meta': meta,
        'items': filtered_items
    }
    
    # Сохранение в файл
    print(f"Сохранение в файл: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Готово! Извлечено {total_words} записей")
    print(f"\nСтатистика:")
    print(f"  - Всего слов: {total_words}")
    print(f"  - По полю field: {dict(breakdown_by_field)}")
    print(f"  - По полю role: {dict(breakdown_by_role)}")
    if count_EVENT > 0 or count_LEXEME > 0:
        print(f"  - EVENT: {count_EVENT}, LEXEME: {count_LEXEME}")
    if duplicates:
        print(f"  - Дубликаты: {len(duplicates)} групп")
    if conflicts:
        print(f"  - Конфликты: {len(conflicts)} групп")
    
    return result

if __name__ == '__main__':
    input_file = '/Users/airos/Downloads/QPM старый.json'
    output_file = '/Users/airos/Desktop/AI ANGEL/Cursor Quantum/QuantumEncoder/qpm_2025-12_extracted.json'
    
    extract_month_data(input_file, output_file, target_tone='2025-12')

