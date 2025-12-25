#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Подсчёт хвостов с датой внутри word в файле qpm_2025-12_clean.json
"""

import json
from collections import Counter
from typing import List, Dict

def count_tails_with_date(input_file: str):
    """
    Поиск элементов с хвостами даты в word (field != "TIME")
    
    Args:
        input_file: Путь к входному JSON-файлу
    """
    print(f"Чтение файла: {input_file}")
    
    # Чтение JSON-файла
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    items = data.get('items', [])
    print(f"Всего записей: {len(items)}")
    
    # Подстрока для поиска
    target_substring = "ДЕКАБРЯДВЕТЫСЯЧИДВАДЦАТЬПЯТОГОГОДА"
    
    # Поиск элементов: field != "TIME" И word содержит подстроку
    found_items = []
    
    for item in items:
        word = item.get('word', '')
        field = item.get('field', '')
        
        if field != "TIME" and target_substring in word:
            found_items.append(item)
    
    total_count = len(found_items)
    
    print(f"\nНайдено элементов с хвостом даты (field != TIME): {total_count}")
    
    if total_count == 0:
        print("Хвосты с датой не найдены.")
        return
    
    # Breakdown по field
    breakdown_by_field = Counter()
    for item in found_items:
        field = item.get('field', 'UNKNOWN')
        breakdown_by_field[field] += 1
    
    # Breakdown по role
    breakdown_by_role = Counter()
    for item in found_items:
        role = item.get('role', 'UNKNOWN')
        breakdown_by_role[role] += 1
    
    # Примеры слов (до 10)
    examples = [item.get('word', '') for item in found_items[:10]]
    
    # Вывод результатов
    print(f"\nОбщее количество: {total_count}")
    
    print(f"\nBreakdown по field:")
    for field, count in sorted(breakdown_by_field.items()):
        print(f"  {field}: {count}")
    
    print(f"\nBreakdown по role:")
    for role, count in sorted(breakdown_by_role.items()):
        print(f"  {role}: {count}")
    
    print(f"\nПримеры слов (до 10):")
    for i, word in enumerate(examples, 1):
        print(f"  {i}. {word}")
    
    return {
        'total_count': total_count,
        'breakdown_by_field': dict(breakdown_by_field),
        'breakdown_by_role': dict(breakdown_by_role),
        'examples': examples
    }

if __name__ == '__main__':
    input_file = '/Users/airos/Desktop/AI ANGEL/Cursor Quantum/QuantumEncoder/qpm_2025-12_clean.json'
    
    result = count_tails_with_date(input_file)
    
    if result:
        print(f"\nНайдено {result['total_count']} хвостов с датой внутри word (без TIME).")



