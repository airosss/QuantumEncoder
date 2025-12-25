#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Нормализация token_kind для файла qpm_2025-12_extracted.json
"""

import json
from collections import defaultdict, Counter
from typing import Dict, Any, List

def normalize_token_kind(input_file: str, output_file: str):
    """
    Нормализация token_kind по каноническим правилам
    
    Args:
        input_file: Путь к входному JSON-файлу
        output_file: Путь к выходному JSON-файлу
    """
    print(f"Чтение файла: {input_file}")
    
    # Чтение исходного JSON-файла
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    items = data.get('items', [])
    meta = data.get('meta', {}).copy()
    
    print(f"Всего записей: {len(items)}")
    
    # Поля для EVENT при role == "signal"
    EVENT_FIELDS = {"TIME", "LUNAR", "SOLAR", "MAGNETIC", "METEOR", "SCHUMANN"}
    
    # Списки для ошибок
    missing_fields = []
    missing_roles = []
    
    # Счётчики для статистики
    count_EVENT = 0
    count_LEXEME = 0
    breakdown_token_kind_by_field = defaultdict(lambda: {'EVENT': 0, 'LEXEME': 0})
    breakdown_token_kind_by_role = defaultdict(lambda: {'EVENT': 0, 'LEXEME': 0})
    
    # Обработка каждого элемента
    for item in items:
        word = item.get('word', '')
        role = item.get('role')
        field = item.get('field')
        
        # Определение token_kind по правилам
        token_kind = None
        
        # Проверка наличия role и field
        if role is None or role == '':
            missing_roles.append(word)
            token_kind = "LEXEME"
        elif field is None or field == '':
            missing_fields.append(word)
            token_kind = "LEXEME"
        else:
            # Правило A: role == "temporal" → EVENT
            if role == "temporal":
                token_kind = "EVENT"
            # Правило B: role == "marker" → EVENT
            elif role == "marker":
                token_kind = "EVENT"
            # Правило C: role == "state" → LEXEME
            elif role == "state":
                token_kind = "LEXEME"
            # Правило D: role == "signal"
            elif role == "signal":
                # D1: field в EVENT_FIELDS → EVENT
                if field in EVENT_FIELDS:
                    token_kind = "EVENT"
                # D2: иначе → LEXEME
                else:
                    token_kind = "LEXEME"
            else:
                # Неизвестный role → LEXEME
                token_kind = "LEXEME"
        
        # Установка token_kind (добавить или перезаписать)
        item['token_kind'] = token_kind
        
        # Подсчёт статистики
        if token_kind == "EVENT":
            count_EVENT += 1
        else:
            count_LEXEME += 1
        
        # Статистика по field
        if field:
            breakdown_token_kind_by_field[field][token_kind] += 1
        
        # Статистика по role
        if role:
            breakdown_token_kind_by_role[role][token_kind] += 1
    
    # Обновление meta
    meta['count_EVENT'] = count_EVENT
    meta['count_LEXEME'] = count_LEXEME
    
    # Преобразование breakdown_token_kind_by_field в обычный dict
    meta['breakdown_token_kind_by_field'] = {
        field: dict(counts) 
        for field, counts in breakdown_token_kind_by_field.items()
    }
    
    # Преобразование breakdown_token_kind_by_role в обычный dict
    meta['breakdown_token_kind_by_role'] = {
        role: dict(counts) 
        for role, counts in breakdown_token_kind_by_role.items()
    }
    
    # Добавление списков ошибок, если они есть
    if missing_fields:
        meta['missing_fields'] = missing_fields
    
    if missing_roles:
        meta['missing_roles'] = missing_roles
    
    # Формирование результирующего объекта
    result = {
        'meta': meta,
        'items': items
    }
    
    # Сохранение в файл
    print(f"Сохранение в файл: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Готово! Обработано {len(items)} записей")
    print(f"  - EVENT: {count_EVENT}")
    print(f"  - LEXEME: {count_LEXEME}")
    if missing_fields:
        print(f"  - Отсутствует field: {len(missing_fields)} записей")
    if missing_roles:
        print(f"  - Отсутствует role: {len(missing_roles)} записей")
    
    return result

if __name__ == '__main__':
    input_file = '/Users/airos/Desktop/AI ANGEL/Cursor Quantum/QuantumEncoder/qpm_2025-12_extracted.json'
    output_file = '/Users/airos/Desktop/AI ANGEL/Cursor Quantum/QuantumEncoder/qpm_2025-12_tokenkind.json'
    
    normalize_token_kind(input_file, output_file)



