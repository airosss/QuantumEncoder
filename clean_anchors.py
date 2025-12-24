#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Очистка файла qpm_2025-12_tokenkind.json от неверных TIME/PSY-якорей
"""

import json
from typing import Dict, Any, List

def clean_anchors(input_file: str, output_file: str):
    """
    Удаление неверных якорей по правилам A и B
    
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
    
    original_count = len(items)
    print(f"Исходное количество записей: {original_count}")
    
    # Списки для удаления
    removed_rule_A = []
    removed_rule_B = []
    
    # Правило A: field == "TIME" И word содержит "ДВЕТЫСЯЧИДВАДЦАТЬГОДА"
    # Правило B: field == "PSY" И (word == "ДВЕТЫСЯЧИДВАДЦАТЬПЯТОГОГОДА" 
    #            ИЛИ word == "ДВЕТЫСЯЧИДВАДЦАТЬПЯТЫЙГОД"
    #            ИЛИ word содержит "ДЕКАБРЬДВЕТЫСЯЧИДВАДЦАТЬПЯТОГОГОДА")
    
    # Фильтрация элементов
    cleaned_items = []
    
    for item in items:
        word = item.get('word', '')
        field = item.get('field', '')
        
        # Проверка правила A
        if field == "TIME" and "ДВЕТЫСЯЧИДВАДЦАТЬГОДА" in word:
            removed_rule_A.append(word)
            continue
        
        # Проверка правила B
        if field == "PSY":
            if (word == "ДВЕТЫСЯЧИДВАДЦАТЬПЯТОГОГОДА" or
                word == "ДВЕТЫСЯЧИДВАДЦАТЬПЯТЫЙГОД" or
                "ДЕКАБРЬДВЕТЫСЯЧИДВАДЦАТЬПЯТОГОГОДА" in word):
                removed_rule_B.append(word)
                continue
        
        # Элемент проходит фильтрацию
        cleaned_items.append(item)
    
    removed_count_A = len(removed_rule_A)
    removed_count_B = len(removed_rule_B)
    removed_count_total = removed_count_A + removed_count_B
    new_total_words = len(cleaned_items)
    
    print(f"Удалено по правилу A: {removed_count_A}")
    print(f"Удалено по правилу B: {removed_count_B}")
    print(f"Всего удалено: {removed_count_total}")
    print(f"Осталось записей: {new_total_words}")
    
    # Обновление meta
    # Сохраняем исходный meta как baseline (до добавления новых полей)
    baseline = meta.copy()
    
    # Добавляем новые поля в meta
    # Сначала сохраняем baseline
    meta['baseline'] = baseline
    
    # Затем добавляем информацию об удалении
    meta['removed_count_total'] = removed_count_total
    meta['removed_count_rule_A'] = removed_count_A
    meta['removed_count_rule_B'] = removed_count_B
    meta['new_total_words'] = new_total_words
    
    # Примеры удалённых слов (до 10 для каждого правила)
    removed_examples = {}
    if removed_rule_A:
        removed_examples['rule_A'] = removed_rule_A[:10]
    if removed_rule_B:
        removed_examples['rule_B'] = removed_rule_B[:10]
    
    if removed_examples:
        meta['removed_examples'] = removed_examples
    
    # Формирование результирующего объекта
    result = {
        'meta': meta,
        'items': cleaned_items
    }
    
    # Проверки
    print("\nВыполнение проверок...")
    
    # Проверка 1: нет TIME-word с "ДВЕТЫСЯЧИДВАДЦАТЬГОДА"
    time_with_wrong_year = [item for item in cleaned_items 
                            if item.get('field') == 'TIME' 
                            and 'ДВЕТЫСЯЧИДВАДЦАТЬГОДА' in item.get('word', '')]
    if time_with_wrong_year:
        print(f"ОШИБКА: Найдено {len(time_with_wrong_year)} TIME-слов с 'ДВЕТЫСЯЧИДВАДЦАТЬГОДА'")
    else:
        print("✓ Проверка 1: Нет TIME-слов с 'ДВЕТЫСЯЧИДВАДЦАТЬГОДА'")
    
    # Проверка 2: нет PSY-word из правила B
    psy_rule_B_words = []
    for item in cleaned_items:
        if item.get('field') == 'PSY':
            word = item.get('word', '')
            if (word == "ДВЕТЫСЯЧИДВАДЦАТЬПЯТОГОГОДА" or
                word == "ДВЕТЫСЯЧИДВАДЦАТЬПЯТЫЙГОД" or
                "ДЕКАБРЬДВЕТЫСЯЧИДВАДЦАТЬПЯТОГОГОДА" in word):
                psy_rule_B_words.append(word)
    
    if psy_rule_B_words:
        print(f"ОШИБКА: Найдено {len(psy_rule_B_words)} PSY-слов по правилу B")
    else:
        print("✓ Проверка 2: Нет PSY-слов по правилу B")
    
    # Проверка 3: все элементы имеют tone == "2025-12" и sphere == "Quantum Prognosis Monthly"
    wrong_tone = [item for item in cleaned_items if item.get('tone') != '2025-12']
    wrong_sphere = [item for item in cleaned_items 
                   if item.get('sphere') != 'Quantum Prognosis Monthly']
    
    if wrong_tone:
        print(f"ОШИБКА: Найдено {len(wrong_tone)} элементов с неверным tone")
    else:
        print("✓ Проверка 3a: Все элементы имеют tone == '2025-12'")
    
    if wrong_sphere:
        print(f"ОШИБКА: Найдено {len(wrong_sphere)} элементов с неверным sphere")
    else:
        print("✓ Проверка 3b: Все элементы имеют sphere == 'Quantum Prognosis Monthly'")
    
    # Сохранение в файл
    print(f"\nСохранение в файл: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Готово!")
    
    return result

if __name__ == '__main__':
    input_file = '/Users/airos/Desktop/AI ANGEL/Cursor Quantum/QuantumEncoder/qpm_2025-12_tokenkind.json'
    output_file = '/Users/airos/Desktop/AI ANGEL/Cursor Quantum/QuantumEncoder/qpm_2025-12_clean.json'
    
    clean_anchors(input_file, output_file)

