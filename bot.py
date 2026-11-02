import os
import re
import warnings
import cv2
import numpy as np
import telebot
import easyocr
from scipy.stats import poisson

# Подавляем технические предупреждения PyTorch и Deprecation Warnings для чистоты логов
warnings.filterwarnings("ignore", category=UserWarning)

TOKEN = "9223011917:AAF2aonZ5kaX9TkD7fx0kz6U2gFzc52Cx_Y"
bot = telebot.TeleBot(TOKEN)

print("Инициализация EasyOCR движка...")
reader = easyocr.Reader(['ru', 'en'], gpu=False)
print("Движок OCR успешно запущен.")

def parse_screenshot(image_bytes: bytes) -> dict:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    results = reader.readtext(img, detail=1)
    
    elements = []
    match_title = "Не распознано"
    form_tokens = []
    
    print("\n================ [OCR DEBUG: НАЧАЛО РАСПОЗНАВАНИЯ СКРИНШОТА] ================")
    if not results:
        print("⚠️ ВНИМАНИЕ: EasyOCR вернул абсолютно пустой результат (ничего не смог прочитать)!")
    
    # 1. Собираем все элементы с их точными координатами
    for bbox, text, prob in results:
        y_min = int(bbox[0][1])
        x_min = int(bbox[0][0])
        clean_txt = text.strip()
        elements.append({'y': y_min, 'x': x_min, 'text': clean_txt})
        print(f"📍 Y:{y_min:4d} | X:{x_min:4d} | Текст: '{clean_txt}' (уверенность: {prob:.2f})")
        
        # Название матча (ищем в верхней части с дефисом)
        if "-" in clean_txt and len(clean_txt) > 6 and y_min < 350:
            match_title = clean_txt
            
        # Форма команд
        if clean_txt.upper() in ['В', 'Н', 'П', 'W', 'D', 'L'] and len(clean_txt) == 1:
            form_tokens.append(clean_txt.upper())

    print("----------------------------------------------------------------------------")

    # Сортируем элементы сверху вниз
    elements.sort(key=lambda e: e['y'])
    
    # 2. Ищем координаты ключевых заголовков (якорей блоков)
    anchors = {}
    for el in elements:
        t_low = el['text'].lower()
        if ('1x2' in t_low or 'исход' in t_low) and '1x2' not in anchors:
            anchors['1x2'] = el['y']
        elif 'тотал' in t_low and 'total' not in anchors and 'комбинирован' not in t_low:
            anchors['total'] = el['y']
        elif 'фора' in t_low and 'handicap' not in anchors:
            anchors['handicap'] = el['y']
        elif ('комбинирован' in t_low or 'исход и тотал' in t_low) and 'combo' not in anchors:
            anchors['combo'] = el['y']

    print(f"🎯 [OCR DEBUG: Найденные текстовые якоря блоков] -> {anchors}")

    if '1x2' not in anchors:
        print("❌ ОШИБКА ПАРСИНГА: Обязательный якорь '1x2' / 'Исход' не обнаружен на скриншоте.")
        print("================ [OCR DEBUG: КОНЕЦ РАСПОЗНАВАНИЯ (ПРЕРВАНО)] ================\n")
        raise ValueError("На скриншоте не найден блок 'Исход 1X2'. Откройте основную роспись матча.")
        
    y_1x2 = anchors['1x2']
    y_total = anchors.get('total', y_1x2 + 250)
    y_hc = anchors.get('handicap', y_total + 250)
    y_combo = anchors.get('combo', y_hc + 250)
    
    # Универсальная функция извлечения чисел (коэффициентов) в заданном вертикальном диапазоне
    def extract_odds_in_zone(y_start, y_end, zone_name=""):
        zone_items = []
        for e in elements:
            if y_start <= e['y'] < y_end:
                clean_t = e['text'].replace(',', '.')
                found_nums = re.findall(r'\b\d{1,3}\.\d{2}\b', clean_t)
                for num_str in found_nums:
                    try:
                        val = float(num_str)
                        if 1.01 <= val <= 50.0:
                            zone_items.append((e['x'], val))
                    except ValueError:
                        continue
        # Сортируем слева направо по оси X
        zone_items.sort(key=lambda item: item[0])
        extracted = [item[1] for item in zone_items]
        print(f"📊 [ZONA {zone_name}] Y от {y_start} до {y_end} -> Найденные котировки: {extracted}")
        return extracted

    # Нижняя граница с запасом для сбора всего экрана
    max_y = max([e['y'] for e in elements]) + 100 if elements else 1000

    # 3. Извлекаем блоки по зонам
    print("--- [Детальный разбор зон росписи] ---")
    odds_1x2 = extract_odds_in_zone(y_1x2, y_total, "1X2")
    if len(odds_1x2) < 3:
        print(f"❌ ОШИБКА ПАРСИНГА 1X2: Найдено котировок: {len(odds_1x2)}. Требуется ровно 3.")
        print("================ [OCR DEBUG: КОНЕЦ РАСПОЗНАВАНИЯ (ПРЕРВАНО)] ================\n")
        raise ValueError(f"Ошибка парсинга 1X2: найдено коэффициентов: {len(odds_1x2)}. Требуется ровно 3.")
    
    k_p1, k_x, k_p2 = odds_1x2[0], odds_1x2[1], odds_1x2[2]

    # Тоталы (берем ТМ 2.5 и ТБ 2.5 из блока тоталов)
    odds_totals = extract_odds_in_zone(y_total, y_hc, "TOTALS")
    k_tu25 = odds_totals[2] if len(odds_totals) > 2 else (odds_totals[0] if len(odds_totals) > 0 else None)
    k_tb25 = odds_totals[5] if len(odds_totals) > 5 else (odds_totals[1] if len(odds_totals) > 1 else None)

    # Форы (захватываем все котировки из зоны фор)
    odds_handicaps = extract_odds_in_zone(y_hc, y_combo, "HANDICAPS")

    # Комбо-ставки (исход и тотал)
    odds_combos = extract_odds_in_zone(y_combo, max_y, "COMBOS")

    print("----------------------------------------------------------------------------")
    print(f"✅ [ИТОГ ПАРСИНГА] Матч: '{match_title}' | 1X2: [{k_p1}, {k_x}, {k_p2}] | ТМ2.5: {k_tu25} | Т
