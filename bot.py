import os
import re
import warnings
import traceback
import cv2
import numpy as np
import telebot
import easyocr
from scipy.stats import poisson

# Подавляем технические предупреждения PyTorch и Депрекационные варнинги для чистоты логов
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
    if not results:
        pass
    
    for bbox, text, prob in results:
        y_min = int(bbox[0][1])
        x_min = int(bbox[0][0])
        x_max = int(bbox[2][0])
        clean_txt = text.strip()
        elements.append({'y': y_min, 'x': x_min, 'x_max': x_max, 'text': clean_txt})

    elements.sort(key=lambda e: e['y'])
    screen_center_x = img.shape[1] / 2

    # 1. ЖЕЛЕЗОБЕТОННЫЙ ПОИСК НАЗВАНИЯ МАТЧА В ШАПКЕ
    match_title = "Команда 1 — Команда 2"
    team_candidates = []
    
    # Ищем название под логотипами (y от 120 до 220)
    for e in elements:
        if 120 <= e['y'] <= 220:
            txt = e['text'].strip()
            txt_low = txt.lower()
            # Отсекаем мусор, счетчик баланса, время и слова БК
            if any(w in txt_low for w in ['тур', 'сегодня', 'вчера', 'трансляция', 'футбол', 'лига', 'матч', 'основное', 'винлайн', 'winline', '₽', '33', '35']):
                continue
            if re.search(r'\d{2}:\d{2}', txt) or len(txt) <= 1:
                continue
            team_candidates.append(e)

    if len(team_candidates) >= 2:
        team_candidates.sort(key=lambda item: item['x'])
        t1_name = team_candidates[0]['text']
        t2_name = team_candidates[-1]['text']
        match_title = f"{t1_name} — {t2_name}"
    else:
        # Запасной вариант: поиск единственной строки с дефисом в шапке
        for e in elements:
            if 50 <= e['y'] <= 250 and re.search(r'\s+[-—–]\s+', e['text']):
                txt = e['text']
                if not re.search(r'\d{2}:\d{2}', txt) and not any(w in txt.lower() for w in ['тур', 'сегодня', 'вчера', 'футбол']):
                    parts = re.split(r'\s+[-—–]\s+', txt)
                    if len(parts) >= 2:
                        match_title = f"{parts[0].strip()} — {parts[1].strip()}"
                        break

    # 2. ИЩЕМ ГЛАВНЫЙ ЯКОРЬ: «Исход 1X2»
    y_1x2 = None
    for el in elements:
        t_low = el['text'].lower().replace(" ", "").replace(".", "")
        if 'исход1x2' in t_low or 'исход1х2' in t_low or t_low in ['1x2', '1х2']:
            y_1x2 = el['y']
            break

    if y_1x2 is None:
        y_1x2 = 450

    # ==========================================
    # СКАНИРОВАНИЕ ФОРМЫ КОМАНД
    # ==========================================
    form_debug_lines = [f"<b>[DEBUG FORM] Матч: {match_title}</b>", f"Якорь 1X2 найден на y={y_1x2}"]
    form_allowed_chars = ['В', 'Н', 'П', 'V', 'H', 'N', 'P', 'B', 'X', 'W', 'D', 'Л']
    t1_form_raw = []
    t2_form_raw = []

    for e in elements:
        if 130 <= e['y'] < y_1x2 - 10:
            txt_raw = e['text'].strip()
            txt_lower = txt_raw.lower()
            if any(w in txt_lower for w in ['сегодня', 'вчера', 'трансляция', 'тур', 'лига', 'футбол']):
                continue
            if re.search(r'\d{2}:\d{2}', txt_raw) or re.search(r'\d{1,2}\.\d{1,2}', txt_raw):
                continue
            
            form_debug_lines.append(f"y={e['y']}, x={e['x']}, txt: '{txt_raw}'")
            cleaned_box_text = txt_raw.upper().replace(' ', '')
            for char in cleaned_box_text:
                if char in form_allowed_chars:
                    clean_char = (char.replace('V', 'В')
                                      .replace('H', 'Н')
                                      .replace('N', 'Н')
                                      .replace('P', 'П')
                                      .replace('B', 'В')
                                      .replace('W', 'В')
                                      .replace('D', 'Н')
                                      .replace('X', 'Н')
                                      .replace('Л', 'П'))
                    if clean_char in ['В', 'Н', 'П']:
                        if e['x'] < screen_center_x:
                            t1_form_raw.append({'x': e['x'], 'char': clean_char})
                        else:
                            t2_form_raw.append({'x': e['x'], 'char': clean_char})

    t1_form_raw.sort(key=lambda item: item['x'])
    t2_form_raw.sort(key=lambda item: item['x'])

    t1_form = [item['char'] for item in t1_form_raw][-5:]
    t2_form = [item['char'] for item in t2_form_raw][-5:]

    form_debug_lines.append(f"<b>ИТОГ Т1:</b> {t1_form} | <b>ИТОГ Т2:</b> {t2_form}")

    def extract_odds_in_zone(y_start, y_end):
        zone_items = []
        for e in elements:
            if y_start <= e['y'] < y_end:
                clean_t = e['text'].replace(',', '.').replace('․', '.').replace('·', '.').replace(' ', '')
                found_nums = re.findall(r'\d{1,3}\.\d{1,2}', clean_t)
                for num_str in found_nums:
                    try:
                        val = float(num_str)
                        if 1.01 <= val <= 50.0:
                            zone_items.append((e['x'], val))
                    except ValueError:
                        continue
        zone_items.sort(key=lambda item: item[0])
        return [item[1] for item in zone_items]

    odds_1x2 = extract_odds_in_zone(y_1x2, y_1x2 + 80)
    if len(odds_1x2) < 3:
        odds_1x2 = extract_odds_in_zone(y_1x2 - 40, y_1x2 + 120)
    
    if len(odds_1x2) < 3:
        raise ValueError(f"Ошибка парсинга 1X2: найдено коэффициентов: {len(odds_1x2)}. Требуется ровно 3.")
    
    k_p1, k_x, k_p2 = odds_1x2[0], odds_1x2[1], odds_1x2[2]

    # 3. ПОИСК ЯКОРЕЙ ВСЕХ БЛОКОВ НИЖЕ 1X2
    block_anchors = []
    for e in elements:
        if e['y'] > y_1x2:
            t_low = e['text'].lower().replace(" ", "")
            if 'исходтотал' in t_low or 'результатитотал' in t_low or ('исход' in t_low and 'тотал' in t_low):
                block_anchors.append({'type': 'combo', 'y': e['y']})
            elif 'тотал' in t_low:
                block_anchors.append({'type': 'total', 'y': e['y']})
            elif 'фора' in t_low:
                block_anchors.append({'type': 'handicap', 'y': e['y']})

    block_anchors.sort(key=lambda b: b['y'])

    # 4. СКАНИРУЕМ БЛОК «ТОТАЛ»
    scanned_totals = []
    total_anchor = next((b for b in block_anchors if b['type'] == 'total'), None)
    if total_anchor:
        next_anchors = [b['y'] for b in block_anchors if b['y'] > total_anchor['y']]
        total_end_y = min(next_anchors) if next_anchors else img.shape[0]

        for e in elements:
            if total_anchor['y'] < e['y'] < total_end_y:
                clean_t = e['text'].replace(',', '.').replace(' ', '')
                if re.match(r'^[0-5]\.5$', clean_t):
                    try:
                        line_val = float(clean_t)
                        row_odds = []
                        for item in elements:
                            if abs(item['y'] - e['y']) <= 12 and total_anchor['y'] < item['y'] < total_end_y:
                                c_text = item['text'].replace(',', '.').replace(' ', '')
                                found = re.findall(r'\d{1,3}\.\d{1,2}', c_text)
                                for f_num in found:
                                    try:
                                        val = float(f_num)
                                        if val == line_val and abs(item['x'] - e['x']) < 15:
                                            continue
                                        if 1.01 <= val <= 50.0:
                                            row_odds.append((item['x'], val))
                                    except ValueError:
                                        continue
                        row_odds.sort(key=lambda x: x[0])
                        left_vals = [v[1] for v in row_odds if v[0] < e['x']]
                        right_vals = [v[1] for v in row_odds if v[0] > e['x']]
                        if left_vals and right_vals:
                            if not any(t['line'] == line_val for t in scanned_totals):
                                scanned_totals.append({
                                    "line": line_val,
                                    "tu": left_vals[-1],
                                    "tb": right_vals[0]
                                })
                    except ValueError:
                        continue

    # 5. СКАНИРУЕМ БЛОК «ФОРА» (ОБНОВЛЕННЫЙ АЛГОРИТМ С ИГНОРИРОВАНИЕМ МУСОРА)
    scanned_handicaps = []
    seen_handicap_lines = set()
    handicap_anchor = next((b for b in block_anchors if b['type'] == 'handicap'), None)

    if handicap_anchor:
        next_anchors = [b['y'] for b in block_anchors if b['y'] > handicap_anchor['y']]
        handicap_end_y = min(next_anchors) if next_anchors else img.shape[0]

        rows_map = {}
        for e in elements:
            if handicap_anchor['y'] < e['y'] < handicap_end_y:
                txt_low = e['text'].lower()
                if any(w in txt_low for w in ['свернуть', 'развернуть', 'маркет', 'событий', 'live', 'главная', 'сейчас', 'история', 'игры', 'кибер', 'профиль']):
                    continue
                
                found_row_y = None
                for ry in rows_map.keys():
                    if abs(ry - e['y']) <= 10:
                        found_row_y = ry
                        break
                if found_row_y is None:
                    found_row_y = e['y']
                    rows_map[found_row_y] = []
                rows_map[found_row_y].append(e)

        for row_y, row_elems in rows_map.items():
            row_elems.sort(key=lambda item: item['x'])
            row_full_text = " ".join([el['text'] for el in row_elems])
            
            # ДЕБАГ-ЛОГ СТРОКИ ФОРЫ
            form_debug_lines.append(f"[DEBUG HC ROW] y={row_y} | текст: '{row_full_text}'")

            val = None
            h1_val, h2_val = 0.0, 0.0

            if re.search(r'\b0\b', row_full_text) and not re.search(r'[1-9]', row_full_text.replace('0', '', 1)):
                val = 0.0
                h1_val, h2_val = 0.0, 0.0
            else:
                match_handicap = re.search(r'([+-]\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*[+-])', row_full_text)
                if match_handicap:
                    raw_hc_str = match_handicap.group(1).replace(' ', '')
                    extracted_nums = re.findall(r'\d+(?:\.\d+)?', raw_hc_str)
                    if extracted_nums:
                        try:
                            val = float(extracted_nums[0])
                        except ValueError:
                            continue

                        if val > 5.0 or val == 0:
                            continue

                        is_h1_negative = False
                        if '-' in raw_hc_str:
                            if raw_hc_str.startswith('-') or raw_hc_str.endswith('+'):
                                is_h1_negative = True
                        else:
                            for el in row_elems:
                                if '-' in el['text'] and el['x'] < screen_center_x + 30:
                                    is_h1_negative = True
                                    break

                        if is_h1_negative:
                            h1_val, h2_val = -val, val
                        else:
                            h1_val, h2_val = val, -val

            if val is None or h1_val in seen_handicap_lines:
                continue

            row_odds = []
            for el in row_elems:
                c_text = el['text'].replace(',', '.').replace(' ', '')
                found = re.findall(r'\d{1,3}\.\d{1,2}', c_text)
                for f_num in found:
                    try:
                        f_val = float(f_num)
                        if f_val == val:
                            continue
                        if 1.01 <= f_val <= 50.0:
                            row_odds.append((el['x'], f_val))
                    except ValueError:
                        continue

            row_odds.sort(key=lambda x: x[0])
            left_vals = [v[1] for v in row_odds if v[0] < screen_center_x]
            right_vals = [v[1] for v in row_odds if v[0] > screen_center_x]

            # ДЕБАГ-ЛОГ НАЙДЕННЫХ ЗНАЧЕНИЙ
            form_debug_lines.append(f"-> Распознано: Ф1={h1_val:+.1f} (кэфы слева: {left_vals}), Ф2={h2_val:+.1f} (кэфы справа: {right_vals})")

            if left_vals and right_vals:
                scanned_handicaps.append({
                    "h1_val": h1_val,
                    "h2_val": h2_val,
                    "h1": left_vals[0],
                    "h2": right_vals[-1]
                })
                seen_handicap_lines.add(h1_val)

    form_log_str = "\n".join(form_debug_lines)

    if len(t1_form) >= 3 and len(t2_form) >= 3:
        form_tokens = t1_form + t2_form
    else:
        form_tokens = []

    k_tu25, k_tb25 = None, None
    for t in scanned_totals:
        if t["line"] == 2.5:
            k_tu25, k_tb25 = t["tu"], t["tb"]
            break
    if not k_tu25 and scanned_totals:
        k_tu25, k_tb25 = scanned_totals[0]["tu"], scanned_totals[0]["tb"]

    return {
        "match": match_title,
        "k_p1": k_p1, "k_x": k_x, "k_p2": k_p2,
        "k_tu25": k_tu25, "k_tb25": k_tb25,
        "scanned_totals": scanned_totals,
        "scanned_handicaps": scanned_handicaps,
        "form_tokens": form_tokens,
        "form_log": form_log_str
    }


def execute_v31_audit(data: dict) -> str:
    k1, kx, k2 = data["k_p1"], data["k_x"], data["k_p2"]
    
    try:
        implied_p1 = 1 / k1
        implied_kx = 1 / kx
        implied_p2 = 1 / k2
        sum_1x2 = implied_p1 + implied_kx + implied_p2
        
        p1_norm = implied_p1 / sum_1x2
        p2_norm = implied_p2 / sum_1x2
        
        if data["k_tu25"] and data["k_tb25"]:
            impl_tb = 1 / data["k_tb25"]
            impl_tm = 1 / data["k_tu25"]
            xG_total = max(1.0, min(5.0, 2.5 * (impl_tb / (impl_tb + impl_tm) + 0.5)))
        else:
            xG_total = 2.50
            
        xG_1 = max(0.2, xG_total * (p1_norm / (p1_norm + p2_norm + 0.001)))
        xG_2 = max(0.2, xG_total - xG_1)
    except ZeroDivisionError:
        xG_total, xG_1, xG_2 = 2.50, 1.25, 1.25

    grid = np.zeros((10, 10))
    score_probs = []
    for i in range(10):
        for j in range(10):
            prob = poisson.pmf(i, xG_1) * poisson.pmf(j, xG_2) * 100
            grid[i, j] = prob
            score_probs.append(((i, j), prob))
            
    score_probs.sort(key=lambda x: x[1], reverse=True)
    
    top2_prob = score_probs[0][1] + score_probs[1][1]
    top3_prob = top2_prob + score_probs[2][1]
    top4_prob = top3_prob + score_probs[3][1]
    tail_prob = 100.0 - top4_prob

    cascade_passed = (top2_prob <= 46.0) and (top3_prob <= 62.0) and (top4_prob <= 72.0) and (tail_prob >= 28.0)
    if 69.0 <= top4_prob <= 72.0 and (top4_prob + 3.0) > 72.0:
        cascade_passed = False

    teams = re.split(r'\s+[-—–]\s+', data['match'])
    t1 = teams[0].strip() if len(teams) > 0 and teams[0] else "Команда 1"
    t2 = teams[1].strip() if len(teams) > 1 and teams[1] else "Команда 2"

    markets_to_test = [
        {"name": "Победа 1 (П1)", "odds": k1, "real_prob": sum(grid[i, j] for i in range(10) for j in range(10) if i > j)},
        {"name": "Ничья (Х)", "odds": kx, "real_prob": sum(grid[i, j] for i in range(10) for j in range(10) if i == j)},
        {"name": "Победа 2 (П2)", "odds": k2, "real_prob": sum(grid[i, j] for i in range(10) for j in range(10) if i < j)},
        
        {"name": "1X", "odds": round(100.0 / max(0.1, sum(grid[i, j] for i in range(10) for j in range(10) if i >= j)) * 1.05, 2), "real_prob": sum(grid[i, j] for i in range(10) for j in range(10) if i >= j)},
        {"name": "12", "odds": round(100.0 / max(0.1, sum(grid[i, j] for i in range(10) for j in range(10) if i != j)) * 1.05, 2), "real_prob": sum(grid[i, j] for i in range(10) for j in range(10) if i != j)},
        {"name": "X2", "odds": round(100.0 / max(0.1, sum(grid[i, j] for i in range(10) for j in range(10) if i <= j)) * 1.05, 2), "real_prob": sum(grid[i, j] for i in range(10) for j in range(10) if i <= j)},
    ]

    scanned_lines = {t["line"] for t in data.get("scanned_totals", [])}
    for t_item in data.get("scanned_totals", []):
        line = t_item["line"]
        k_tu = t_item["tu"]
        k_tb = t_item["tb"]
        markets_to_test.append({
            "name": f"Тотал меньше ({line})", 
            "odds": k_tu, 
            "real_prob": sum(grid[i, j] for i in range(10) for j in range(10) if (i + j) < line)
        })
        markets_to_test.append({
            "name": f"Тотал больше ({line})", 
            "odds": k_tb, 
            "real_prob": sum(grid[i, j] for i in range(10) for j in range(10) if (i + j) > line)
        })

    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
        if line not in scanned_lines:
            p_tu = sum(grid[i, j] for i in range(10) for j in range(10) if (i + j) < line)
            p_tb = sum(grid[i, j] for i in range(10) for j in range(10) if (i + j) > line)
            if p_tu > 0:
                markets_to_test.append({"name": f"Тотал меньше ({line})", "odds": round(100.0 / p_tu * 1.055, 2), "real_prob": p_tu})
            if p_tb > 0:
                markets_to_test.append({"name": f"Тотал больше ({line})", "odds": round(100.0 / p_tb * 1.055, 2), "real_prob": p_tb})

    for h_item in data.get("scanned_handicaps", []):
        h1_val = h_item["h1_val"]
        h2_val = h_item["h2_val"]
        
        # Фора 1
        if h1_val == 0.0:
            prob_h1 = sum(grid[i, j] for i in range(10) for j in range(10) if i > j) + (sum(grid[i, j] for i in range(10) for j in range(10) if i == j) * 0.5)
        elif h1_val.is_integer():
            h_int = int(h1_val)
            prob_win = sum(grid[i, j] for i in range(10) for j in range(10) if (i - j) > -h_int)
            prob_push = sum(grid[i, j] for i in range(10) for j in range(10) if (i - j) == -h_int)
            prob_h1 = prob_win + (prob_push * 0.5)
        else:
            prob_h1 = sum(grid[i, j] for i in range(10) for j in range(10) if (i - j) > -h1_val)

        markets_to_test.append({
            "name": f"Фора 1 ({h1_val:+.1f})", 
            "odds": h_item["h1"], 
            "real_prob": prob_h1
        })

        # Фора 2
        if h2_val == 0.0:
            prob_h2 = sum(grid[i, j] for i in range(10) for j in range(10) if j > i) + (sum(grid[i, j] for i in range(10) for j in range(10) if i == j) * 0.5)
        elif h2_val.is_integer():
            h_int = int(h2_val)
            prob_win = sum(grid[i, j] for i in range(10) for j in range(10) if (j - i) > -h_int)
            prob_push = sum(grid[i, j] for i in range(10) for j in range(10) if (j - i) == -h_int)
            prob_h2 = prob_win + (prob_push * 0.5)
        else:
            prob_h2 = sum(grid[i, j] for i in range(10) for j in range(10) if (j - i) > -h2_val)

        markets_to_test.append({
            "name": f"Фора 2 ({h2_val:+.1f})", 
            "odds": h_item["h2"], 
            "real_prob": prob_h2
        })

    combo_markets = [
        ("П1 и ТБ(2.5)", lambda i, j: i > j and (i + j) > 2.5),
        ("П1 и ТМ(2.5)", lambda i, j: i > j and (i + j) < 2.5),
        ("П2 и ТБ(2.5)", lambda i, j: i < j and (i + j) > 2.5),
        ("П2 и ТМ(2.5)", lambda i, j: i < j and (i + j) < 2.5),
        ("Обе забьют (ОЗ) и ТБ(2.5)", lambda i, j: i > 0 and j > 0 and (i + j) > 2.5),
        ("Обе забьют (ОЗ) и ТМ(2.5)", lambda i, j: i > 0 and j > 0 and (i + j) < 2.5),
    ]
    for combo_name, condition in combo_markets:
        p_combo = sum(grid[i, j] for i in range(10) for j in range(10) if condition(i, j))
        if p_combo > 0:
            markets_to_test.append({
                "name": combo_name,
                "odds": round(100.0 / p_combo * 1.085, 2),
                "real_prob": p_combo
            })

    best_pick = None
    max_margin_val = -999.0
    final_status = "РАБОЧИЙ ВАЛУЙ"

    if data["k_tu25"] and data["k_tu25"] < 1.75:
        vector_lock = "Низовой"
    elif data["k_tb25"] and data["k_tb25"] < 1.75:
        vector_lock = "Верховой"
    else:
        vector_lock = "Стандартный"

    if not cascade_passed:
        final_status = "МАТЧ РЕКОМЕНДОВАН К ПРОПУСКУ (SKIP) [Нарушен каскадный лимит матрицы Пуассона]"
    else:
        for m in markets_to_test:
            odds = m["odds"]
            
            if odds > 5.50 or odds < 1.85:
                continue

            if vector_lock == "Низовой" and "Тотал больше" in m["name"]:
                continue
            if vector_lock == "Верховой" and "Тотал меньше" in m["name"]:
                continue

            real_prob = m["real_prob"]
            dynamic_penalty = max(1.5, min(4.5, (100 - real_prob) * 0.04))
            real_prob_min = real_prob - dynamic_penalty
            implied_prob = (1 / odds) * 100
            
            if odds >= 2.10:
                vm_min = real_prob_min - implied_prob
                if vm_min >= 1.50 and vm_min > max_margin_val:
                    max_margin_val = vm_min
                    best_pick = (m["name"], odds, vm_min, real_prob, real_prob_min, implied_prob)
                    final_status = "ЖЕЛЕЗНЫЙ ВАЛУЙ" if vm_min >= 6.0 else "РАБОЧИЙ ВАЛУЙ"
            else:
                if real_prob_min > (implied_prob + 6.0):
                    excess = real_prob_min - (implied_prob + 6.0)
                    if excess > max_margin_val:
                        max_margin_val = excess
                        best_pick = (m["name"], odds, real_prob_min, real_prob, real_prob_min, implied_prob)
                        final_status = "ЖЕЛЕЗНЫЙ ВАЛУЙ (ПЛАН Б)"

        if not best_pick:
            final_status = "МАТЧ РЕКОМЕНДОВАН К ПРОПУСКУ (SKIP) [Нет подтвержденного валуя по всему массиву рынков или сработал защитный фильтр вектора]"

    form_bonus = 0
    if data["form_tokens"]:
        wins_count = data["form_tokens"].count('В')
        losses_count = data["form_tokens"].count('П')
        form_bonus = (wins_count - losses_count) * 2

    totals_str = ", ".join([f"Т({t['line']}): М={t['tu']}/Б={t['tb']}" for t in data["scanned_totals"]]) if data["scanned_totals"] else "Не найдены"
    hc_str = ", ".join([f"Ф1({h['h1_val']:+.1f}):{h['h1']} / Ф2({h['h2_val']:+.1f}):{h['h2']}" for h in data["scanned_handicaps"]]) if data["scanned_handicaps"] else "Не найдены"
    form_str = ", ".join(data["form_tokens"]) if data["form_tokens"] else "Не считывались"

    if best_pick:
        market_name, chosen_odds, val_score, r_prob, r_prob_min, imp_prob = best_pick
        confidence = min(95, max(60, int(75 + (max_margin_val if max_margin_val > 0 else val_score) * 2 + form_bonus)))
        stake_pct = 5 if confidence >= 80 else 3
    else:
        market_name, chosen_odds, r_prob, r_prob_min, imp_prob = "Отсутствует", 0.0, 0.0, 0.0, 0.0
        confidence = 0
        stake_pct = 0

    report = f"""{data.get("form_log", "")}

<b>Шаг 0: Калибровка и привязка котировок</b>
- <b>Матч</b>: {t1} — {t2}
- <b>1X2</b>: П1={k1}, Х={kx}, П2={k2}
- <b>Тоталы (все линии)</b>: [{totals_str}]
- <b>Форы (все линии)</b>: [{hc_str}]
- <b>Форма команд</b>: [{form_str}] (Бонус: {form_bonus:+d}%)
- xG: {t1} ({xG_1:.2f}) / {t2} ({xG_2:.2f}) | Сумма: {xG_total:.2f}
- Вектор матча: {vector_lock}

<b>Шаг 1: Матрица Пуассона и каскадный контроль</b>
- ТОП-2 счетов: {top2_prob:.1f}% (Лимит &le; 46%)
- ТОП-3 счетов: {top3_prob:.1f}% (Лимит &le; 62%)
- ТОП-4 счетов: {top4_prob:.1f}% (Лимит &le; 72%)
- Статус каскада: {"Пройдено успешно" if cascade_passed else "Нарушен лимит"}

<b>Шаг 2 &amp; 3: Массив рынков и стресс-тест буфера (Динамический штраф)</b>
- Проверен рынок: <b>{market_name}</b> (Коэф: {chosen_odds})
- Real Prob: {r_prob:.1f}% | Real Prob(min с учетом динамического буфера): {r_prob_min:.1f}%
- Implied Prob БК: {imp_prob:.1f}%

<b>Шаг 4 &amp; 5: Уверенность и финансовый менеджмент</b>
- Индекс уверенности: {confidence}%
- Рекомендуемый банк: {stake_pct}% от общего банка
- Статус: [СТАТУС: {final_status}]

<b>Шаг 6: ИТОГОВЫЙ ПРИКАЗ ПО СТАВКЕ</b>"""

    if best_pick:
        if chosen_odds >= 2.10:
            final_line = f"\n{t1} — {t2} | {market_name} — {chosen_odds:.2f} — {stake_pct}% от банка (Уверенность: {confidence}%) | [СТАТУС: {final_status}]"
        else:
            final_line = f"\n{t1} — {t2} | Альтернативная ставка по Плану Б: {market_name} — {chosen_odds:.2f} — {stake_pct}% от банка (Уверенность: {confidence}%) | [СТАТУС: {final_status}]"
    else:
        final_line = f"\n{t1} — {t2} | МАТЧ РЕКОМЕНДОВАН К ПРОПУСКУ (SKIP)"

    return report + final_line


@bot.message_handler(commands=['start'])
def func_start(message):
    bot.reply_to(message, "Аудитор v3.1 запущен. Отправьте скриншот линии.")


@bot.message_handler(content_types=['photo'])
def func_handle_photo(message):
    processing_msg = bot.reply_to(message, "🛡️ Аудит видимых рынков по модели v3.1...")
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        parsed_data = parse_screenshot(downloaded_file)
        result_verdict = execute_v31_audit(parsed_data)
        
        bot.edit_message_text(result_verdict, chat_id=message.chat.id, message_id=processing_msg.message_id, parse_mode='HTML')
    except Exception as e:
        tb_str = traceback.format_exc()
        error_message = f"❌ Ошибка аудита:\n<pre>{tb_str}</pre>"
        if len(error_message) > 4000:
            error_message = error_message[:4000] + "\n...[обрезано]"
        bot.edit_message_text(error_message, chat_id=message.chat.id, message_id=processing_msg.message_id, parse_mode='HTML')


if __name__ == "__main__":
    print("Бот v3.1 успешно запущен и готов к работе...")
    bot.infinity_polling()
