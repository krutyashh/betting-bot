import os
import re
import warnings
import traceback
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
    if not results:
        pass
    
    for bbox, text, prob in results:
        y_min = int(bbox[0][1])
        x_min = int(bbox[0][0])
        x_max = int(bbox[2][0])
        clean_txt = text.strip()
        elements.append({'y': y_min, 'x': x_min, 'x_max': x_max, 'text': clean_txt})

    # Сортируем элементы сверху вниз
    elements.sort(key=lambda e: e['y'])
    
    # ПРАВИЛО 2: В шапке матча идем снизу вверх по строкам, ищем строку с командами по разделителю "пробел дефис пробел"
    match_title = "Команда 1 — Команда 2"
    header_elements = [e for e in elements if 50 <= e['y'] <= 350]
    header_elements_bottom_up = sorted(header_elements, key=lambda e: e['y'], reverse=True)
    
    screen_center_x = img.shape[1] / 2

    for e in header_elements_bottom_up:
        txt = e['text']
        # Ищем строку с пробел + дефис + пробел
        if re.search(r'\s+[-—–]\s+', txt):
            if not re.search(r'\d{2}:\d{2}', txt) and not re.search(r'(тур|часть|сегодня|вчера)', txt.lower()):
                parts = re.split(r'\s+[-—–]\s+', txt)
                if len(parts) >= 2:
                    match_title = f"{parts[0].strip()} — {parts[1].strip()}"
                    break

    # ПРАВИЛО 3: Ищем буквы формы (п, в, н / w, d, l), разделенные пробелами в зеленой зоне шапки (снизу вверх)
    potential_form_chars = ['В', 'Н', 'П', 'W', 'D', 'L', 'B']
    t1_form = []
    t2_form = []

    for e in header_elements_bottom_up:
        txt_clean = e['text'].upper().strip()
        if txt_clean in potential_form_chars and len(txt_clean) == 1:
            char = txt_clean.replace('B', 'В')
            x_pos = e['x']
            if x_pos < screen_center_x:
                if len(t1_form) < 5:
                    t1_form.append(char)
            else:
                if len(t2_form) < 5:
                    t2_form.append(char)

    # Так как шли снизу вверх, разворачиваем обратно для правильного хронологического порядка
    t1_form.reverse()
    t2_form.reverse()
    form_tokens = t1_form + t2_form

    # Ищем стартовый якорь блока 1X2
    y_1x2 = None
    for el in elements:
        t_low = el['text'].lower().replace(" ", "").replace(".", "")
        if 'исход1x2' in t_low or 'исход1х2' in t_low:
            y_1x2 = el['y']
            break

    if y_1x2 is None:
        y_1x2 = 350

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

    # Извлекаем 1X2 (обязательно)
    odds_1x2 = extract_odds_in_zone(y_1x2, y_1x2 + 100)
    if len(odds_1x2) < 3:
        odds_1x2 = extract_odds_in_zone(y_1x2 - 50, y_1x2 + 150)
    
    if len(odds_1x2) < 3:
        raise ValueError(f"Ошибка парсинга 1X2: найдено коэффициентов: {len(odds_1x2)}. Требуется ровно 3.")
    
    k_p1, k_x, k_p2 = odds_1x2[0], odds_1x2[1], odds_1x2[2]

    # Универсальный сканер тоталов по паттерну
    scanned_totals = []
    for e in elements:
        if e['y'] > y_1x2:
            clean_t = e['text'].replace(',', '.').replace(' ', '')
            if re.match(r'^[0-5]\.5$', clean_t):
                try:
                    line_val = float(clean_t)
                    row_odds = []
                    for item in elements:
                        if abs(item['y'] - e['y']) <= 12:
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

    # Универсальный сканер фор по паттерну (исправлен захват точных дробных значений с .5)
    scanned_handicaps = []
    for e in elements:
        if e['y'] > y_1x2:
            raw_txt = e['text'].replace(',', '.').replace(' ', '')
            # Ищем числа со знаками + или - и обязательным или опциональным .5 / целым значением
            if re.search(r'[-+]\d+(\.\d+)?', raw_txt):
                try:
                    match_num = re.search(r'([-+]\d+(\.\d+)?)', raw_txt)
                    if match_num:
                        h_val = float(match_num.group(1))
                        row_odds = []
                        for item in elements:
                            if abs(item['y'] - e['y']) <= 12:
                                c_text = item['text'].replace(',', '.').replace(' ', '')
                                found = re.findall(r'\d{1,3}\.\d{1,2}', c_text)
                                for f_num in found:
                                    try:
                                        val = float(f_num)
                                        if val == abs(h_val) and abs(item['x'] - e['x']) < 15:
                                            continue
                                        if 1.01 <= val <= 50.0:
                                            row_odds.append((item['x'], val))
                                    except ValueError:
                                        continue
                        row_odds.sort(key=lambda x: x[0])
                        left_vals = [v[1] for v in row_odds if v[0] < e['x']]
                        right_vals = [v[1] for v in row_odds if v[0] > e['x']]
                        if left_vals and right_vals:
                            if not any(h['line'] == h_val for h in scanned_handicaps):
                                scanned_handicaps.append({
                                    "line": h_val,
                                    "h1": left_vals[-1],
                                    "h2": right_vals[0]
                                })
                except (ValueError, IndexError):
                    continue

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
        "form_tokens": form_tokens
    }


def execute_v31_audit(data: dict) -> str:
    k1, kx, k2 = data["k_p1"], data["k_x"], data["k_p2"]
    
    if data["k_tu25"] and data["k_tb25"]:
        p_under = 1 / data["k_tu25"]
        xG_total = max(1.2, min(4.5, 3.2 - (p_under - 0.5) * 2.5))
    else:
        xG_total = 2.55
        
    power_diff = (k2 - k1) / (k1 + k2 + kx)
    xG_1 = max(0.3, (xG_total / 2) * (1 + power_diff * 0.8))
    xG_2 = max(0.3, xG_total - xG_1)

    grid = np.zeros((8, 8))
    score_probs = []
    for i in range(8):
        for j in range(8):
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

    teams = re.split(r'\s*[-—–]\s*', data['match'])
    t1 = teams[0].strip() if len(teams) > 0 and teams[0] else "Команда 1"
    t2 = teams[1].strip() if len(teams) > 1 and teams[1] else "Команда 2"

    if not cascade_passed:
        return f"{t1} — {t2} | МАТЧ РЕКОМЕНДОВАН К ПРОПУСКУ (SKIP) [Причина: Нарушен каскадный лимит матрицы Пуассона]"

    markets_to_test = [
        {"name": "Победа 1 (П1)", "odds": k1, "real_prob": sum(grid[i, j] for i in range(8) for j in range(8) if i > j)},
        {"name": "Ничья (Х)", "odds": kx, "real_prob": sum(grid[i, j] for i in range(8) for j in range(8) if i == j)},
        {"name": "Победа 2 (П2)", "odds": k2, "real_prob": sum(grid[i, j] for i in range(8) for j in range(8) if i < j)},
    ]

    for t_item in data.get("scanned_totals", []):
        line = t_item["line"]
        k_tu = t_item["tu"]
        k_tb = t_item["tb"]
        markets_to_test.append({
            "name": f"Тотал меньше ({line})", 
            "odds": k_tu, 
            "real_prob": sum(grid[i, j] for i in range(8) for j in range(8) if (i + j) < line)
        })
        markets_to_test.append({
            "name": f"Тотал больше ({line})", 
            "odds": k_tb, 
            "real_prob": sum(grid[i, j] for i in range(8) for j in range(8) if (i + j) > line)
        })

    for h_item in data.get("scanned_handicaps", []):
        h_val = h_item["line"]
        markets_to_test.append({
            "name": f"Фора 1 ({h_val:+.1f})", 
            "odds": h_item["h1"], 
            "real_prob": sum(grid[i, j] for i in range(8) for j in range(8) if (i - j) > -h_val)
        })
        markets_to_test.append({
            "name": f"Фора 2 ({-h_val:+.1f})", 
            "odds": h_item["h2"], 
            "real_prob": sum(grid[i, j] for i in range(8) for j in range(8) if (j - i) > h_val)
        })

    best_pick = None
    max_margin_val = -999.0
    final_status = "РАБОЧИЙ ВАЛУЙ"

    for m in markets_to_test:
        odds = m["odds"]
        
        # ЗАЩИТНЫЙ ФИЛЬТР: Отсекаем дикие коэффициенты-ловушки выше 5.50
        if odds > 5.50:
            continue

        real_prob = m["real_prob"]
        real_prob_min = real_prob - 3.0
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
                    final_status = "ЖЕЛЕЗНЫЙ ВАЛУЙ"

    form_bonus = 0
    if data["form_tokens"]:
        wins_count = data["form_tokens"].count('В') + data["form_tokens"].count('W')
        losses_count = data["form_tokens"].count('П') + data["form_tokens"].count('L')
        form_bonus = (wins_count - losses_count) * 2

    if not best_pick:
        return f"{t1} — {t2} | МАТЧ РЕКОМЕНДОВАН К ПРОПУСКУ (SKIP) [Нет подтвержденного валуя по всему массиву рынков]"

    market_name, chosen_odds, val_score, r_prob, r_prob_min, imp_prob = best_pick
    confidence = min(95, max(60, int(75 + (max_margin_val if max_margin_val > 0 else val_score) * 2 + form_bonus)))
    stake_pct = 5 if confidence >= 80 else 3

    vector_lock = "Верховой" if data["k_tb25"] and data["k_tb25"] < 1.75 else "Стандартный"

    totals_str = ", ".join([f"Т({t['line']}): М={t['tu']}/Б={t['tb']}" for t in data["scanned_totals"]]) if data["scanned_totals"] else "Не найдены"
    hc_str = ", ".join([f"{h['line']:+.1f} (к1:{h['h1']}/к2:{h['h2']})" for h in data["scanned_handicaps"]]) if data["scanned_handicaps"] else "Не найдены"
    form_str = ", ".join(data["form_tokens"]) if data["form_tokens"] else "Не считывались"

    report = f"""<b>Шаг 0: Калибровка и привязка котировок</b>
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
- Статус каскада: Пройдено успешно

<b>Шаг 2 &amp; 3: Массив рынков и стресс-тест буфера</b>
- Проверен рынок: <b>{market_name}</b> (Коэф: {chosen_odds})
- Real Prob: {r_prob:.1f}% | Real Prob(min с учетом -3%): {r_prob_min:.1f}%
- Implied Prob БК: {imp_prob:.1f}%

<b>Шаг 4 &amp; 5: Уверенность и финансовый менеджмент</b>
- Индекс уверенности: {confidence}%
- Рекомендуемый банк: {stake_pct}% от общего банка
- Статус: [СТАТУС: {final_status}]

<b>Шаг 6: ИТОГОВЫЙ ПРИКАЗ ПО СТАВКЕ</b>"""

    if chosen_odds >= 2.10:
        final_line = f"\n{t1} — {t2} | {market_name} — {chosen_odds:.2f} — {stake_pct}% от банка (Уверенность: {confidence}%) | [СТАТУС: {final_status}]"
    else:
        final_line = f"\n{t1} — {t2} | Основная ставка по строгому регламенту отсутствует, зафиксирован подтвержденный математический валуй (План Б): {market_name} — {chosen_odds:.2f} — {stake_pct}% от банка (Уверенность: {final_status})"

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
