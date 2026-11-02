import os
import re
import cv2
import numpy as np
import telebot
import easyocr
from scipy.stats import poisson

TOKEN = "9223011917:AAHB3RofroQou1MVLc5GoUHDFAm_wdEVxWo"
bot = telebot.TeleBot(TOKEN)

print("Инициализация EasyOCR движка...")
reader = easyocr.Reader(['ru', 'en'], gpu=False)
print("Движок OCR успешно запущен.")

def parse_screenshot(image_bytes: bytes) -> dict:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    results = reader.readtext(img, detail=1)
    
    anchors = {}
    match_title = "Не распознано"
    form_tokens = []
    
    for bbox, text, prob in results:
        clean_txt = text.lower().strip()
        if "1x2" in clean_txt or "исход" in clean_txt:
            anchors['1x2'] = bbox[0][1]
        elif "тотал" in clean_txt:
            anchors['total'] = bbox[0][1]
        elif "фора" in clean_txt:
            anchors['handicap'] = bbox[0][1]
        elif "комбинирован" in clean_txt or "исход и тотал" in clean_txt or "победа и" in clean_txt:
            anchors['combo'] = bbox[0][1]
            
        if "-" in text and len(text) > 6 and bbox[0][1] < anchors.get('1x2', 500):
            match_title = text.strip()
            
        if text.strip().upper() in ['В', 'Н', 'П', 'W', 'D', 'L'] and len(text.strip()) == 1:
            form_tokens.append(text.strip().upper())

    if '1x2' not in anchors:
        raise ValueError("На скриншоте не найден блок 'Исход 1X2'. Убедитесь, что открыт нужный рынок матча.")
        
    y_1x2 = anchors['1x2']
    y_total = anchors.get('total', y_1x2 + 200)
    y_hc = anchors.get('handicap', y_total + 200)
    y_combo = anchors.get('combo', y_hc + 200)
    
    def extract_odds_in_zone(y_start, y_end):
        collected = []
        for bbox, text, prob in results:
            y_coord = bbox[0][1]
            if y_start <= y_coord < y_end:
                clean_text = text.replace(',', '.')
                found_nums = re.findall(r'\b\d{1,3}\.\d{2}\b', clean_text)
                for num_str in found_nums:
                    try:
                        val = float(num_str)
                        if val > 1.0:
                            collected.append(val)
                    except ValueError:
                        continue
        return collected

    odds_1x2 = extract_odds_in_zone(y_1x2, min(y_total, y_1x2 + 160))
    if len(odds_1x2) < 3:
        raise ValueError(f"Ошибка парсинга 1X2: найдено коэффициентов: {len(odds_1x2)}. Требуется ровно 3.")

    k_p1, k_x, k_p2 = odds_1x2[0], odds_1x2[1], odds_1x2[2]
    
    odds_totals = extract_odds_in_zone(y_total, y_total + 200) if 'total' in anchors else []
    k_tu25 = odds_totals[2] if len(odds_totals) > 2 else None
    k_tb25 = odds_totals[5] if len(odds_totals) > 5 else None

    odds_handicaps = extract_odds_in_zone(y_hc, y_hc + 180) if 'handicap' in anchors else []
    odds_combos = extract_odds_in_zone(y_combo, y_combo + 250) if 'combo' in anchors else []

    return {
        "match": match_title,
        "k_p1": k_p1, "k_x": k_x, "k_p2": k_p2,
        "k_tu25": k_tu25, "k_tb25": k_tb25,
        "handicaps": odds_handicaps,
        "combos": odds_combos,
        "form_tokens": form_tokens
    }


def execute_v31_audit(data: dict) -> str:
    k1, kx, k2 = data["k_p1"], data["k_x"], data["k_p2"]
    
    # Точный расчет xG через имплицитную вероятность тотала 2.5
    if data["k_tu25"] and data["k_tb25"]:
        p_under = 1 / data["k_tu25"]
        xG_total = max(1.2, min(4.5, 3.2 - (p_under - 0.5) * 2.5))
    else:
        xG_total = 2.55 # Фоллбек, если тоталы не захвачены в кадр
        
    power_diff = (k2 - k1) / (k1 + k2 + kx)
    xG_1 = max(0.3, (xG_total / 2) * (1 + power_diff * 0.8))
    xG_2 = max(0.3, xG_total - xG_1)

    # Генерация матрицы Пуассона 6x6
    grid = np.zeros((6, 6))
    score_probs = []
    for i in range(6):
        for j in range(6):
            prob = poisson.pmf(i, xG_1) * poisson.pmf(j, xG_2) * 100
            grid[i, j] = prob
            score_probs.append(((i, j), prob))
            
    score_probs.sort(key=lambda x: x[1], reverse=True)
    
    top2_prob = score_probs[0][1] + score_probs[1][1]
    top3_prob = top2_prob + score_probs[2][1]
    top4_prob = top3_prob + score_probs[3][1]
    tail_prob = 100.0 - top4_prob

    # Каскадный лимит матрицы
    cascade_passed = (top2_prob <= 46.0) and (top3_prob <= 62.0) and (top4_prob <= 72.0) and (tail_prob >= 28.0)
    if 69.0 <= top4_prob <= 72.0 and (top4_prob + 3.0) > 72.0:
        cascade_passed = False

    teams = data['match'].split('-') if '-' in data['match'] else ["Команда 1", "Команда 2"]
    t1, t2 = teams[0].strip(), teams[1].strip() if len(teams) > 1 else "Команда 2"

    if not cascade_passed:
        return f"{t1} — {t2} | МАТЧ РЕКОМЕНДОВАН К ПРОПУСКУ (SKIP) [Причина: Нарушен каскадный лимит матрицы Пуассона]"

    # Формируем пул рынков СТРОГО из реальных данных со скриншота (без выдуманных заглушек)
    markets_to_test = [
        {"name": "Победа 1 (П1)", "odds": k1, "real_prob": sum(grid[i, j] for i in range(6) for j in range(6) if i > j)},
        {"name": "Ничья (Х)", "odds": kx, "real_prob": sum(grid[i, j] for i in range(6) for j in range(6) if i == j)},
        {"name": "Победа 2 (П2)", "odds": k2, "real_prob": sum(grid[i, j] for i in range(6) for j in range(6) if i < j)},
    ]

    if data["k_tb25"]:
        markets_to_test.append({"name": "Тотал больше (2.5)", "odds": data["k_tb25"], "real_prob": sum(grid[i, j] for i in range(6) for j in range(6) if (i + j) > 2.5)})
    if data["k_tu25"]:
        markets_to_test.append({"name": "Тотал меньше (2.5)", "odds": data["k_tu25"], "real_prob": sum(grid[i, j] for i in range(6) for j in range(6) if (i + j) < 2.5)})

    if len(data["handicaps"]) >= 2:
        markets_to_test.append({"name": "Фора 2 (-1.0)", "odds": data["handicaps"][1], "real_prob": sum(grid[i, j] for i in range(6) for j in range(6) if (j - i) >= 2)})

    if len(data["combos"]) >= 1:
        markets_to_test.append({"name": "П2 + ТБ(2.5) [Комбо]", "odds": data["combos"][0], "real_prob": sum(grid[i, j] for i in range(6) for j in range(6) if (j > i) and (i + j > 2.5))})

    best_pick = None
    max_margin_val = -999.0
    final_status = "РАБОЧИЙ ВАЛУЙ"

    # Аудит по Плану А / Плану Б с буфером -3%
    for m in markets_to_test:
        odds = m["odds"]
        real_prob = m["real_prob"]
        real_prob_min = real_prob - 3.0  # Дисперсионный буфер безопасности -3%
        implied_prob = (1 / odds) * 100
        
        if odds >= 2.10:
            # План А: Value Margin(min) >= +1.50%
            vm_min = real_prob_min - implied_prob
            if vm_min >= 1.50 and vm_min > max_margin_val:
                max_margin_val = vm_min
                best_pick = (m["name"], odds, vm_min)
                final_status = "ЖЕЛЕЗНЫЙ ВАЛУЙ" if vm_min >= 6.0 else "РАБОЧИЙ ВАЛУЙ"
        else:
            # План Б: Real Prob(min) > Implied Prob БК + 6.0% (Всегда Железный валуй)
            if real_prob_min > (implied_prob + 6.0):
                excess = real_prob_min - (implied_prob + 6.0)
                if excess > max_margin_val:
                    max_margin_val = excess
                    best_pick = (m["name"], odds, real_prob_min)
                    final_status = "ЖЕЛЕЗНЫЙ ВАЛУЙ"

    form_bonus = 0
    if data["form_tokens"]:
        wins_count = data["form_tokens"].count('В') + data["form_tokens"].count('W')
        losses_count = data["form_tokens"].count('П') + data["form_tokens"].count('L')
        form_bonus = (wins_count - losses_count) * 2

    if not best_pick:
        return f"{t1} — {t2} | МАТЧ РЕКОМЕНДОВАН К ПРОПУСКУ (SKIP) [Нет подтвержденного валуя по видимым рынкам]"

    market_name, chosen_odds, val_score = best_pick
    confidence = min(95, max(60, int(75 + val_score * 2 + form_bonus)))
    stake_pct = 5 if confidence >= 80 else 3

    if chosen_odds >= 2.10:
        return f"{t1} — {t2} | {market_name} — {chosen_odds:.2f} — {stake_pct}% от банка (Уверенность: {confidence}%) | [СТАТУС: {final_status}]"
    else:
        return f"{t1} — {t2} | Основная ставка по строгому регламенту отсутствует, зафиксирован подтвержденный математический валуй (План Б): {market_name} — {chosen_odds:.2f} — {stake_pct}% от банка (Уверенность: {final_status})"


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
        
        bot.edit_message_text(result_verdict, chat_id=message.chat.id, message_id=processing_msg.message_id)
    except Exception as e:
        bot.edit_message_text(f"❌ Ошибка аудита: {str(e)}", chat_id=message.chat.id, message_id=processing_msg.message_id)


if __name__ == "__main__":
    print("Бот v3.1 запущен и работает по честным данным...")
    bot.infinity_polling()
