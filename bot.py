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
    
    # 1. Собираем все элементы с их точными координатами
    for bbox, text, prob in results:
        y_min = bbox[0][1]
        x_min = bbox[0][0]
        clean_txt = text.strip()
        elements.append({'y': y_min, 'x': x_min, 'text': clean_txt})
        
        # Название матча (ищем в верхней части с дефисом)
        if "-" in clean_txt and len(clean_txt) > 6 and y_min < 350:
            match_title = clean_txt
            
        # Форма команд
        if clean_txt.upper() in ['В', 'Н', 'П', 'W', 'D', 'L'] and len(clean_txt) == 1:
            form_tokens.append(clean_txt.upper())

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

    if '1x2' not in anchors:
        raise ValueError("На скриншоте не найден блок 'Исход 1X2'. Откройте основную роспись матча.")
        
    y_1x2 = anchors['1x2']
    y_total = anchors.get('total', y_1x2 + 250)
    y_hc = anchors.get('handicap', y_total + 250)
    y_combo = anchors.get('combo', y_hc + 250)
    
    # Универсальная функция извлечения чисел (коэффициентов) в заданном вертикальном диапазоне
    def extract_odds_in_zone(y_start, y_end):
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
        return [item[1] for item in zone_items]

    # Нижняя граница с запасом для сбора всего экрана
    max_y = max([e['y'] for e in elements]) + 100

    # 3. Извлекаем блоки по зонам
    odds_1x2 = extract_odds_in_zone(y_1x2, y_total)
    if len(odds_1x2) < 3:
        raise ValueError(f"Ошибка парсинга 1X2: найдено коэффициентов: {len(odds_1x2)}. Требуется ровно 3.")
    k_p1, k_x, k_p2 = odds_1x2[0], odds_1x2[1], odds_1x2[2]

    # Тоталы (берем ТМ 2.5 и ТБ 2.5 из блока тоталов)
    odds_totals = extract_odds_in_zone(y_total, y_hc)
    k_tu25 = odds_totals[2] if len(odds_totals) > 2 else (odds_totals[0] if len(odds_totals) > 0 else None)
    k_tb25 = odds_totals[5] if len(odds_totals) > 5 else (odds_totals[1] if len(odds_totals) > 1 else None)

    # Форы (захватываем все котировки из зоны фор)
    odds_handicaps = extract_odds_in_zone(y_hc, y_combo)

    # Комбо-ставки (исход и тотал)
    odds_combos = extract_odds_in_zone(y_combo, max_y)

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
    
    # Калиброванный расчет xG через имплицитную вероятность тотала 2.5
    if data["k_tu25"] and data["k_tb25"]:
        p_under = 1 / data["k_tu25"]
        xG_total = max(1.2, min(4.5, 3.2 - (p_under - 0.5) * 2.5))
    else:
        xG_total = 2.55  # Стандартный fallback при отсутствии тоталов в кадре
        
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

    # Каскадный стоп-кран
    cascade_passed = (top2_prob <= 46.0) and (top3_prob <= 62.0) and (top4_prob <= 72.0) and (tail_prob >= 28.0)
    if 69.0 <= top4_prob <= 72.0 and (top4_prob + 3.0) > 72.0:
        cascade_passed = False

    teams = data['match'].split('-') if '-' in data['match'] else ["Команда 1", "Команда 2"]
    t1, t2 = teams[0].strip(), teams[1].strip() if len(teams) > 1 else "Команда 2"

    if not cascade_passed:
        return f"{t1} — {t2} | МАТЧ РЕКОМЕНДОВАН К ПРОПУСКУ (SKIP) [Причина: Нарушен каскадный лимит матрицы Пуассона]"

    # Пул рынков формируется СТРОГО из реальных подтвержденных данных со скриншота
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
        markets_to_test.append({"name": "Фора (по котировке)", "odds": data["handicaps"][-1], "real_prob": sum(grid[i, j] for i in range(6) for j in range(6) if i != j)})

    if len(data["combos"]) >= 1:
        markets_to_test.append({"name": "Комбо-рынок (П1/ТМ или П1/ТБ)", "odds": data["combos"][0], "real_prob": sum(grid[i, j] for i in range(6) for j in range(6) if (i > j) and (i + j < 3.5))})

    best_pick = None
    max_margin_val = -999.0
    final_status = "РАБОЧИЙ ВАЛУЙ"

    # Аудит по Плану А / Плану Б с дисперсионным буфером -3%
    for m in markets_to_test:
        odds = m["odds"]
        real_prob = m["real_prob"]
        real_prob_min = real_prob - 3.0  # Буфер безопасности -3%
        implied_prob = (1 / odds) * 100
        
        if odds >= 2.10:
            vm_min = real_prob_min - implied_prob
            if vm_min >= 1.50 and vm_min > max_margin_val:
                max_margin_val = vm_min
                best_pick = (m["name"], odds, vm_min)
                final_status = "ЖЕЛЕЗНЫЙ ВАЛУЙ" if vm_min >= 6.0 else "РАБОЧИЙ ВАЛУЙ"
        else:
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
    print("Бот v3.1 успешно запущен и готов к работе...")
    bot.infinity_polling()
