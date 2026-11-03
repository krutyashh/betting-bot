import os
import re
import json
import warnings
import traceback
import cv2
import numpy as np
import telebot
from PIL import Image
import io
from google import genai
from google.genai import types
from scipy.stats import poisson

warnings.filterwarnings("ignore", category=UserWarning)

TOKEN = "9223011917:AAF2aonZ5kaX9TkD7fx0kz6U2gFzc52Cx_Y"
bot = telebot.TeleBot(TOKEN)

print("Инициализация Gemini API...")
GEMINI_KEY = "AQ.Ab8RN6Jn-Vik_Xw_u9RQIUOdihZQrSWZDR6YcWasuszCz2_43A"
ai_client = genai.Client(api_key=GEMINI_KEY)
print("Движок Gemini Vision успешно запущен вместо EasyOCR.")

def parse_screenshot(image_bytes: bytes) -> dict:
    image = Image.open(io.BytesIO(image_bytes))

    prompt = """
    Ты — профессиональный парсер спортивных линий букмекерских контор. Твоя задача — идеально считать данные со скриншота для математической модели.
    Внимательно проанализируй изображение и извлеки параметры. Верни ответ СТРОГО в формате JSON без каких-либо ```json ... ``` или пояснений. Только чистый JSON.

    Структура JSON:
    {
      "match": "Название Команды 1 — Название Команды 2",
      "k_p1": 2.10, "k_x": 3.40, "k_p2": 3.20, 
      "scanned_totals": [
         {"line": 2.5, "tu": 1.95, "tb": 1.85}
      ],
      "scanned_handicaps": [
         {"h1_val": 0.0, "h2_val": 0.0, "h1": 1.55, "h2": 2.40}
      ],
      "t1_form": ["В", "Н", "П"], 
      "t2_form": ["В", "В", "В"]
    }
    Правила извлечения:
    1. Форма команд: найди кружочки последних матчей (W/D/L или В/Н/П). Раздели их для t1_form и t2_form. Символы строго: 'В', 'Н', 'П'. Если формы нет, верни [].
    2. Форы: строго соблюдай знаки плюс/минус. Если фора (0), запиши как 0.0.
    """

    try:
        response = ai_client.models.generate_content(
            model='gemini-1.5-flash',
            contents=[image, prompt]
        )
        
        raw_text = response.text.strip()
        if raw_text.startswith("```"):
            raw_text = re.sub(r'^```(?:json)?\n|```$', '', raw_text, flags=re.MULTILINE).strip()

        data = json.loads(raw_text)

        k_tu25 = None
        k_tb25 = None
        for t in data.get("scanned_totals", []):
            if t.get("line") == 2.5:
                k_tu25 = t.get("tu")
                k_tb25 = t.get("tb")
                break
        
        if not k_tu25 and data.get("scanned_totals"):
            k_tu25 = data["scanned_totals"].get("tu")
            k_tb25 = data["scanned_totals"].get("tb")

        form_tokens = data.get("t1_form", []) + data.get("t2_form", [])

        t1_f_str = ", ".join(data.get("t1_form", [])) if data.get("t1_form") else "нет данных"
        t2_f_str = ", ".join(data.get("t2_form", [])) if data.get("t2_form") else "нет данных"
        
        form_log_str = (
            f"<b>[GEMINI AI VISION LOG] Матч: {data.get('match', 'Команда 1 — Команда 2')}</b>\n"
            f"Форма Команды 1: [{t1_f_str}] | Форма Команды 2: [{t2_f_str}]"
        )

        return {
            "match": data.get("match", "Команда 1 — Команда 2"),
            "k_p1": float(data.get("k_p1", 1.0)),
            "k_x": float(data.get("k_x", 1.0)),
            "k_p2": float(data.get("k_p2", 1.0)),
            "k_tu25": k_tu25,
            "k_tb25": k_tb25,
            "scanned_totals": data.get("scanned_totals", []),
            "scanned_handicaps": data.get("scanned_handicaps", []),
            "form_tokens": form_tokens,
            "form_log": form_log_str
        }

    except Exception as ex:
        raise ValueError(f"Не удалось распарсить скриншот через Gemini ИИ. Ошибка: {str(ex)}")
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
