import os
import re
import json
import warnings
import traceback
import telebot
from scipy.stats import poisson
import google.generativeai as genai

# Подавляем технические предупреждения
warnings.filterwarnings("ignore", category=UserWarning)

TOKEN = os.getenv('BOT_TOKEN')
bot = telebot.TeleBot(TOKEN)

GEMINI_KEY = os.getenv('GEMINI_KEY')

# Инициализация Gemini API с актуальной моделью gemini-3.8-flash
genai.configure(api_key=GEMINI_KEY)
gemini_model = genai.GenerativeModel('gemini-3.8-flash')

print("Gemini API успешно инициализирован с моделью gemini-3.8-flash.")

def parse_screenshot(image_bytes: bytes) -> dict:
    """
    Отправляет скриншот в Gemini с жестким таймаутом, чтобы исключить зависания.
    Если данные нечитаемы или отсутствуют, выбрасывает исключение.
    """
    prompt = """
Ты — профессиональный спортивный аналитик и строгий парсер букмекерских линий.
Проанализируй этот скриншот букмекерской конторы и верни **строго JSON** (без блоков кода типа ```json, только чистый JSON-текст).

Внимательно изучи изображение и заполни следующие поля:
1. "match": строка, точное название матча в формате "Команда 1 — Команда 2" (если названия не видны, укажи null).
2. "k_p1": число (float), коэффициент на победу первой команды (П1). Если его нет на скриншоте, строго укажи null.
3. "k_x": число (float), коэффициент на ничью (Х). Если его нет, строго укажи null.
4. "k_p2": число (float), коэффициент на победу второй команды (П2). Если его нет, строго укажи null.
5. "scanned_totals": список объектов тоталов, где каждый объект имеет вид:
   {"line": 2.5, "tu": 1.75, "tb": 2.10} (если линии тоталов видны, перечисли их все; если нет — пустой список []).
6. "scanned_handicaps": список объектов фор, где каждый объект имеет вид:
   {"h1_val": -1.0, "h2_val": 1.0, "h1": 2.05, "h2": 1.78} (если форы видны, перечисли их; если нет — пустой список []).

КРИТИЧЕСКОЕ ТРЕБОВАНИЕ: Не выдумывай и не подставляй коэффициенты «из головы», если они неразборчивы или отсутствуют на скриншоте. Если базовые коэффициенты (k_p1, k_x, k_p2) не удается достоверно считать, поля должны содержать null.
"""

    image_part = {
        "mime_type": "image/jpeg",
        "data": image_bytes
    }

    try:
        response = gemini_model.generate_content(
            [prompt, image_part],
            request_options={"timeout": 25.0}
        )
        
        if not response or not response.text:
            raise RuntimeError("Gemini вернул пустой ответ или сработал фильтр безопасности.")

        clean_text = response.text.strip()
        
        if clean_text.startswith("```"):
            clean_text = re.sub(r"^```(?:json)?\n", "", clean_text)
            clean_text = re.sub(r"\n```$", "", clean_text)
            clean_text = clean_text.strip()

        data = json.loads(clean_text)
        
        k1 = data.get("k_p1")
        kx = data.get("k_x")
        k2 = data.get("k_p2")
        match_title = data.get("match")

        if not match_title or k1 is None or kx is None or k2 is None:
            raise ValueError(
                "Не удалось достоверно считать обязательные данные с изображения "
                "(название матча или основные коэффициенты 1X2 отсутствуют/нечитаемы)."
            )

        data.setdefault("scanned_totals", [])
        data.setdefault("scanned_handicaps", [])

        return data

    except json.JSONDecodeError:
        raise RuntimeError("Ошибка парсинга ответа Gemini: модель вернула некорректный JSON формат.")
    except Exception as e:
        raise RuntimeError(f"Ошибка считывания скриншота: {e}")


def execute_v31_audit(data: dict, full_report: bool = False) -> str:
    k1, kx, k2 = data["k_p1"], data["k_x"], data["k_p2"]
    
    try:
        implied_p1 = 1 / k1
        implied_kx = 1 / kx
        implied_p2 = 1 / k2
        sum_1x2 = implied_p1 + implied_kx + implied_p2
        
        p1_norm = implied_p1 / sum_1x2
        p2_norm = implied_p2 / sum_1x2
        
        k_tu25, k_tb25 = None, None
        for t in data.get("scanned_totals", []):
            if t.get("line") == 2.5:
                k_tu25, k_tb25 = t.get("tu"), t.get("tb")
                break
        if not k_tu25 and data.get("scanned_totals"):
            k_tu25, k_tb25 = data["scanned_totals"][0].get("tu"), data["scanned_totals"][0].get("tb")

        if k_tu25 and k_tb25:
            impl_tb = 1 / k_tb25
            impl_tm = 1 / k_tu25
            xG_total = max(1.0, min(5.0, 2.5 * (impl_tb / (impl_tb + impl_tm) + 0.5)))
        else:
            xG_total = 2.50
            
        # Строгий детерминированный расчет xG по линии без искажений
        xG_1 = max(0.2, xG_total * (p1_norm / (p1_norm + p2_norm + 0.001)))
        xG_2 = max(0.2, xG_total - xG_1)

    except ZeroDivisionError:
        xG_total, xG_1, xG_2 = 2.50, 1.25, 1.25

    import numpy as np
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
        elif float(h1_val).is_integer():
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
        elif float(h2_val).is_integer():
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

    k_tu_check = k_tu25 if k_tu25 else 2.0
    k_tb_check = k_tb25 if k_tb25 else 2.0
    if k_tu_check < 1.75:
        vector_lock = "Низовой"
    elif k_tb_check < 1.75:
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
            final_status = "МАТЧ РЕКОМЕНДОВАН К ПРОПУСКУ (SKIP) [Нет подтвержденного валуя по всему массиву рынков]"

    if best_pick:
        market_name, chosen_odds, val_score, r_prob, r_prob_min, imp_prob = best_pick
        confidence = min(95, max(60, int(75 + (max_margin_val if max_margin_val > 0 else val_score) * 2)))
        stake_pct = 5 if confidence >= 80 else 3
    else:
        market_name, chosen_odds, r_prob, r_prob_min, imp_prob = "Отсутствует", 0.0, 0.0, 0.0, 0.0
        confidence = 0
        stake_pct = 0

    # Короткий дефолтный вывод
    if best_pick:
        if chosen_odds >= 2.10:
            short_line = f"🟢 <b>[СТАТУС: {final_status}]</b>\n\n{t1} — {t2} | {market_name} — {chosen_odds:.2f} — {stake_pct}% от банка (Уверенность: {confidence}%)"
        else:
            short_line = f"🟢 <b>[СТАТУС: {final_status}]</b>\n\n{t1} — {t2} | Альтернативная ставка по Плану Б: {market_name} — {chosen_odds:.2f} — {stake_pct}% от банка (Уверенность: {confidence}%)"
    else:
        short_line = f"🔴 <b>[СТАТУС: МАТЧ РЕКОМЕНДОВАН К ПРОПУСКУ (SKIP)]</b>\n\n{t1} — {t2} | Матч пропущен по регламенту математической безопасности."

    # Если запрошен full — возвращаем подробный отчет без слова "шаг"
    if full_report:
        totals_str = ", ".join([f"Т({t['line']}): М={t['tu']}/Б={t['tb']}" for t in data.get("scanned_totals", [])]) if data.get("scanned_totals") else "Не найдены"
        hc_str = ", ".join([f"Ф1({h['h1_val']:+.1f}):{h['h1']} / Ф2({h['h2_val']:+.1f}):{h['h2']}" for h in data.get("scanned_handicaps", [])]) if data.get("scanned_handicaps") else "Не найдены"

        full_text = f"""<b>Калибровка и привязка котировок</b>
- <b>Матч</b>: {t1} — {t2}
- <b>1X2</b>: П1={k1}, Х={kx}, П2={k2}
- <b>Тоталы</b>: [{totals_str}]
- <b>Форы</b>: [{hc_str}]
- xG (расчет по линии): {t1} ({xG_1:.2f}) / {t2} ({xG_2:.2f}) | Сумма: {xG_total:.2f}
- Вектор матча: {vector_lock}

<b>Матрица Пуассона и каскадный контроль</b>
- ТОП-2 счетов: {top2_prob:.1f}% (Лимит &le; 46%)
- ТОП-3 счетов: {top3_prob:.1f}% (Лимит &le; 62%)
- ТОП-4 счетов: {top4_prob:.1f}% (Лимит &le; 72%)
- Статус каскада: {"Пройдено успешно" if cascade_passed else "Нарушен лимит"}

<b>Массив рынков и стресс-тест буфера</b>
- Проверен рынок: <b>{market_name}</b> (Коэф: {chosen_odds})
- Real Prob: {r_prob:.1f}% | Real Prob(min): {r_prob_min:.1f}%
- Implied Prob БК: {imp_prob:.1f}%

<b>Уверенность и финансовый менеджмент</b>
- Индекс уверенности: {confidence}%
- Рекомендуемый банк: {stake_pct}% от общего банка

-----------------------------------
{short_line}"""
        return full_text

    return short_line


@bot.message_handler(commands=['start'])
def func_start(message):
    bot.reply_to(message, "Аудитор v3.1 (Чистый детерминизм) запущен. Отправьте скриншот линии (с текстом full для полного отчета).")


@bot.message_handler(content_types=['photo'])
def func_handle_photo(message):
    processing_msg = bot.reply_to(message, "🛡️ Анализируем скриншот через Gemini Flash...")
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        # Проверяем подпись к фотографии на наличие слова full
        caption = message.caption.strip().lower() if message.caption else ""
        show_full = "full" in caption or "полно" in caption

        parsed_data = parse_screenshot(downloaded_file)
        result_verdict = execute_v31_audit(parsed_data, full_report=show_full)
        
        bot.edit_message_text(result_verdict, chat_id=message.chat.id, message_id=processing_msg.message_id, parse_mode='HTML')
    except Exception as e:
        tb_str = traceback.format_exc()
        safe_tb = tb_str.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        if len(safe_tb) > 3500:
            safe_tb = safe_tb[:3500] + "\n...[обрезано]"
            
        error_message = f"❌ <b>Ошибка аудита:</b>\n<pre>{safe_tb}</pre>"
        bot.edit_message_text(error_message, chat_id=message.chat.id, message_id=processing_msg.message_id, parse_mode='HTML')


if __name__ == "__main__":
    print("Бот v3.1 (Детерминизм + Режим Full) успешно запущен...")
    bot.infinity_polling()
