import math
import re
import cv2
import numpy as np
import easyocr
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command

# =====================================================================
# МАТЕМАТИЧЕСКОЕ ЯДРО И РЕГЛАМЕНТ ДЕТЕРМИНИРОВАННОГО АУДИТОРА v3.1
# =====================================================================

def poisson_pmf(k: int, lambd: float) -> float:
    return (lambd ** k) * math.exp(-lambd) / math.factorial(k)

def calculate_xg_from_total2_5(k_under: float, k_over: float, home_ratio: float) -> tuple[float, float]:
    impl_under = 1.0 / k_under
    impl_over = 1.0 / k_over
    margin_t = impl_under + impl_over
    norm_under = impl_under / margin_t
    
    best_lambda = 2.5
    min_diff = 999.0
    for l_val in np.linspace(0.5, 6.0, 12000):
        p_under = math.exp(-l_val) * (1.0 + l_val + (l_val**2)/2.0)
        diff = abs(p_under - norm_under)
        if diff < min_diff:
            min_diff = diff
            best_lambda = l_val
            
    xg_total = best_lambda
    xg1 = xg_total * home_ratio
    xg2 = xg_total * (1.0 - home_ratio)
    return xg1, xg2

def run_deterministic_audit(data: dict) -> str:
    match_name = data.get("match", "Хозяева — Гости")
    league = data.get("league", "Турнир (Верифицирован)")
    
    k_p1 = data["k_p1"]
    k_x = data["k_x"]
    k_p2 = data["k_p2"]
    k_tu25 = data["k_tu25"]
    k_tb25 = data["k_tb25"]
    
    # ШАГ 0: Калибровка и вектор
    impl_p1 = 1.0 / k_p1
    impl_x = 1.0 / k_x
    impl_p2 = 1.0 / k_p2
    margin_1x2 = impl_p1 + impl_x + impl_p2
    
    norm_p1 = impl_p1 / margin_1x2
    norm_p2 = impl_p2 / margin_1x2
    
    if k_tu25 < 1.75:
        vector = "Низовой"
    elif k_tb25 < 1.75:
        vector = "Верховой"
    else:
        vector = "Пограничный [1.76–2.05]"
        
    home_ratio = norm_p1 / (norm_p1 + norm_p2) if (norm_p1 + norm_p2) > 0 else 0.5
    xg1, xg2 = calculate_xg_from_total2_5(k_tu25, k_tb25, home_ratio)
    
    output = []
    output.append("📋 **ШАГ 0: Математическая калибровка линии**")
    output.append(f"• Матч: {match_name} | Турнир: {league}")
    output.append(f"• Связка 1X2: П1={k_p1:.2f} | Х={k_x:.2f} | П2={k_p2:.2f} (Маржа: {(margin_1x2-1)*100:.2f}%)")
    output.append(f"• Тотал 2.5: ТМ={k_tu25:.2f} | ТБ={k_tb25:.2f} ➔ Вектор: **{vector}**")
    output.append(f"• Детерминированный xG: xG1={xg1:.2f} | xG2={xg2:.2f} | Total={xg1+xg2:.2f}\n")
    
    # ШАГ 1: Матрица Пуассона и Каскадный Стоп-Кран
    output.append("📊 **ШАГ 1: Детерминированная матрица счетов (Пуассон)**")
    
    grid = np.zeros((6, 6))
    for h in range(6):
        for a in range(6):
            grid[h, a] = poisson_pmf(h, xg1) * poisson_pmf(a, xg2)
            
    scores = []
    for h in range(6):
        for a in range(6):
            scores.append(((h, a), grid[h, a]))
    scores.sort(key=lambda item: item[1], reverse=True)
    
    top1_p = scores[0][1]
    top2_p = top1_p + scores[1][1]
    top3_p = top2_p + scores[2][1]
    top4_p = top3_p + scores[3][1]
    tail_p = 1.0 - top4_p
    
    output.append(f"• ТОП-2 счетов: {top2_p*100:.2f}% (Лимит: <= 46.0%)")
    output.append(f"• ТОП-3 счетов: {top3_p*100:.2f}% (Лимит: <= 62.0%)")
    output.append(f"• ТОП-4 счетов: {top4_p*100:.2f}% (Лимит: <= 72.0%)")
    output.append(f"• Хвост неопределенности: {tail_p*100:.2f}% (Лимит: >= 28.0%)")
    
    cascade_fail = False
    if top2_p > 0.4601 or top3_p > 0.6201 or tail_p < 0.2799:
        cascade_fail = True
        
    if 0.690 <= top4_p <= 0.720:
        if (top4_p + 0.03) > 0.720:
            cascade_fail = True
            output.append("⚠️ Сработал превентивный буфер каскада (+3.0%).")
    elif top4_p > 0.7201:
        cascade_fail = True
        
    if cascade_fail:
        output.append("\n❌ **НАРУШЕН КАСКАДНЫЙ ЛИМИТ МАТРИЦЫ! АВАРИЙНЫЙ СТОП-КРАН СРАБОТАЛ.**")
        output.append(f"\n🚨 **ИТОГОВЫЙ ПРИКАЗ ПО СТАВКЕ:**\n`{match_name} | МАТЧ РЕКОМЕНДОВАН К ПРОПУСКУ (SKIP)`")
        return "\n".join(output)
        
    output.append("✅ Каскадный контроль матрицы пройден успешно.\n")
    
    # ШАГ 2 & 3: Массив рынков и буфер -3%
    output.append("🔍 **ШАГ 2 & 3: Сканирование рынков и Закон Честной Математики**")
    
    p_p1 = float(np.sum(np.tril(grid, -1)))
    p_draw = float(np.sum(np.diag(grid)))
    p_p2 = float(np.sum(np.triu(grid, 1)))
    p_tu25 = float(sum(grid[h, a] for h in range(6) for a in range(6) if h + a < 2.5))
    p_tb25 = float(sum(grid[h, a] for h in range(6) for a in range(6) if h + a > 2.5))
    
    markets = [
        {"name": "Home Win (1)", "k": k_p1, "prob": p_p1},
        {"name": "Draw (X)", "k": k_x, "prob": p_draw},
        {"name": "Away Win (2)", "k": k_p2, "prob": p_p2},
        {"name": "Under 2.5 Goals", "k": k_tu25, "prob": p_tu25},
        {"name": "Over 2.5 Goals", "k": k_tb25, "prob": p_tb25},
    ]

    best_market = None
    best_plan = None
    
    for m in markets:
        real_prob = m["prob"]
        real_prob_min = real_prob - 0.03  # Анти-галлюцинационный буфер -3%
        impl_prob = 1.0 / m["k"]
        value_margin_min = (real_prob_min - impl_prob) * 100
        
        output.append(f"• [{m['name']}] K={m['k']:.2f} | Real Prob: {real_prob*100:.2f}% | Prob(min -3%): {real_prob_min*100:.2f}% | VM(min): {value_margin_min:+.2f}%")
        
        if m["k"] >= 2.10:
            if value_margin_min >= 1.50:
                best_market = {**m, "vm_min": value_margin_min, "real_prob_min": real_prob_min, "impl_prob": impl_prob}
                best_plan = "A"
                break
        else:
            if real_prob_min > (impl_prob + 0.06):
                best_market = {**m, "vm_min": value_margin_min, "real_prob_min": real_prob_min, "impl_prob": impl_prob}
                best_plan = "B"
                break

    if not best_market:
        output.append("\n⚠️ Ни один рынок не прошел пороги валуя (План А / План Б).")
        output.append(f"\n🚨 **ИТОГОВЫЙ ПРИКАЗ ПО СТАВКЕ:**\n`{match_name} | МАТЧ РЕКОМЕНДОВАН К ПРОПУСКУ (SKIP)`")
        return "\n".join(output)

    # ШАГ 4, 5 & 6: Финансы и вердикт
    confidence = 85 if best_plan == "A" else 72
    bank_pct = 5 if confidence >= 80 else 3
    status_str = "ЖЕЛЕЗНЫЙ ВАЛУЙ" if (best_plan == "A" and confidence >= 80) else "РАБОЧИЙ ВАЛУЙ"
    
    output.append(f"\n🎯 **ШАГ 4 & 5: Финансовый менеджмент**")
    output.append(f"• Уверенность: {confidence}% ➔ Размер ставки: {bank_pct}% от банка")
    
    output.append(f"\n🚨 **ИТОГОВЫЙ ПРИКАЗ ПО СТАВКЕ:**")
    if best_plan == "A":
        final_cmd = f"{match_name} | {best_market['name']} — {best_market['k']:.2f} — {bank_pct}% от банка (Уверенность: {confidence}%) | [СТАТУС: {status_str}]"
    el
        final_cmd = f"{match_name} | Основная ставка отсутствует, зафиксирован валуй (План Б): {best_market['name']} — {best_market['k']:.2f} — {bank_pct}% от банка | [СТАТУС: РАБОЧИЙ ВАЛУЙ (План Б)]"
        
    output.append(f"`{final_cmd}`")
    return "\n".join(output)

# =====================================================================
# OCR И ТЕЛЕГРАМ-БОТ (aiogram 3.x)
# =====================================================================

reader = easyocr.Reader(['ru', 'en'], gpu=False)

def parse_screenshot(image_bytes: bytes) -> dict:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = reader.readtext(img, detail=0)
    full_text = " ".join(results)
    
    numbers = re.findall(r'\b\d+[\.,]\d+\b', full_text)
    numbers = [float(n.replace(',', '.')) for n in numbers]
    
    return {
        "match": "Матч со скриншота БК",
        "league": "Авто-определение",
        "k_p1": numbers[0] if len(numbers) > 0 else 1.70,
        "k_x": numbers[1] if len(numbers) > 1 else 3.95,
        "k_p2": numbers[2] if len(numbers) > 2 else 4.20,
        "k_tu25": numbers[3] if len(numbers) > 3 else 2.66,
        "k_tb25": numbers[4] if len(numbers) > 4 else 1.38,
    }

TOKEN = "9223011917:AAHB3RofroQou1MVLc5GoUHDFAm_wdEVxWo"
bot = Bot(token=TOKEN)
dp = Dispatcher()

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("🤖 **Детерминированный Аудитор v3.1 запущен.**\n\nПришли скриншот линии БК (1X2 и Тотал 2.5), и я проведу полный математический аудит!")

@dp.message(F.photo)
async def handle_screenshot(message: types.Message):
    await message.answer("🔄 Считываю скриншот нейросетью EasyOCR и считаю Пуассона...")
    photo = message.photo[-1]
    file_info = await bot.get_file(photo.file_id)
    file = await bot.download_file(file_info.file_path)
    
    data = parse_screenshot(file.read())
    result = run_deterministic_audit(data)
    await message.answer(result, parse_mode="Markdown")

if __name__ == "__main__":
    import asyncio
    asyncio.run(dp.start_polling(bot))