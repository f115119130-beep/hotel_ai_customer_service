"""Tool-choice test for the hotel agent, adapted from Unit G-3 Step 9.
Run from the same folder as app.py and the hotel's CSV files.
"""
import time
from app import run_agent

TESTS = [
    ("請問幾點可以入住？", "get_hotel_info"),
    ("飯店有提供 Wi-Fi 嗎？", "get_hotel_info"),
    ("可以帶寵物入住嗎？", "get_hotel_info"),
    ("標準雙人房多少錢？", "get_room_info"),
    ("有家庭房嗎？", "get_room_info"),
    ("有提供機場接送服務嗎？", "get_room_info"),
    ("標準雙人房住三晚，總共多少錢？", "calculate_stay_total"),
    ("兩間家庭房住兩晚要多少錢？", "calculate_stay_total"),
    ("我要取消訂房並退款。", "transfer_to_human"),
    ("我想找真人客服。", "transfer_to_human"),
    ("請問明天還有空房嗎？", "transfer_to_human"),
    ("可以幫我寫期中報告嗎？", None),
]

correct = 0
for i, (question, expected) in enumerate(TESTS, 1):
    try:
        _, trace = run_agent([{"role": "user", "content": question}])
        used = [entry["tool"] for entry in trace]
        hit = (not used) if expected is None else expected in used
        correct += int(hit)
        print(f"{'OK  ' if hit else 'MISS'} Q{i:02d} {question}")
        print(f"       expected: {expected or '(no tool)'} | used: {used or '(no tool)'}", flush=True)
    except Exception as exc:
        print(f"ERROR Q{i:02d} {question} | {type(exc).__name__}: {exc}", flush=True)
    time.sleep(1)
print(f"\ntool choice = {correct}/{len(TESTS)} = {correct / len(TESTS):.1%}")
