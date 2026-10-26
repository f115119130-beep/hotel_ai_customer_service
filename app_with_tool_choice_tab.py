"""
AI 飯店客服：Gradio + SQLite + tool calling
這是課程作業用的虛構示範飯店「晴旅小屋」。價格與規定均為示範資料，
不是即時房況或真實飯店承諾。
Run: python app.py
"""
import csv
import inspect
import json
import os
import re
import sqlite3
import threading
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

CHAT_MODEL = os.getenv("MODEL_NAME", "").strip()
API_KEY = os.getenv("API_KEY", "").strip()
LLM_API_URL = os.getenv("LLM_API_URL", "").strip()
if not CHAT_MODEL or not API_KEY or not LLM_API_URL:
    raise RuntimeError(
        "尚未設定 API。請將 .env.example 複製為 .env，並填入 MODEL_NAME、API_KEY、LLM_API_URL。"
    )
client = OpenAI(api_key=API_KEY, base_url=LLM_API_URL, timeout=60.0)

# ---------------------------------------------------------------------
# 1. 從 CSV 建立 SQLite 資料表（作業要求：繳交 CSV，不繳交 .db）
# ---------------------------------------------------------------------
DB_PATH = BASE_DIR / "hotel_app.db"
connection = sqlite3.connect(DB_PATH, check_same_thread=False)
connection.row_factory = sqlite3.Row
db_lock = threading.Lock()

with db_lock:
    cur = connection.cursor()
    cur.execute("DROP TABLE IF EXISTS hotel_info")
    cur.execute("""CREATE TABLE hotel_info (
        topic TEXT PRIMARY KEY, content TEXT NOT NULL
    )""")
    cur.execute("DROP TABLE IF EXISTS hotel_items")
    cur.execute("""CREATE TABLE hotel_items (
        item_id INTEGER PRIMARY KEY, name TEXT NOT NULL, name_zh TEXT NOT NULL,
        category TEXT NOT NULL, price INTEGER NOT NULL, available INTEGER NOT NULL,
        description TEXT NOT NULL
    )""")

    with open(BASE_DIR / "hotel_info.csv", newline="", encoding="utf-8-sig") as f:
        cur.executemany(
            "INSERT INTO hotel_info(topic, content) VALUES (?, ?)",
            [(r["topic"], r["content"]) for r in csv.DictReader(f)]
        )
    with open(BASE_DIR / "hotel_items.csv", newline="", encoding="utf-8-sig") as f:
        cur.executemany(
            """INSERT INTO hotel_items
            (item_id, name, name_zh, category, price, available, description)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [(r["item_id"], r["name"], r["name_zh"], r["category"],
              int(r["price"]), int(r["available"]), r["description"])
             for r in csv.DictReader(f)]
        )
    connection.commit()
    HOTEL_TOPICS = [r["topic"] for r in cur.execute("SELECT topic FROM hotel_info ORDER BY topic")]
    item_count = cur.execute("SELECT COUNT(*) FROM hotel_items").fetchone()[0]

print(f"資料庫建立完成：資訊 {len(HOTEL_TOPICS)} 個主題，房型／服務 {item_count} 筆。")

# ---------------------------------------------------------------------
# 2. 工具函式：只回傳資料，不直接生成客服句子
# ---------------------------------------------------------------------
def get_hotel_info(topic):
    """查詢飯店規定或設施等資訊。topic 必須是資料庫已有的主題代碼。"""
    with db_lock:
        row = connection.execute(
            "SELECT content FROM hotel_info WHERE topic = ?", (topic,)
        ).fetchone()
    if row is None:
        return {"found": False, "available_topics": HOTEL_TOPICS}
    return {"found": True, "topic": topic, "content": row["content"]}


def _escape_like(text):
    """跳脫 LIKE 的萬用字元 % 與 _，避免使用者輸入被當成模式。"""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def get_room_info(item_name):
    """查詢房型／方案／服務的價格、供應狀態和說明。優先完全相符，其次模糊比對。"""
    term = str(item_name or "").strip()
    if not term:
        return {"found": False, "note": "未提供查詢名稱。"}
    columns = "name, name_zh, category, price, available, description"
    with db_lock:
        # 先找完全相符，避免「標準雙人房」同時匹配到「標準雙人房含早餐」
        rows = connection.execute(
            f"""SELECT {columns} FROM hotel_items
                WHERE LOWER(name) = LOWER(?) OR name_zh = ?
                ORDER BY item_id""", (term, term)
        ).fetchall()
        if not rows:
            like = f"%{_escape_like(term)}%"
            rows = connection.execute(
                f"""SELECT {columns} FROM hotel_items
                    WHERE name LIKE ? ESCAPE '\\' OR name_zh LIKE ? ESCAPE '\\'
                    ORDER BY item_id""", (like, like)
            ).fetchall()
    if not rows:
        return {"found": False, "note": "資料庫中沒有符合的房型或服務。"}
    return {"found": True, "items": [
        {"name": r["name"], "name_zh": r["name_zh"], "category": r["category"],
         "price_per_night_or_fee": r["price"], "available": bool(r["available"]),
         "description": r["description"]} for r in rows
    ]}


def calculate_stay_total(room_name, nights, rooms):
    """由程式計算住宿總額；價格依資料庫，房數與晚數由客人問題解析而來。"""
    if isinstance(nights, bool) or not isinstance(nights, int) or not 1 <= nights <= 365:
        return {"ok": False, "note": "住宿晚數必須是 1 到 365 的整數。"}
    if isinstance(rooms, bool) or not isinstance(rooms, int) or not 1 <= rooms <= 50:
        return {"ok": False, "note": "房間數必須是 1 到 50 的整數。"}
    found = get_room_info(room_name)
    if not found["found"] or len(found["items"]) != 1:
        return {"ok": False, "note": "無法唯一確認房型，請先查詢房型或請客人補充說明。"}
    item = found["items"][0]
    if item["category"] not in ("room", "package"):
        return {"ok": False, "note": "此項目不是住宿房型，不能用住宿晚數計算。"}
    if not item["available"]:
        return {"ok": False, "note": "資料目前標示此房型／方案未供應，無法計算可訂住宿費。"}
    unit = item["price_per_night_or_fee"]
    return {
        "ok": True, "room_name": item["name"], "room_name_zh": item["name_zh"],
        "price_per_room_per_night": unit, "nights": nights, "rooms": rooms,
        "total": unit * nights * rooms,
        "note": "金額依示範資料計算；不代表指定日期的即時房況、稅費或最終訂房報價。"
    }


def transfer_to_human():
    """無法回答、資料未確認、客訴或要求真人服務時，提供人工協助指引。"""
    return {
        "message": "請透過飯店櫃檯或您原本使用的訂房平台聯絡真人服務人員。",
        "note": "本示範專案沒有設定真實電話、地址或真人轉接功能。"
    }


TOOL_FUNCTIONS = {
    "get_hotel_info": get_hotel_info,
    "get_room_info": get_room_info,
    "calculate_stay_total": calculate_stay_total,
    "transfer_to_human": transfer_to_human,
}

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "get_hotel_info",
        "description": "查詢入住退房時間、地址、停車、付款、早餐、取消、加床、寵物、Wi-Fi 等飯店資訊。必須使用資料庫中的 topic 代碼。",
        "parameters": {"type": "object", "properties": {
            "topic": {"type": "string", "enum": HOTEL_TOPICS}},
            "required": ["topic"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "get_room_info",
        "description": "優先查詢 hotel_items 資料表中的房型、住宿方案及服務項目（例如機場接送）的價格、供應狀態與說明；即使服務未供應也要先查資料。使用中文或英文項目名稱的一小段。",
        "parameters": {"type": "object", "properties": {
            "item_name": {"type": "string"}},
            "required": ["item_name"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "calculate_stay_total",
        "description": "計算住宿費總額時一定要呼叫此工具，不可自行心算。只接受資料庫中唯一的房型／住宿方案名稱，以及住宿晚數與房間數。這不是即時訂房確認。",
        "parameters": {"type": "object", "properties": {
            "room_name": {"type": "string"},
            "nights": {"type": "integer", "minimum": 1, "maximum": 365},
            "rooms": {"type": "integer", "minimum": 1, "maximum": 50}},
            "required": ["room_name", "nights", "rooms"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "transfer_to_human",
        "description": "用於客訴、要求真人／經理、資料庫沒有的飯店問題，或需要確認真實房況、訂房、取消與退款等事項。",
        "parameters": {"type": "object", "properties": {},
            "required": [], "additionalProperties": False}}},
]

SYSTEM_PROMPT = """
你是「晴旅小屋」的 AI 飯店客服。這是一個課程作業用的虛構示範飯店。
所有金額均為新臺幣 NT$，但房價與規定只是示範資料，不是即時訂房報價。

規則：
1. 只能根據工具回傳的結果回答，不可依模型記憶作答，不可編造。
2. 查不到資料時要明確說明，並呼叫 transfer_to_human；不要猜測。
3. 只回答這家飯店的住宿、房型、設施與服務問題。天氣、作業、寫程式等非客服問題，禮貌拒絕並引導回飯店服務；不需要呼叫飯店資料工具。
4. 回答要有禮貌，原則上三句以內。依客人使用的語言回答；中文請使用繁體中文和房型中文名稱。
5. 飯店規定、設施、付款等問題 -> get_hotel_info。
6. 詢問房型、住宿方案，以及 hotel_items 資料表中的服務項目（例如機場接送）的價格、說明或供應狀態時，必須先呼叫 get_room_info 查詢。即使項目目前未供應，也應先查詢資料，不可直接猜測或跳過查詢；若資料查不到，或需要確認最新供應狀況，再呼叫 transfer_to_human。
7. 任何住宿總額都必須呼叫 calculate_stay_total；不得自己計算。房數或晚數不清楚時，先向客人詢問。
8. 客訴、要求真人、資料缺漏、即時房況／訂房確認、取消退款等需要人工核實的問題 -> transfer_to_human。
9. available 是示範資料中的供應標記，不代表指定日期的即時空房。
10. 不要聲稱已完成訂房、取消、退款或已真正轉接真人。
"""

MAX_STEPS = 6


def content_to_text(content):
    """將 Gradio 對話內容轉成 API 可使用的文字。"""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)

    if isinstance(content, dict):
        text = content.get("text")
        return text if isinstance(text, str) else ""

    return str(content or "")


def clean_history(chat_history):
    """只保留 role/content，並把 content 轉成純文字，避免 metadata 等欄位被送進 API。"""
    return [
        {"role": m["role"], "content": content_to_text(m.get("content"))}
        for m in chat_history
        if m.get("role") in ("user", "assistant")
    ]


def run_agent(chat_history):
    """處理最後一個使用者問題，回傳答案與工具軌跡。"""
    history = clean_history(chat_history)
    question = history[-1]["content"]
    language = "Traditional Chinese" if re.search(r"[\u4e00-\u9fff]", question) else "English"
    system = SYSTEM_PROMPT + f"\n本次請以 {language} 回答。"
    messages = [{"role": "system", "content": system}] + history

    trace = []
    for step in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, tools=TOOL_SPECS,
            temperature=0.2
        )
        message = response.choices[0].message
        if not message.tool_calls:
            return (message.content or "抱歉，我目前無法產生回答。").strip(), trace

        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [{
                "id": c.id, "type": "function",
                "function": {"name": c.function.name, "arguments": c.function.arguments}
            } for c in message.tool_calls]
        })

        for call in message.tool_calls:
            tool_name = call.function.name
            arguments = {}
            try:
                # 安全檢查：只執行明確登記的工具，拒絕模型指定的其他名稱。
                if tool_name not in TOOL_FUNCTIONS:
                    result = {"error": "拒絕執行未登記的工具。"}
                else:
                    arguments = json.loads(call.function.arguments or "{}")
                    if not isinstance(arguments, dict):
                        raise ValueError("工具參數必須是 JSON 物件。")
                    result = TOOL_FUNCTIONS[tool_name](**arguments)
            except (json.JSONDecodeError, TypeError, ValueError, KeyError) as error:
                arguments = {}
                result = {"error": "工具參數無效。", "detail": str(error)}
            except Exception as error:  # 例如資料庫錯誤，避免整個對話中斷
                arguments = {}
                result = {"error": "工具執行失敗。", "detail": type(error).__name__}
                print(f"Tool error ({tool_name}): {type(error).__name__}: {error}")
            trace.append({
                "step": step + 1, "tool": tool_name,
                "arguments": arguments,
                "result": result
            })
            messages.append({
                "role": "tool", "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False)
            })
    return "抱歉，我暫時無法完成這個問題。請透過飯店櫃檯或訂房平台詢問真人服務人員。", trace


# ---------------------------------------------------------------------
# 3. Gradio 介面
# ---------------------------------------------------------------------
GREETING = (
    "您好！我是晴旅小屋的 AI 飯店客服（課程示範版）。"
    "您可以詢問入住規定、房型價格，或住宿費用試算。"
    "房價與供應狀態為示範資料，並非即時訂房資訊。"
)
EMPTY_TRACE = "### 工具使用紀錄\n\n（尚未呼叫工具）"
SAMPLE_GROUPS = {
    "飯店資訊": [
        "請問幾點可以入住？",
        "有提供 Wi-Fi 嗎？",
        "可以帶寵物入住嗎？",
    ],
    "房型與價格": [
        "標準雙人房多少錢？",
        "家庭房有包含早餐的方案嗎？",
        "機場接送目前有供應嗎？",
    ],
    "住宿費用試算": [
        "標準雙人房住三晚，總共多少錢？",
        "兩間家庭房住兩晚要多少錢？",
    ],
    "需要真人或非客服問題": [
        "我要取消訂房並要求退款。",
        "請問明天晚上還有套房嗎？",
        "可以幫我寫期中報告嗎？",
    ],
}


def format_trace(trace):
    if not trace:
        return "### 工具使用紀錄\n\n（這次回答沒有呼叫工具）"
    lines = ["### 工具使用紀錄", ""]
    for entry in trace:
        lines += [
            f"**{entry['step']}. {entry['tool']}**", "",
            f"參數：`{json.dumps(entry.get('arguments', {}), ensure_ascii=False)[:220]}`", "",
            f"結果：`{json.dumps(entry['result'], ensure_ascii=False)[:420]}`", ""
        ]
    return "\n".join(lines)


def start_chat():
    return [{"role": "assistant", "content": GREETING}]


def add_user_message(user_text, chat_history):
    user_text = (user_text or "").strip()
    if not user_text:
        return "", chat_history
    return "", (chat_history or []) + [{"role": "user", "content": user_text}]


def answer_question(chat_history):
    if not chat_history or chat_history[-1]["role"] != "user":
        return chat_history, EMPTY_TRACE
    try:
        answer, trace = run_agent(chat_history)
        panel = format_trace(trace)
    except Exception as error:
        # 不把 API 金鑰或環境設定值寫入畫面。
        answer = "系統目前無法連線至 AI 服務，請檢查 API 設定與網路後再試。"
        panel = "### 工具使用紀錄\n\n（呼叫失敗；請查看終端機錯誤訊息。）"
        print(f"API/agent error: {type(error).__name__}: {error}")
    return chat_history + [{"role": "assistant", "content": answer}], panel


LAKE = gr.themes.Color(
    c50="#eef7f9", c100="#d6ebf0", c200="#b0d7e0", c300="#7fbccb", c400="#4a9bb0",
    c500="#2c7f96", c600="#226779", c700="#1f5463", c800="#1b4653", c900="#183a45",
    c950="#0f252d", name="lake",
)

THEME = gr.themes.Soft(
    primary_hue=LAKE,
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_md,
    font=["Noto Sans TC", "Microsoft JhengHei", "PingFang TC", "system-ui", "sans-serif"],
).set(
    body_background_fill="#EAF4F8",
    body_background_fill_dark="#10212A",
    button_primary_background_fill="#226779",
    button_primary_background_fill_hover="#1B4B5A",
    button_primary_text_color="white",
)

CSS = """
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }

#hero { position: relative; overflow: hidden; padding: 24px 30px;
        border-radius: 16px; background: #1B4B5A; }
#hero::after { content: ""; position: absolute; right: -36px; top: -48px;
               width: 160px; height: 160px; border-radius: 50%; background: #F4B942; }
#hero h1, #hero p, #hero .tag { position: relative; z-index: 1; }
#hero h1 { margin: 0 0 8px; font-size: 1.75rem; font-weight: 700; color: #fff; }
#hero p { margin: 0; max-width: 40em; line-height: 1.75; color: #D6EBF0; }
#hero .tag { display: inline-block; margin-top: 14px; padding: 3px 14px;
             border-radius: 999px; background: rgba(255,255,255,.15);
             font-size: .85rem; color: #fff; }
@media (max-width: 640px) { #hero::after { display: none; } }

.message.user { background: #226779 !important; border-color: #226779 !important; }
.message.user, .message.user * { color: #fff !important; }
.message.bot { background: var(--background-fill-primary) !important; }

#trace-panel { padding: 14px 18px; min-height: 220px; max-height: 640px; overflow-y: auto;
               background: var(--background-fill-primary);
               border: 1px solid var(--border-color-primary);
               border-left: 4px solid #F4B942; border-radius: 12px; }
#trace-panel h3 { margin-top: 0; font-size: 1.05rem; }
#trace-panel code { white-space: pre-wrap; word-break: break-all; font-size: .82rem; }
"""

HERO_HTML = """
<div id="hero">
  <h1>晴旅小屋 客服</h1>
  <p>問入住規定、查房型價格、試算住宿費用。每次回答都由 AI 查詢資料庫，右側會列出它用了哪些工具。</p>
  <span class="tag">課程示範，房價與供應狀態不是即時資訊</span>
</div>
"""

blocks_kwargs = {"title": "晴旅小屋 AI 客服"}
launch_kwargs = {}
# Gradio 6 把 theme / css 移到 launch()；舊版仍放在 Blocks()
if "theme" in inspect.signature(gr.Blocks.launch).parameters:
    launch_kwargs.update(theme=THEME, css=CSS)
else:
    blocks_kwargs.update(theme=THEME, css=CSS)


# Tool-choice test cases: same purpose as Unit G-3 Step 9.
# A pass means the expected tool was called at least once, or no tool was called
# for an out-of-scope question. This does not grade answer correctness.
TOOL_CHOICE_TESTS = [
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


def run_tool_choice_test():
    """Run fresh single-turn conversations and show actual tool choices."""
    rows = []
    correct = 0
    for index, (question, expected) in enumerate(TOOL_CHOICE_TESTS, 1):
        try:
            _, trace = run_agent([{"role": "user", "content": question}])
            used = [entry["tool"] for entry in trace]
            passed = (not used) if expected is None else expected in used
            correct += int(passed)
            status = "OK" if passed else "MISS"
            actual = ", ".join(used) if used else "無工具"
        except Exception as exc:
            status = "ERROR"
            actual = f"{type(exc).__name__}（詳見終端機）"
            print(f"Tool-choice test Q{index}: {type(exc).__name__}: {exc}")
        rows.append([index, question, expected or "無工具", actual, status])

    total = len(TOOL_CHOICE_TESTS)
    errors = sum(row[-1] == "ERROR" for row in rows)
    score = f"### 測試結果：{correct}/{total} = {correct / total:.1%}"
    if errors:
        score += f"\n\n有 {errors} 題執行錯誤，已按未通過計算；請檢查終端機。"
    score += "\n\n請截取這張表及分數，作為報告第 5.2 題的測試輸出。"
    return rows, score

with gr.Blocks(**blocks_kwargs) as demo:
    gr.HTML(HERO_HTML)
    with gr.Tabs():
        with gr.Tab("第一頁｜AI 客服"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(value=start_chat(), height=520, show_label=False)
                    with gr.Row():
                        user_box = gr.Textbox(
                            placeholder="輸入問題，例如：幾點可以入住？",
                            show_label=False, scale=8, container=False,
                        )
                        send_button = gr.Button("送出", variant="primary", scale=1, min_width=80)
                        clear_button = gr.Button("清除對話", scale=1, min_width=96)
                    with gr.Accordion("範例問題（點選後按送出）", open=True):
                        for group, questions in SAMPLE_GROUPS.items():
                            gr.Examples(examples=questions, inputs=user_box, label=group)
                with gr.Column(scale=2):
                    trace_panel = gr.Markdown(EMPTY_TRACE, elem_id="trace-panel")

            send_button.click(add_user_message, [user_box, chatbot], [user_box, chatbot]).then(
                answer_question, chatbot, [chatbot, trace_panel]
            )
            user_box.submit(add_user_message, [user_box, chatbot], [user_box, chatbot]).then(
                answer_question, chatbot, [chatbot, trace_panel]
            )
            clear_button.click(start_chat, outputs=chatbot).then(
                lambda: ("", EMPTY_TRACE), outputs=[user_box, trace_panel]
            )

        with gr.Tab("第二頁｜Tool-Choice Test"):
            gr.Markdown(
                "## 工具選擇測試（Unit G-3 Step 9）\\n"
                "點擊按鈕，讓 AI 逐題回答 12 個獨立問題，檢查是否選擇預期工具。"
                "每次執行都會重新呼叫模型，結果可能不同。"
            )
            test_button = gr.Button("開始 12 題測試", variant="primary")
            test_score = gr.Markdown("### 尚未測試")
            test_table = gr.Dataframe(
                headers=["題號", "問題", "預期工具", "實際呼叫工具", "結果"],
                datatype=["number", "str", "str", "str", "str"],
                value=[],
                interactive=False,
                label="逐題測試紀錄",
            )
            gr.Markdown(
                "**判分方式：** 預期工具至少被呼叫一次即算 OK；非飯店問題預期不呼叫工具。"
                "這裡只評估工具選擇，不評估回答內容正確性。"
            )
            test_button.click(
                run_tool_choice_test, inputs=[], outputs=[test_table, test_score]
            )

if __name__ == "__main__":
    demo.launch(**launch_kwargs)
