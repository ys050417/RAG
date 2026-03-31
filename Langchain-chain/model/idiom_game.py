from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_ollama import OllamaLLM
import re
import random
import time

# ===================== 模型：qwen3:8b =====================
MODEL_NAME = "qwen3:8b"
llm = OllamaLLM(model=MODEL_NAME, temperature=0.1)

# ===================== 加载本地成语库 =====================
def load_idiom_lib(file_path="成语大全.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"❌ 未找到成语文件！错误信息：{e}")
        exit()

    idioms = re.findall(r"[\u4e00-\u9fa5]{4}", text)
    idiom_set = list(set([i.strip() for i in idioms if len(i.strip()) == 4]))
    return idiom_set, {idiom: True for idiom in idiom_set}

IDIOM_LIST, IDIOM_DICT = load_idiom_lib()

# ===================== 工具1：校验成语（本地 + 模型双保险） =====================
@tool
def validate_idiom(idiom: str) -> dict:
    """
    校验输入的内容是否为合法的四字成语
    参数:
        idiom: 需要校验的字符串
    返回:
        包含校验结果、原因、最后一个字的字典
    """
    idiom = idiom.strip()
    if len(idiom) != 4:
        return {"valid": False, "reason": "必须四字", "last": ""}
    if idiom not in IDIOM_DICT:
        return {"valid": False, "reason": "不在成语库", "last": ""}
    return {"valid": True, "reason": "合法", "last": idiom[-1]}

# ===================== 工具2：AI接成语（本地高速版，绝不卡顿） =====================
@tool
def find_next_idiom_fast(start_char: str) -> dict:
    """
    根据开头汉字快速查找下一个成语（本地成语库匹配）
    参数:
        start_char: 成语开头的汉字
    返回:
        包含是否找到、对应成语的字典
    """
    candidates = [idiom for idiom in IDIOM_LIST if idiom[0] == start_char]
    if not candidates:
        return {"ok": False, "idiom": ""}
    chosen = random.choice(candidates)
    IDIOM_LIST.remove(chosen)
    return {"ok": True, "idiom": chosen}

# ===================== 调用链 =====================
chain = (
    RunnableParallel({
        "user_idiom": RunnablePassthrough(),
        "target_char": RunnablePassthrough()
    })
    | RunnableLambda(lambda x: validate_idiom.invoke(x["user_idiom"]))
)

# ===================== 模式选择 =====================
def choose_mode():
    print("\n📌 模式选择：")
    print("1. 玩家先出（人VS AI）")
    print("2. AI 先出（人VS AI）")
    print("3. AI 全自动对战（推荐）")
    while True:
        c = input("请输入 1/2/3：").strip()
        if c in ["1", "2", "3"]:
            return c

# ===================== AI VS AI =====================
def ai_vs_ai():
    print("\n🤖 AI 全自动成语接龙启动！")
    print("=" * 50)
    current = random.choice(IDIOM_LIST)
    IDIOM_LIST.remove(current)
    print(f"AI1：{current}")

    while True:
        time.sleep(0.4)
        ai2 = find_next_idiom_fast.invoke(current[-1])
        if not ai2["ok"]:
            print("\nAI2 接不上 → AI1 胜利！🏆")
            break
        current = ai2["idiom"]
        print(f"AI2：{current}")

        time.sleep(0.4)
        ai1 = find_next_idiom_fast.invoke(current[-1])
        if not ai1["ok"]:
            print("\nAI1 接不上 → AI2 胜利！🏆")
            break
        current = ai1["idiom"]
        print(f"AI1：{current}")

# ===================== 人 VS AI =====================
def play_game():
    print("🎉 成语接龙（qwen3:8b）")
    print("✅ 本地成语库 + AI模型双支持")
    print("💡 输入 退出 结束游戏\n")
    mode = choose_mode()

    if mode == "3":
        ai_vs_ai()
        return

    current = ""
    if mode == "1":
        while True:
            user = input("\n你先出：").strip()
            if user == "退出":
                return
            res = chain.invoke(user)
            if not res["valid"]:
                print(f"❌ {res['reason']}")
                continue
            current = user
            if current in IDIOM_LIST:
                IDIOM_LIST.remove(current)
            print(f"你：{current}")
            break

    else:
        current = random.choice(IDIOM_LIST)
        IDIOM_LIST.remove(current)
        print(f"AI：{current}")

    while True:
        need = current[-1]
        user = input(f"\n请以【{need}】开头：").strip()
        if user == "退出":
            break

        res = chain.invoke(user)
        if not res["valid"]:
            print(f"❌ {res['reason']}")
            break
        if user[0] != need:
            print(f"❌ 必须以【{need}】开头！你输了")
            break
        if user in IDIOM_LIST:
            IDIOM_LIST.remove(user)

        ai = find_next_idiom_fast.invoke(user[-1])
        if not ai["ok"]:
            print("\n🎉 AI接不上，你赢了！")
            break
        current = ai["idiom"]
        print(f"AI：{current}")

if __name__ == "__main__":
    play_game()