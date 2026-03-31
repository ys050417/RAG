# 使⽤开源⼤语⾔模型 Qwen搭建 — RAG 

## 一、Langchain-chain 链

LangChain的链（Chain）是将多个组件组合在一起以实现复杂应用的核心机制，允许开发者创建模块化和可复用的工作流。

### （一）什么是链（Chain）

在LangChain中，链是对组件调用的序列，可以包含其他链。它允许将多个模块化组件（如模型、提示模板、记忆模块等）以特定方式组合，从而实现共同的用例。链的基本接口简单而强大，极大地简化了复杂应用的实现，并使其更模块化，便于调试和维护。

### （二）链的工作原理

链的核心运行逻辑是以串行化数据流的方式传递处理结果，前一环节的输出自动作为下一环节的输入，最终完成复杂任务的端到端执行。通过将多个链组合在一起，或将链与其他组件组合，可以构建更复杂的工作流。

### （三）链的类型

|   类型   |                             说明                             |
| :------: | :----------------------------------------------------------: |
| LLMChain | 最基本的链类型，结合提示模板和语言模型。它接受用户输入，格式化提示，并返回模型的响应。 |
|  顺序链  | 如SimpleSequentialChain和SequentialChain，允许按顺序执行多个步骤，处理多个输入和输出。 |
| 自定义链 |    开发者可以通过继承Chain类来实现自定义链，满足特定需求     |

### （四）安装 Langchain 相关核心依赖

|                   依赖库                    |                 安装命令                 |
| :-----------------------------------------: | :--------------------------------------: |
|               核心Langchain库               |      pip install langchain==0.3.23       |
|        OpenAI接口适配（对接ollama）         |   pip install langchain-openai==0.3.12   |
| 检索链所需依赖（Embedding+向量库+社区组件） | pip install sentence-transformers==3.3.0 |
| 检索链所需依赖（Embedding+向量库+社区组件） |       pip install faiss-cpu==1.9.0       |
| 检索链所需依赖（Embedding+向量库+社区组件） | pip install langchain-huggingface==0.1.2 |
| 检索链所需依赖（Embedding+向量库+社区组件） | pip install langchain-community==0.3.21  |

### （五）下载 Embedding 模型（用于检索链）

```
from modelscope.hub.snapshot_download import snapshot_download
# 下载Embedding模型到本地models目录
emb_model_dir = snapshot_download('AI-ModelScope/bge-large-zh-v1.5', cache_dir='models')
print("Embedding模型下载完成，路径：", emb_model_dir)
```

<img src = "结果\模型目录.png">

<img  src = "结果\Langchain-chainy实现结果.png">

### （六）实现Langchain-chain 三大核心链

```
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 通用配置：对接ollama的Qwen2.5模型（OpenAI兼容接口）
from langchain_openai import ChatOpenAI

# 初始化大语言模型（ollama本地服务，默认端口11434）
chat_model = ChatOpenAI(
    openai_api_key="ollama",
    base_url="http://localhost:11434/v1",
    model="qwen2.5:0.5b"
)

# 1. 基础LLMChain（新版本LCEL写法）
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

system_message = "你是一个贴心的智能助手，回答简洁易懂。"
human_message = "{user_question}"
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("human", human_message),
])
llm_chain = chat_prompt | chat_model | StrOutputParser()

# 2. 检索链RetrievalQA（RAG核心）
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# 自动创建测试文件
if not os.path.exists("sanguoyanyi.txt"):
    with open("sanguoyanyi.txt", "w", encoding="utf-8") as f:
        f.write("三国演义中刘备的五虎上将是：关羽、张飞、赵云、马超、黄忠。")

# 加载文档
loader = TextLoader("sanguoyanyi.txt", encoding='utf-8')
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = text_splitter.split_documents(docs)

# 离线嵌入，不联网
embedding = FakeEmbeddings(size=1024)

# 构建向量库
vs = FAISS.from_documents(chunks, embedding)
retriever = vs.as_retriever()

# 提示模板
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "根据以下已知信息回答用户问题，仅用已知信息，不要编造：\n{context}"),
    ("human", "{question}"),
])

# 构建RAG链
retrieval_qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": qa_prompt}
)

# 3. 自定义链
custom_prompt = ChatPromptTemplate.from_template("说出一句包含{topic}的古诗，仅输出诗句，不要多余内容。")
custom_chain = custom_prompt | chat_model | StrOutputParser()

# ==================== 运行 ====================
if __name__ == "__main__":
    # 1 基础链
    print("=== LLMChain测试结果 ===")
    response = llm_chain.invoke({"user_question": "你好，介绍一下你自己"})
    print(response)

    # 2 RAG检索链
    print("\n=== RetrievalQA链测试结果 ===")
    qa_response = retrieval_qa_chain.invoke({"query": "五虎上将有哪些？"})
    print(qa_response["result"])

    # 检索文档
    related_docs = retriever.invoke("五虎上将有哪些？")
    print("\n=== 检索到的相关文档 ===")
    for i, doc in enumerate(related_docs):
        print(f"文档{i + 1}：{doc.page_content}")

    # 3 自定义链
    print("\n=== 自定义链测试结果 ===")
    poem_response = custom_chain.invoke({"topic": "花"})
    print(poem_response)
```

### （七）成语接龙实现

```
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
```

<img src = "结果\成语接龙实现结果.png">

