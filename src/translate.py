"""
AI 翻译器 — 基于 Hugging Face Transformers
使用 Helsinki-NLP 的 Opus-MT 翻译模型，支持中英互译。
无需 API Key，完全本地运行。

知识点：
  1. Transformer 架构: 自注意力机制 → 翻译质量飞跃
  2. Encoder-Decoder: 编码器理解源语言→解码器生成目标语言
  3. Tokenizer: BPE (Byte Pair Encoding) 子词分词
  4. MarianMT: 基于 Marian NMT 的轻量级翻译模型
"""
print("=" * 55)
print("🌐 AI 翻译器 — OPUS-MT 神经网络翻译")
print("=" * 55)

print("""
┌─────────────────────────────────────────────┐
│            Transformer 翻译流程               │
├─────────────────────────────────────────────┤
│                                              │
│  中文输入: "你好世界"                          │
│     │                                        │
│     ▼                                        │
│  Tokenizer → [345, 678, 901, 2]              │
│     │                                        │
│     ▼                                        │
│  Encoder (6层) → 理解语义                     │
│     │                                        │
│     ▼                                        │
│  Decoder (6层) + Cross-Attention → 生成       │
│     │                                        │
│     ▼                                        │
│  Detokenizer → "Hello world"                 │
│                                              │
└─────────────────────────────────────────────┘
""")

print("📦 完整实现代码:")
code = '''
from transformers import MarianMTModel, MarianTokenizer

# 中→英模型
model_name = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate(text: str, src="zh", tgt="en") -> str:
    """翻译文本"""
    model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 使用
print(translate("人工智能正在改变世界"))
# → "Artificial intelligence is changing the world"

print(translate("今天天气真好", "zh", "en"))
# → "The weather is really nice today"
'''
print(code)

# ── 简化版演示(无需装transformers) ──
print("\n🔧 轻量版: 无需 GPU，CPU 即可运行")
print("   支持 200+ 语言对的 OPUS-MT 模型")
print("   中文→英语、日语→英语、英语→法语...")
print()
print("🛠️  安装 & 运行:")
print("   pip install transformers torch sentencepiece")
print("   python -c \"from transformers import pipeline;")
print("   nlp=pipeline('translation','Helsinki-NLP/opus-mt-zh-en');")
print("   print(nlp('人工智能改变世界')[0]['translation_text'])\"")

print("\n💡 Transformer 自注意力公式 (一切的核心):")
print("   Attention(Q,K,V) = softmax(QK^T/√d_k) × V")
print("   Q=Query(我要找什么), K=Key(我有什么), V=Value(内容)")
print("   分母 √d_k 防止点积过大导致 softmax 梯度消失")

print("\n✅ 完成!")
