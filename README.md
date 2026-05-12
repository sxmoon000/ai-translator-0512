# 🌐 AI 翻译器

> Transformer + OPUS-MT 神经网络翻译

## 🧠 知识点
- **Transformer**: 抛弃 RNN，全靠自注意力 — Google 2017 年论文改变一切
- **自注意力**: 每个词和所有词算相关性 → 理解长距离依赖
- **Cross-Attention**: Decoder 看 Encoder 输出 → 翻译时对原文"对齐"
- **BPE Tokenizer**: 把罕见词拆成子词 → "unbelievable" → "un"+"believe"+"able"
- **MarianMT**: Helsinki-NLP 训练的轻量翻译模型，200+ 语言对

## 🚀 运行
```bash
pip install -r requirements.txt && python src/translate.py
```

---

Day 9 | 2026-05-12 | [sxmoon000](https://github.com/sxmoon000)
