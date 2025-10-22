# app.py (PyTorch + LSTM 版本)
import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

st.set_page_config(page_title="IMDb The Sentiment Analysis of Movie Reviews（PyTorch LSTM）", layout="wide")
st.title("🎬 IMDb The Sentiment Analysis of Movie Reviews（PyTorch + LSTM）")

# ---------- 辅助：文本预处理 / 词表 ----------
def simple_tokenize(text):
    # 非严格分词：小写，按空白分词，去掉前后空白
    return str(text).lower().strip().split()

def build_vocab(texts, num_words=10000, oov_token="<OOV>"):
    # 统计词频并建立词到索引映射，保留最常见 num_words-1（留一个给 OOV）
    freq = {}
    for t in texts:
        for w in simple_tokenize(t):
            freq[w] = freq.get(w, 0) + 1
    # 排序
    sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    # 保留词表（从1开始），0 用作 padding
    vocab = {oov_token:1}
    idx = 2
    for w, _ in sorted_items:
        if idx >= num_words + 1:  # 1 reserved for OOV; index starts at 2
            break
        vocab[w] = idx
        idx += 1
    return vocab

def texts_to_sequences(texts, vocab, maxlen=200):
    sequences = []
    oov_idx = vocab.get("<OOV>", 1)
    for t in texts:
        seq = [vocab.get(w, oov_idx) for w in simple_tokenize(t)]
        # pad/truncate
        if len(seq) >= maxlen:
            seq = seq[:maxlen]
        else:
            seq = seq + [0] * (maxlen - len(seq))
        sequences.append(seq)
    return np.array(sequences, dtype=np.int64)

# ---------- PyTorch Dataset ----------
class ReviewDataset(Dataset):
    def __init__(self, sequences, labels):
        self.x = torch.tensor(sequences, dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# ---------- 模型定义 ----------
class LSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_layers=1, dropout=0.3):
        super().__init__()
        # padding index 0 被用于填充
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)  # (batch, seq_len, emb)
        out, (hn, cn) = self.lstm(emb)  # out: (batch, seq_len, hidden)
        # 取最后时间步输出（或用 hn）
        last = out[:, -1, :]  # (batch, hidden)
        last = self.dropout(last)
        out = self.fc(last)
        return self.sigmoid(out).squeeze(1)  # (batch,)

# ---------- 加载数据 ----------
@st.cache_data
def load_data():
    if not os.path.exists("IMDB Dataset.csv"):
        st.warning("请将 'IMDB Dataset.csv' 放到应用根目录后再运行（从 Kaggle 下载）。")
        return pd.DataFrame({"review": [], "sentiment": []})
    df = pd.read_csv("IMDB Dataset.csv")
    if 'review' not in df.columns or 'sentiment' not in df.columns:
        st.error("CSV 文件缺少 'review' 或 'sentiment' 列，请确认数据文件格式。")
        return pd.DataFrame({"review": [], "sentiment": []})
    df = df[['review', 'sentiment']].dropna()
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    df['review_len'] = df['review'].astype(str).apply(len)
    return df

df = load_data()

# ---------- 侧边栏：筛选与超参 ----------
st.sidebar.header("Data Filtering and Training Settings")
min_len = int(df['review_len'].min()) if not df.empty else 0
max_len = int(df['review_len'].max()) if not df.empty else 1000
length_range = st.sidebar.slider("Length range of film reviews", min_len, max_len, (min_len, max_len))

filtered = df[(df['review_len'] >= length_range[0]) & (df['review_len'] <= length_range[1])] if not df.empty else df

sent_select = st.sidebar.multiselect("选择情感", options=['positive', 'negative'], default=['positive', 'negative'])
filtered = filtered[filtered['sentiment'].isin(sent_select)] if not df.empty else filtered

# 训练超参数
st.sidebar.markdown("---")
num_words = st.sidebar.number_input("(num_words)", min_value=2000, max_value=50000, value=10000, step=1000)
maxlen = st.sidebar.number_input("(maxlen)", min_value=50, max_value=1000, value=200, step=50)
embedding_dim = st.sidebar.selectbox("Embedding dimension", options=[50, 100, 128, 200], index=2)
hidden_dim = st.sidebar.number_input("number of LSTM hidden layers", min_value=32, max_value=512, value=128, step=32)
batch_size = st.sidebar.selectbox("Batch size", options=[32, 64, 128], index=0)
epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=10, value=3)
train_on_subset = st.sidebar.checkbox("Train only on a small subset (to speed up demonstration)", value=True)
subset_size = st.sidebar.number_input("Number of subset samples (if subset is used）", min_value=100, max_value=20000, value=2000, step=100)

st.header("Data Preview")
st.write(f"Number of samples after filtering:{len(filtered)}")
st.dataframe(filtered.sample(10) if len(filtered) >= 10 else filtered)

# ---------- EDA 区域 ----------
st.header(" EDA：Exploratory Data Analysis")
st.subheader("length distribution of reviews（based on filtered data）")
fig, ax = plt.subplots()
if not filtered.empty:
    filtered['review_len'].hist(bins=50, ax=ax)
ax.set_xlabel("length of review")
ax.set_ylabel("number of reviews")
st.pyplot(fig)

st.subheader("positive/negative word cloud")
sent_choice = st.radio("choose positive or negative", ['positive', 'negative'])
text = " ".join(filtered[filtered['sentiment'] == sent_choice]['review'].astype(str).tolist()) if not filtered.empty else ""
if text.strip():
    wc = WordCloud(width=800, height=300, background_color='white').generate(text)
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.imshow(wc, interpolation='bilinear')
    ax2.axis('off')
    st.pyplot(fig2)
else:
    #use English
    st.info("no data available for the selected sentiment.")

st.subheader("sentiment distribution")
if not filtered.empty:
    vc = filtered['sentiment'].value_counts()
    fig3, ax3 = plt.subplots()
    ax3.bar(vc.index, vc.values, color=['green', 'red'])
    ax3.set_ylabel("number")
    st.pyplot(fig3)
else:
    st.info("no data available after filtering.")

# ---------- 模型训练 / 加载 / 预测 ----------
st.header("LSTM model training and prediction")

model_path = "pytorch_lstm_model.pt"
vocab_path = "vocab.json"

use_existing = st.checkbox("use model that exists", value=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"device：{device}")

model = None
vocab = None

if use_existing and os.path.exists(model_path) and os.path.exists(vocab_path):
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        # vocab size = max index + 1 or num_words + 2 etc. We'll use num_words+2 as safe upper bound
        vocab_size = max(list(vocab.values())) + 1
        model = LSTMSentiment(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        st.success("The existing model and vocabulary have been loaded.")
    except Exception as e:
        st.error(f"failed：{e}")
        model = None
        vocab = None

# 训练触发
if st.button("training new model"):
    if df.empty:
        st.error("data is empty, cannot train model.")
    else:
        # 选择数据（可用全量或子集）
        data = df.sample(n=subset_size, random_state=42) if train_on_subset else df
        texts = data['review'].astype(str).tolist()
        labels = data['label'].astype(int).tolist()
        st.info(f"begin to train：num of input={len(texts)}，length of vocabulary={num_words}，length of sequence={maxlen}")

        # build vocab
        vocab = build_vocab(texts, num_words=num_words)
        # 保存词表
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False)

        sequences = texts_to_sequences(texts, vocab, maxlen=maxlen)
        X_train, X_val, y_train, y_val = train_test_split(sequences, labels, test_size=0.15, random_state=42)

        train_ds = ReviewDataset(X_train, y_train)
        val_ds = ReviewDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        vocab_size = max(list(vocab.values())) + 1
        model = LSTMSentiment(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_val_loss = float('inf')
        patience = 2
        wait = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)
            train_loss = running_loss / len(train_loader.dataset)
            # validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    val_loss += loss.item() * xb.size(0)
            val_loss = val_loss / len(val_loader.dataset)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            st.write(f"Epoch {epoch}/{epochs} — train_loss: {train_loss:.4f} — val_loss: {val_loss:.4f}")

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
                # 保存模型
                torch.save(model.state_dict(), model_path)
                st.write("best model saved.")
            else:
                wait += 1
                if wait >= patience:
                    st.write("Early stopping triggered.")
                    break

        # 绘制训练曲线
        fig_hist, ax_hist = plt.subplots()
        ax_hist.plot(history["train_loss"], label="train_loss")
        ax_hist.plot(history["val_loss"], label="val_loss")
        ax_hist.set_xlabel("epoch")
        ax_hist.set_ylabel("loss")
        ax_hist.legend()
        st.pyplot(fig_hist)

        st.success("Model training completed.")

# ---------- 单条文本预测 ----------
st.subheader("Single Text Prediction")
input_text = st.text_area("please enter the movie review text for sentiment prediction:", height=150)
if st.button("Predict Sentiment"):
    if input_text.strip() == "":
        st.warning("please enter valid text.")
    else:
        if model is None or vocab is None:
            st.error("No model available. Please train a new model or load an existing one.")
        else:
            seq = texts_to_sequences([input_text], vocab, maxlen=maxlen)
            x_tensor = torch.tensor(seq, dtype=torch.long).to(device)
            model.eval()
            with torch.no_grad():
                pred = model(x_tensor)
                prob = float(pred.cpu().numpy()[0])
                label = "Positive" if prob >= 0.5 else "Negative"
                st.success(f"prediction：{label}（probability={prob:.3f}）")
