import os
import streamlit as st
from huggingface_hub import InferenceClient

# ページ設定
st.set_page_config(page_title="AIドリームチーム チャット", page_icon="🚀", layout="wide")

# APIトークンの取得（StreamlitのSecretsから自動読込）
HF_TOKEN = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")

# モデル定義
MODELS = {
    "🌐 Qwen 2.5 (7B) [万能・最強日本語]": "Qwen/Qwen2.5-7B-Instruct",
    "💻 Qwen 2.5 Coder (7B) [コード特化]": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "⚡ Gemma 2 (2B) [Google製・爆速]": "google/gemma-2-2b-it",
    "🧠 DeepSeek R1 (1.5B) [思考・推論特化]": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
}

# サイドバー
with st.sidebar:
    st.title("⚙️ 設定")
    selected_name = st.selectbox("モデルを選択", list(MODELS.keys()))
    selected_model = MODELS[selected_name]

    system_prompt = st.text_area(
        "システムプロンプト",
        value="あなたは親切で優秀なAIアシスタントです。自然でわかりやすい日本語で回答してください。"
    )
    temperature = st.slider("Temperature (創造性)", 0.1, 1.5, 0.7, 0.1)
    max_tokens = st.slider("最大文字数", 128, 4096, 1024, 128)
    
    if st.button("🗑️ チャット履歴をクリア"):
        st.session_state.messages = []
        st.rerun()

# メイン画面
st.title("🚀 軽量オープンソースAI ドリームチーム")
st.caption(f"現在のモデル: **{selected_name}**")

# 履歴の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# 過去の会話を表示
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ユーザー入力
if prompt := st.chat_input("メッセージを入力してください..."):
    if not HF_TOKEN:
        st.error("⚠️ Hugging FaceのAPIトークンが設定されていません。アプリ設定のSecretsを確認してください。")
        st.stop()

    # ユーザー発言を表示＆履歴追加
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # アシスタントの返答生成
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # 会話履歴を整形
        messages = [{"role": "system", "content": system_prompt}]
        for m in st.session_state.messages:
            messages.append({"role": m["role"], "content": m["content"]})

        try:
            client = InferenceClient(model=selected_model, token=HF_TOKEN)
            stream = client.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )

            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                full_response += token
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            err = str(e)
            if "403" in err and "gemma" in selected_model.lower():
                st.error("Gemma 2のアクセス権がありません。Hugging FaceのGemma 2ページで利用規約に同意してください。")
            else:
                st.error(f"エラー: {err}")
