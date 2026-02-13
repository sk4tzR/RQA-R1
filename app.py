# ============================================================
# app.py — основное Streamlit-приложение для RQA
# ============================================================

import streamlit as st
import pandas as pd
import json
import time
from utils import RQAJudge, load_texts_from_uploaded_file, format_result_for_streamlit, ERROR_NAMES_RU

# Настройка страницы
st.set_page_config(
    page_title="RQA — Анализ логических ошибок",
    page_icon="🤖",
    layout="wide"
)

# Заголовок
st.title("🤖 RQA — Детектор логических ошибок")
st.markdown("Модель анализирует текст и выявляет логические ошибки: ложная причинность, неподкреплённые утверждения, обобщения, противоречия и др.")
st.markdown("---")

# Кэширование модели (загружается один раз)
@st.cache_resource
def load_judge():
    return RQAJudge()

# Загружаем модель
with st.spinner("Загружаю модель... Это может занять минуту."):
    judge = load_judge()
st.success("Модель готова к работе!")

# Боковая панель с выбором режима
mode = st.sidebar.radio(
    "Выберите режим работы:",
    ["📝 Одиночный ввод", "📄 Множественный ввод", "📂 Загрузка из файла"]
)

# ============================================================
# Функция для отображения одного результата
# ============================================================
def display_result(result):
    col1, col2 = st.columns([3, 1])
    with col1:
        if result['has_issue']:
            st.error(f"❌ Проблема обнаружена ({result['issue_probability']*100:.1f}%)")
        else:
            st.success(f"✅ Проблем не обнаружено ({result['issue_probability']*100:.1f}%)")
    with col2:
        st.metric("Уверенность", result['confidence'])

    if result['borderline']:
        st.warning("⚠️ Пограничный случай: аргументативный текст")
    if result['hidden_problem']:
        st.info("🟡 Скрытая проблема: возможны неявные предпосылки")

    if result['explicit_errors']:
        st.subheader("❌ Явные логические ошибки:")
        for name, prob in result['explicit_errors']:
            st.error(f"**{ERROR_NAMES_RU[name]}** — {prob*100:.1f}%")

    below = [e for e in result["top_errors"] if not e["above_threshold"] and e["probability"] > 0.01]
    if below:
        with st.expander("📉 Ошибки ниже порога уверенности"):
            for e in below:
                name_ru = ERROR_NAMES_RU.get(e["type"], e["type"])
                st.write(f"- {name_ru}: {e['probability']*100:.1f}% (порог {ERROR_THRESHOLDS[e['type']]*100:.0f}%)")

    st.metric("📊 Disagreement", f"{result['disagreement']:.3f}")

# ============================================================
# Режим 1: Одиночный ввод
# ============================================================
if mode == "📝 Одиночный ввод":
    st.header("📝 Одиночный ввод")
    text = st.text_area("Введите текст для анализа:", height=150)
    if st.button("🔍 Анализировать", key="single_btn") and text:
        with st.spinner("Анализирую..."):
            result = judge.infer(text)
        st.markdown("---")
        display_result(result)

# ============================================================
# Режим 2: Множественный ввод (построчно)
# ============================================================
elif mode == "📄 Множественный ввод":
    st.header("📄 Множественный ввод")
    st.markdown("Введите несколько текстов, **каждый с новой строки**.")
    texts_input = st.text_area("Тексты (каждый с новой строки):", height=200)
    if st.button("🔍 Анализировать все", key="multi_btn") and texts_input.strip():
        texts = [t.strip() for t in texts_input.split("\n") if t.strip()]
        if texts:
            st.info(f"Найдено {len(texts)} текстов. Начинаю анализ...")
            progress_bar = st.progress(0)
            results = []
            for i, txt in enumerate(texts):
                with st.spinner(f"Анализ текста {i+1}..."):
                    res = judge.infer(txt)
                    results.append(res)
                progress_bar.progress((i + 1) / len(texts))
            st.success("Анализ завершён!")

            # Вывод результатов
            for i, res in enumerate(results):
                with st.expander(f"📄 Текст #{i+1}"):
                    st.write(res['text'])
                    display_result(res)

            # Кнопка для скачивания JSON
            export_data = []
            for r in results:
                export_data.append({
                    "text": r["text"],
                    "has_issue": r["has_issue"],
                    "issue_probability": r["issue_probability"],
                    "confidence": r["confidence"],
                    "explicit_errors": [(err, prob) for err, prob in r["explicit_errors"]],
                    "hidden_problem": r["hidden_problem"],
                    "disagreement": r["disagreement"],
                    "top_errors": r["top_errors"]
                })
            json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
            st.download_button(
                label="📥 Скачать результаты в JSON",
                data=json_str,
                file_name="rqa_results.json",
                mime="application/json"
            )

# ============================================================
# Режим 3: Загрузка из файла
# ============================================================
elif mode == "📂 Загрузка из файла":
    st.header("📂 Загрузка из файла")
    st.markdown("Поддерживаются форматы **.txt**, **.csv** (колонка 'text'), **.json** (список строк).")
    uploaded_file = st.file_uploader("Выберите файл", type=['txt', 'csv', 'json'])

    if uploaded_file and st.button("🔍 Анализировать файл", key="file_btn"):
        try:
            texts = load_texts_from_uploaded_file(uploaded_file)
            if not texts:
                st.warning("Файл пуст или не содержит текстов.")
            else:
                st.info(f"Загружено {len(texts)} текстов. Начинаю анализ...")
                progress_bar = st.progress(0)
                results = []
                stats = {"total": 0, "with_issue": 0, "error_counts": {}}
                for i, txt in enumerate(texts):
                    res = judge.infer(txt)
                    results.append(res)
                    stats["total"] += 1
                    if res["has_issue"]:
                        stats["with_issue"] += 1
                        for err, _ in res["explicit_errors"]:
                            stats["error_counts"][err] = stats["error_counts"].get(err, 0) + 1
                    progress_bar.progress((i + 1) / len(texts))

                st.success("Анализ завершён!")

                # Сводная статистика
                st.subheader("📊 Сводная статистика")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Всего текстов", stats["total"])
                with col2:
                    pct = stats["with_issue"] / stats["total"] * 100 if stats["total"] else 0
                    st.metric("С проблемой", f"{stats['with_issue']} ({pct:.1f}%)")

                if stats["error_counts"]:
                    st.write("**Распределение ошибок:**")
                    df = pd.DataFrame(
                        [(ERROR_NAMES_RU[err], count) for err, count in stats["error_counts"].items()],
                        columns=["Тип ошибки", "Количество"]
                    ).sort_values("Количество", ascending=False)
                    st.dataframe(df, use_container_width=True)

                # Вывод первых нескольких результатов (сворачиваемо)
                with st.expander("📄 Детальные результаты по текстам"):
                    for i, res in enumerate(results):
                        st.markdown(f"**Текст #{i+1}**")
                        st.write(res['text'])
                        display_result(res)
                        st.markdown("---")

                # Кнопка для скачивания JSON
                export_data = []
                for r in results:
                    export_data.append({
                        "text": r["text"],
                        "has_issue": r["has_issue"],
                        "issue_probability": r["issue_probability"],
                        "confidence": r["confidence"],
                        "explicit_errors": [(err, prob) for err, prob in r["explicit_errors"]],
                        "hidden_problem": r["hidden_problem"],
                        "disagreement": r["disagreement"],
                        "top_errors": r["top_errors"]
                    })
                json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
                st.download_button(
                    label="📥 Скачать все результаты в JSON",
                    data=json_str,
                    file_name="rqa_file_results.json",
                    mime="application/json"
                )

        except Exception as e:
            st.error(f"Ошибка при обработке файла: {str(e)}")
