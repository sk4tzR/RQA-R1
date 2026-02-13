# ============================================================
# app.py — основное Streamlit-приложение для RQA
# ============================================================

import streamlit as st
import pandas as pd
import json
import time
import gc
import psutil
import os
import torch
from utils import (
    RQAJudge,
    load_texts_from_uploaded_file,
    format_result_for_streamlit,
    ERROR_NAMES_RU,
    ERROR_THRESHOLDS
)

# ============================================================
# Мониторинг памяти и автоматический сброс кэша
# ============================================================

def get_memory_usage():
    """Возвращает использование памяти в процентах и MB"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    # Streamlit Cloud имеет лимит 1GB = 1024MB
    memory_percent = (memory_mb / 1024) * 100
    return memory_percent, memory_mb

def check_memory_and_cleanup(threshold=85):
    """
    Проверяет использование памяти и принудительно очищает кэш,
    если превышен порог (по умолчанию 85%)
    """
    memory_percent, memory_mb = get_memory_usage()
    
    # Сохраняем в session_state для отладки
    st.session_state['last_memory_check'] = {
        'percent': memory_percent,
        'mb': memory_mb,
        'time': time.strftime('%H:%M:%S')
    }
    
    if memory_percent > threshold:
        st.warning(f"⚠️ Использование памяти: {memory_percent:.1f}% ({memory_mb:.0f} MB). Очищаю кэш...")
        
        # Очищаем кэш модели
        if 'judge' in st.session_state:
            del st.session_state['judge']
        
        # Принудительная сборка мусора
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Сбрасываем флаг загрузки
        st.cache_resource.clear()
        st.session_state['model_loaded'] = False
        
        st.success("✅ Кэш очищен. Перезапустите анализ.")
        return True
    return False

# ============================================================
# Кэширование модели с мониторингом
# ============================================================

@st.cache_resource(ttl=300, max_entries=1, show_spinner="Загружаю модель...")
def load_judge():
    """Загружает модель с TTL 5 минут"""
    return RQAJudge()

# ============================================================
# Инициализация session state
# ============================================================

if 'model_loaded' not in st.session_state:
    st.session_state['model_loaded'] = False
if 'last_cleanup' not in st.session_state:
    st.session_state['last_cleanup'] = time.time()

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

# ============================================================
# Боковая панель с мониторингом
# ============================================================

with st.sidebar:
    st.header("📊 Мониторинг")
    
    # Показываем использование памяти
    memory_percent, memory_mb = get_memory_usage()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Память", f"{memory_mb:.0f} MB")
    with col2:
        st.metric("Использование", f"{memory_percent:.1f}%")
    
    # Прогресс-бар памяти
    if memory_percent < 70:
        st.progress(int(memory_percent) / 100, text="✅ Норма")
    elif memory_percent < 85:
        st.progress(int(memory_percent) / 100, text="⚠️ Средне")
    else:
        st.progress(int(memory_percent) / 100, text="🔴 Критично")
    
    # Кнопка ручной очистки
    if st.button("🧹 Очистить кэш сейчас"):
        st.cache_resource.clear()
        if 'judge' in st.session_state:
            del st.session_state['judge']
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        st.session_state['model_loaded'] = False
        st.success("✅ Кэш очищен!")
        st.rerun()
    
    st.markdown("---")
    
    # Режимы работы
    mode = st.radio(
        "Выберите режим:",
        ["📝 Одиночный ввод", "📄 Множественный ввод", "📂 Загрузка из файла"]
    )

# ============================================================
# Загрузка модели (только если память позволяет)
# ============================================================

if memory_percent < 90:
    if not st.session_state['model_loaded']:
        with st.spinner("Загружаю модель... Это может занять минуту."):
            judge = load_judge()
            st.session_state['model_loaded'] = True
        st.success("✅ Модель готова к работе!")
    else:
        judge = load_judge()  # Получаем из кэша
else:
    st.error("🔴 Критическое использование памяти! Невозможно загрузить модель.")
    st.stop()

# ============================================================
# Автоматическая проверка памяти перед каждым анализом
# ============================================================

def safe_infer(judge, text):
    """Безопасный инференс с проверкой памяти"""
    memory_percent, _ = get_memory_usage()
    
    if memory_percent > 85:
        st.warning("⚠️ Высокое использование памяти. Очищаю кэш...")
        st.cache_resource.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Перезагружаем модель
        judge = load_judge()
    
    return judge.infer(text)

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

    # Ошибки ниже порога (закомментировано, но можно включить)
    # below = [e for e in result["top_errors"] if not e["above_threshold"] and e["probability"] > 0.01]
    # if below:
    #     with st.expander("📉 Ошибки ниже порога уверенности"):
    #         for e in below:
    #             name_ru = ERROR_NAMES_RU.get(e["type"], e["type"])
    #             st.write(f"- {name_ru}: {e['probability']*100:.1f}% (порог {ERROR_THRESHOLDS[e['type']]*100:.0f}%)")

    st.metric(
        "📊 Disagreement", 
        f"{result['disagreement']:.3f}",
        help="Согласованность двух классификаторов модели. "
             "0.00–0.10: высокая уверенность, "
             "0.10–0.30: средняя, "
             ">0.30: низкая (текст сложный для интерпретации)."
    )

# ============================================================
# Режим 1: Одиночный ввод
# ============================================================
if mode == "📝 Одиночный ввод":
    st.header("📝 Одиночный ввод")
    text = st.text_area("Введите текст для анализа:", height=150)
    if st.button("🔍 Анализировать", key="single_btn") and text:
        with st.spinner("Анализирую..."):
            result = safe_infer(judge, text)
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
                    res = safe_infer(judge, txt)
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
                    res = safe_infer(judge, txt)
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
