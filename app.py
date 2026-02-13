# ============================================================
# app.py — основное Streamlit-приложение для RQA
# ============================================================

import streamlit as st
import pandas as pd
import json
import time
import gc
import os
import psutil

# Попытка импорта torch (для очистки GPU памяти)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from utils import (
    RQAJudge,
    load_texts_from_uploaded_file,
    ERROR_NAMES_RU,
    ERROR_THRESHOLDS
)

# ============================================================
# Мониторинг памяти и автоматическая очистка
# ============================================================

def get_memory_limit():
    """
    Определяет лимит памяти, доступный приложению (в МБ).
    Возвращает None, если лимит не определён (тогда используем общую память системы).
    """
    # 1. Проверяем переменные окружения (можно задать вручную)
    if 'STREAMLIT_MEMORY_LIMIT_MB' in os.environ:
        return float(os.environ['STREAMLIT_MEMORY_LIMIT_MB'])

    # 2. Для Streamlit Cloud (часто 1 ГБ)
    if os.environ.get('STREAMLIT_RUNTIME') == 'cloud':
        return 1024.0  # можно изменить, если появятся новые данные

    # 3. Проверяем cgroup лимиты (для Docker-контейнеров)
    try:
        # cgroup v1
        with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
            limit_bytes = int(f.read().strip())
            if limit_bytes < 2**63 - 1:  # бесконечность не учитываем
                return limit_bytes / (1024 * 1024)
    except (FileNotFoundError, ValueError):
        pass

    try:
        # cgroup v2
        with open('/sys/fs/cgroup/memory.max', 'r') as f:
            limit_str = f.read().strip()
            if limit_str != 'max':
                return int(limit_str) / (1024 * 1024)
    except (FileNotFoundError, ValueError):
        pass

    # Лимит не определён — будем использовать общую память системы
    return None

def get_memory_usage():
    """
    Возвращает:
        - percent: процент использования относительно доступного лимита (или общей памяти)
        - used_mb: используемая память в МБ
        - total_mb: доступная память (лимит или общая) в МБ
        - limit_known: True, если лимит известен точно
    """
    process = psutil.Process(os.getpid())
    used_mb = process.memory_info().rss / (1024 * 1024)

    limit_mb = get_memory_limit()
    if limit_mb is not None:
        # Используем известный лимит
        total_mb = limit_mb
        percent = (used_mb / limit_mb) * 100
        limit_known = True
    else:
        # Лимит неизвестен — ориентируемся на общую память системы
        total_mb = psutil.virtual_memory().total / (1024 * 1024)
        percent = (used_mb / total_mb) * 100
        limit_known = False

    return percent, used_mb, total_mb, limit_known

def perform_cleanup(reason="не указана"):
    """
    Принудительная очистка: удаляем модель, чистим кэш, GC, GPU.
    """
    # Удаляем модель из session_state, если она там есть
    if 'judge' in st.session_state:
        del st.session_state['judge']
    if 'model_loaded' in st.session_state:
        st.session_state['model_loaded'] = False

    # Сборка мусора Python
    gc.collect()

    # Очистка GPU памяти, если доступно
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Сброс кэша Streamlit (кэш ресурсов)
    st.cache_resource.clear()

    # Обновляем время последней очистки
    st.session_state['last_cleanup_time'] = time.time()

    # Показываем уведомление (только если не в фоновом режиме)
    st.toast(f"🧹 Очистка памяти: {reason}", icon="🗑️")

def auto_cleanup_if_needed(force=False):
    """
    Проверяет, нужно ли выполнить очистку:
      - если использование памяти > 85% от доступного лимита,
      - или если прошло больше 5 минут с последней очистки,
      - или если force=True.
    Возвращает True, если очистка была выполнена.
    """
    # Инициализируем счётчик времени, если его нет
    if 'last_cleanup_time' not in st.session_state:
        st.session_state['last_cleanup_time'] = time.time()

    percent, used_mb, total_mb, limit_known = get_memory_usage()
    current_time = time.time()
    time_since_last = current_time - st.session_state['last_cleanup_time']

    # Условия для очистки
    need_cleanup = False
    reasons = []

    # 1. По времени: каждые 5 минут (300 секунд)
    if time_since_last > 300:
        need_cleanup = True
        reasons.append(f"прошло {int(time_since_last)} сек")

    # 2. По памяти: если превышен порог 85%
    threshold = 85  # можно сделать настраиваемым
    if percent > threshold:
        need_cleanup = True
        reasons.append(f"память {percent:.1f}% > {threshold}%")

    # 3. Принудительная очистка
    if force:
        need_cleanup = True
        reasons.append("принудительно")

    if need_cleanup:
        reason_str = ", ".join(reasons)
        perform_cleanup(reason_str)
        return True

    return False

# ============================================================
# Загрузка модели с учётом состояния
# ============================================================

@st.cache_resource(ttl=300, max_entries=1, show_spinner="Загружаю модель...")
def _load_judge_cached():
    """Внутренняя функция для кэширования модели (используется только при перезагрузке)"""
    return RQAJudge()

def get_judge():
    """
    Возвращает экземпляр RQAJudge, загружая его при необходимости.
    Следит за флагом model_loaded в session_state.
    """
    if st.session_state.get('model_loaded', False) and 'judge' in st.session_state:
        return st.session_state['judge']

    # Если модель не загружена или флаг сброшен — загружаем
    with st.spinner("Загружаю модель... Это может занять минуту."):
        judge = _load_judge_cached()
        st.session_state['judge'] = judge
        st.session_state['model_loaded'] = True
    return judge

# ============================================================
# Инициализация session_state
# ============================================================
if 'model_loaded' not in st.session_state:
    st.session_state['model_loaded'] = False
if 'last_cleanup_time' not in st.session_state:
    st.session_state['last_cleanup_time'] = time.time()

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
# Боковая панель с мониторингом памяти
# ============================================================
with st.sidebar:
    st.header("📊 Мониторинг ресурсов")

    # Получаем текущие показатели памяти
    percent, used_mb, total_mb, limit_known = get_memory_usage()

    # Формируем подписи
    if limit_known:
        total_str = f"{total_mb:.0f} МБ (лимит)"
    else:
        total_str = f"{total_mb:.0f} МБ (система)"

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Используется", f"{used_mb:.0f} МБ")
    with col2:
        st.metric("Доступно", total_str)

    # Прогресс-бар (значение от 0 до 1)
    progress_value = min(percent / 100, 1.0)
    if percent < 70:
        st.progress(progress_value, text=f"🟢 {percent:.1f}%")
    elif percent < 85:
        st.progress(progress_value, text=f"🟡 {percent:.1f}%")
    else:
        st.progress(progress_value, text=f"🔴 {percent:.1f}%")

    # Кнопка ручной очистки
    if st.button("🧹 Очистить сейчас"):
        auto_cleanup_if_needed(force=True)
        st.rerun()  # перезапускаем, чтобы обновились метрики

    st.markdown("---")
    mode = st.radio(
        "Выберите режим работы:",
        ["📝 Одиночный ввод", "📄 Множественный ввод", "📂 Загрузка из файла"]
    )

# ============================================================
# Получаем модель (или перезагружаем, если была очистка)
# ============================================================
# Перед получением модели проверяем, не пора ли очистить
auto_cleanup_if_needed()
judge = get_judge()

# ============================================================
# Функция безопасного инференса (с проверкой памяти перед каждым вызовом)
# ============================================================
def safe_infer(text):
    """Выполняет инференс, при необходимости очищая память перед вызовом."""
    auto_cleanup_if_needed()
    # Получаем актуальный judge (может измениться после очистки)
    current_judge = get_judge()
    return current_judge.infer(text)

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
            result = safe_infer(text)
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
                    res = safe_infer(txt)
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
                    res = safe_infer(txt)
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
