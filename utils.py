# ============================================================
# utils.py — общие константы, класс RQAJudge, загрузчики
# ============================================================

import os
import json
import csv
import torch
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModel

# ============================================================
# Константы
# ============================================================

ERROR_TYPES = [
    "false_causality",
    "unsupported_claim",
    "overgeneralization",
    "missing_premise",
    "contradiction",
    "circular_reasoning",
]

ERROR_NAMES_RU = {
    "false_causality": "Ложная причинно-следственная связь",
    "unsupported_claim": "Неподкреплённое утверждение",
    "overgeneralization": "Чрезмерное обобщение",
    "missing_premise": "Отсутствующая предпосылка",
    "contradiction": "Противоречие",
    "circular_reasoning": "Круговое рассуждение",
}

ERROR_THRESHOLDS = {
    "false_causality": 0.55,
    "unsupported_claim": 0.55,
    "overgeneralization": 0.60,
    "missing_premise": 0.80,
    "contradiction": 0.60,
    "circular_reasoning": 0.60,
}

CONFIDENCE_HIGH = 0.85
CONFIDENCE_MEDIUM = 0.65

# ============================================================
# Класс RQAJudge (полностью из вашего кода)
# ============================================================

class RQAJudge:
    def __init__(self, model_name="skatzR/RQA-X1.1", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()
        cfg = self.model.config
        self.temp_issue = float(cfg.temperature_has_issue)
        self.temp_errors = list(cfg.temperature_errors)

    @torch.no_grad()
    def infer(self, text: str, issue_threshold: float = 0.6, disagreement_threshold: float = 0.4):
        inputs = self.tokenizer(text, truncation=True, max_length=512,
                                padding="max_length", return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        # has_issue
        issue_logit = outputs["has_issue_logits"] / self.temp_issue
        issue_prob = torch.sigmoid(issue_logit).item()
        has_issue = issue_prob >= issue_threshold

        # ошибки (сырые вероятности)
        raw_error_logits = outputs["errors_logits"][0]
        raw_probs = {}
        for i, logit in enumerate(raw_error_logits):
            calibrated = logit / self.temp_errors[i]
            prob = torch.sigmoid(calibrated).item()
            raw_probs[ERROR_TYPES[i]] = prob

        # disagreement
        p_any_error_raw = 1.0
        for p in raw_probs.values():
            p_any_error_raw *= (1.0 - p)
        p_any_error_raw = 1.0 - p_any_error_raw
        disagreement = abs(issue_prob - p_any_error_raw)

        # HARD-GATING
        error_probs = raw_probs.copy() if has_issue else {k: 0.0 for k in raw_probs}

        # явные ошибки (кроме missing_premise)
        explicit_errors = []
        for err, prob in error_probs.items():
            if prob >= ERROR_THRESHOLDS[err] and err != "missing_premise":
                explicit_errors.append((err, prob))
        explicit_errors.sort(key=lambda x: x[1], reverse=True)

        # hidden_problem
        hidden_problem = has_issue and not explicit_errors and issue_prob >= 0.6

        # borderline
        borderline = not has_issue and hidden_problem and disagreement >= disagreement_threshold

        # confidence bands
        if issue_prob >= CONFIDENCE_HIGH:
            confidence = "ВЫСОКАЯ"
        elif issue_prob >= CONFIDENCE_MEDIUM:
            confidence = "СРЕДНЯЯ"
        else:
            confidence = "НИЗКАЯ"

        # топ‑2 ошибки (для информации)
        sorted_all = sorted(error_probs.items(), key=lambda x: x[1], reverse=True)
        top_errors = []
        for err, prob in sorted_all[:2]:
            top_errors.append({
                "type": err,
                "probability": prob,
                "above_threshold": prob >= ERROR_THRESHOLDS[err]
            })

        return {
            "text": text,
            "has_issue": has_issue,
            "issue_probability": issue_prob,
            "confidence": confidence,
            "explicit_errors": explicit_errors,
            "hidden_problem": hidden_problem,
            "borderline": borderline,
            "disagreement": disagreement,
            "top_errors": top_errors,
            "raw_probs": raw_probs
        }

# ============================================================
# Загрузчики текстов из файлов (адаптированы для работы с байтовыми потоками)
# ============================================================

def load_texts_from_uploaded_file(uploaded_file) -> List[str]:
    """
    Загружает тексты из загруженного файла (объект BytesIO).
    Поддерживает .txt, .csv, .json.
    """
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    content = uploaded_file.read().decode("utf-8")

    if ext == ".txt":
        return [line.strip() for line in content.splitlines() if line.strip()]

    if ext == ".csv":
        import csv
        from io import StringIO
        reader = csv.DictReader(StringIO(content))
        return [row["text"] for row in reader]

    if ext == ".json":
        data = json.loads(content)
        if isinstance(data, list):
            return data
        else:
            raise ValueError("JSON должен содержать список строк")

    raise ValueError("Неподдерживаемый формат файла")

# ============================================================
# Функция для форматирования результата в HTML/Markdown для Streamlit
# ============================================================

def format_result_for_streamlit(r: Dict[str, Any]) -> str:
    lines = []
    lines.append("### 📄 Текст")
    lines.append(f">{r['text']}")

    prob_percent = r['issue_probability'] * 100
    status = "✅ Проблем НЕ обнаружено" if not r['has_issue'] else "❌ Проблема обнаружена"
    lines.append(f"\n**{status}**  \nВероятность: {prob_percent:.2f}% — уверенность: {r['confidence']}")

    if r["borderline"]:
        lines.append("⚠️ **Пограничный случай**: аргументативный текст")
    if r["hidden_problem"]:
        lines.append("🟡 **Скрытая проблема**: возможны неявные предпосылки")

    if r["explicit_errors"]:
        lines.append("\n**❌ Явные логические ошибки:**")
        for name, prob in r["explicit_errors"]:
            lines.append(f"- {ERROR_NAMES_RU[name]} — {prob*100:.2f}%")

    below = [e for e in r["top_errors"] if not e["above_threshold"] and e["probability"] > 0.01]
    if below:
        lines.append("\n**📉 Дополнительные ошибки (ниже порога):**")
        for e in below:
            name_ru = ERROR_NAMES_RU.get(e["type"], e["type"])
            lines.append(f"- {name_ru} — {e['probability']*100:.2f}% (порог {ERROR_THRESHOLDS[e['type']]*100:.0f}%)")

    lines.append(f"\n**📊 Disagreement:** {r['disagreement']:.3f}")
    return "\n".join(lines)
