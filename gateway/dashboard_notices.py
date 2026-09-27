"""Friendly presentation of native transport notices; ordinary replies pass through."""


def dashboard_notice(text):
    if text.startswith("📬 No home channel is set for Api_Server."):
        return "", "skip"
    if text.startswith("🗜️ Compacting context"):
        return "أرتّب سياق المحادثة حتى أستطيع المتابعة…", "activity"
    if text.startswith("⚠️ Request payload too large (413)") or text.startswith("⚠️  Request payload too large (413)"):
        return "حجم الطلب أكبر من حد المزوّد؛ أحاول تقليل السياق…", "activity"
    return text, "message"
