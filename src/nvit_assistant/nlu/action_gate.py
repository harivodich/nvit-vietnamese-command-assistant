"""Chặn câu ngoài miền, phủ định và thao tác chưa hỗ trợ trước khi gọi action."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Protocol

from nvit_assistant.schemas import Intent, SlotName


def _fold_for_matching(text: str) -> str:
    """Bỏ dấu cho rule safety để câu ASR/no-diacritics không bị chặn nhầm."""
    decomposed = unicodedata.normalize("NFD", text.casefold())
    stripped = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return unicodedata.normalize("NFC", stripped).replace("đ", "d")


def _compile_folded(pattern: str) -> re.Pattern[str]:
    """Compile regex sau khi fold phần literal, giữ nguyên cú pháp regex."""
    return re.compile(_fold_for_matching(pattern))


UNSUPPORTED_OPERATION_PATTERN = re.compile(
    r"(?<!\w)(?:hủy|huỷ|xóa|xoá|bỏ|dừng|tắt|ngắt|hoãn)\b"
)
UNSUPPORTED_OPERATION_FOLDED_PATTERN = re.compile(
    r"(?<!\w)(?:huy|xoa|hoan)\b|"
    r"(?<!\w)(?:bo|dung|tat|ngat)\s+"
    r"(?:nhac|phat|bai|cuoc goi|goi|bao thuc|chuong|loi nhac|hoi|xem|du bao|thoi tiet)\b"
)
NEGATED_COMMAND_PATTERN = re.compile(
    r"(?<!\w)(?:đừng|chớ|khỏi)\s+(?!quên\s+(?:nhắc|báo|gọi))|"
    r"(?<!\w)(?:không|chưa)\s+(?:cần\s+)?"
    r"(?:gọi|mở|phát|chơi|nghe|đặt|cài|nhắc|hỏi|xem|báo)|"
    r"(?<!\w)tôi\s+không\s+muốn\b"
)
NEGATED_COMMAND_FOLDED_PATTERN = re.compile(
    r"(?<!\w)(?:dung|khoi)\s+(?!quen\s+(?:nhac|bao|goi))"
    r"(?:goi|mo|phat|choi|nghe|dat|cai|nhac|hoi|xem|bao)\b|"
    r"(?<!\w)(?:khong|chua)\s+(?:can\s+)?"
    r"(?:goi|mo|phat|choi|nghe|dat|cai|nhac|hoi|xem|bao)\b|"
    r"(?<!\w)(?:khong|chua)\s+can\s+"
    r"(?:thoi tiet|du bao|bao thuc|chuong|loi nhac|nhac)\b|"
    r"(?<!\w)(?:toi\s+)?khong\s+muon\s+"
    r"(?:thoi tiet|du bao|bao thuc|chuong|loi nhac|nhac|am nhac|bai hat)\b|"
    r"(?<!\w)toi\s+khong\s+muon\b"
)
POSITIVE_DONT_FORGET_FOLDED_PATTERN = re.compile(
    r"(?<!\w)dung\s+quen\s+(?:nhac|bao|goi)\b"
)
OUT_OF_SCOPE_PATTERN = re.compile(r"(?<!\w)gọi\s+món\b")
OUT_OF_SCOPE_FOLDED_PATTERN = re.compile(r"(?<!\w)goi\s+mon\b")
NON_ACTION_STATEMENT_FOLDED_PATTERN = re.compile(
    r"^(?:toi|minh|me|bo|ba|ma|co ay|anh ay|em ay|ho|ban)\s+"
    r"(?:dang|da|vua|thich|yeu|ghet)\b|"
    r"^(?:toi|minh|me|bo|ba|ma|co ay|anh ay|em ay|ho|ban)\s+"
    r"(?:uong|mua|goi|nop|gui|tuoi|hop|kiem tra|thanh toan|tra|don|tap|nghe|mo|phat)\b|"
    r"^(?:hom qua|toi qua|sang qua|chieu qua)\b.*(?<!\w)(?:troi|thoi tiet)\b.*"
    r"(?<!\w)(?:mua|nang|lanh|nong|gio|dep)\s*[.!?]*$|"
    r"^(?:day|do)\s+la\b|(?:roi|qua)\s*[.!?]*$|(?<!\w)chuong\s+cua\b"
    r"|^(?:hom nay\s+)?troi\s+(?:dang\s+)?"
    r"(?:dep|mua|nang|lanh|nong|nhieu gio|co gio)\s*[.!?]*$"
    r"|^(?:toi|minh|me|bo|ba|ma|co ay|anh ay|em ay|ho|ban)\s+"
    r"(?:nghe|mo|phat)\s+(?:nhac|am nhac|bai hat)"
    r"(?:\s+(?:moi ngay|hang ngay|thuong xuyen))?\s*[.!?]*$"
    r"|^(?:nhac|am nhac|bai hat(?: nay)?)\s+(?:dang\s+phat|hay)\s*[.!]*$"
    r"|^nhiet do\s+la\s+(?:\d+|am\s+\d+)\s+do\s*[.!]*$"
    r"|^.{2,40}\s+dang\s+(?:mua|nang|lanh|nong)\s*[.!]*$"
    r"|^(?:hom nay|ngay mai|mai|toi nay)\s+co\s+cuoc hop\s*[.!]*$"
)
SIDE_EFFECT_INTENTS = frozenset(
    {
        Intent.SET_REMINDER,
        Intent.SET_ALARM,
        Intent.PLAY_MUSIC,
        Intent.CALL_CONTACT,
    }
)
ACTION_VERB_FOLDED = (
    r"(?:goi|lien he|quay so|noi may|mo|bat|phat|choi|nghe|"
    r"dat|cai|nhac|tao loi nhac|danh thuc)"
)
NON_COMMAND_QUESTION_FOLDED_PATTERN = re.compile(
    rf"^(?:da\s+)?{ACTION_VERB_FOLDED}(?:\s+.+?)?\s+chua\s*[?!.]*$|"
    rf"^co\s+(?:can\s+|nen\s+)?{ACTION_VERB_FOLDED}"
    rf"(?:\s+.+?)?\s+khong\s*[?!.]*$|"
    rf"^{ACTION_VERB_FOLDED}(?:\s+.+?)?\s+phai\s+khong\s*[?!.]*$|"
    r"^goi(?:\s+.+?)?\s+hay\s+.+\s*[?!.]*$|"
    r"^(?:nhac|am nhac|bai hat|bai)(?:\s+\w+){0,4}\s+"
    r"(?:dang\s+phat(?:\s+(?:a|ha|u|vay))?|ten\s+gi)\s*[?!.]*$|"
    r"^(?:nhac|am nhac|bai hat|bai)(?:\s+\w+){0,4}\s+"
    r"(?:co\s+dang\s+(?:phat|choi)\s+khong|cua\s+(?:ai|nghe si nao))\s*[?!.]*$|"
    r"^dang\s+phat\s+(?:nhac|am nhac|bai hat)"
    r"(?:\s+(?:a|ha|u|vay))?\s*[?!.]*$"
)

INTENT_CUES = {
    Intent.SET_REMINDER: _compile_folded(
        r"(?<!\w)(?:nhắc|lời nhắc|nhớ báo|đừng quên|gọi|uống|kiểm tra|họp|"
        r"nộp|đón|mua|gửi|tập|tưới|thanh toán|trả|mang|đi làm)\b"
    ),
    Intent.SET_ALARM: _compile_folded(
        r"(?<!\w)(?:báo thức|đánh thức|(?:gọi|kêu)"
        r"(?:\s+\w+){0,3}\s+dậy|"
        r"thức dậy|chuông)\b"
    ),
    Intent.ASK_WEATHER: _compile_folded(
        r"(?<!\w)(?:thời tiết|dự báo|nhiệt độ|trời|mưa|nắng|lạnh|nóng|tuyết|gió|"
        r"áo khoác|áo mưa|găng tay|mang theo ô|mặc đồ|mặc áo|quần áo)\b"
    ),
    Intent.PLAY_MUSIC: _compile_folded(
        r"(?<!\w)(?:nhạc|âm nhạc|bài hát|bài|ca khúc|album|danh sách phát|playlist|"
        r"nghe|phát|rap|jazz)\b"
    ),
    Intent.CALL_CONTACT: _compile_folded(
        r"(?<!\w)(?:gọi|liên hệ|quay số|nối máy|điện thoại)\b"
    ),
}


@dataclass(frozen=True)
class ActionGateDecision:
    """Kết quả gate có lý do để API/report giải thích được vì sao bị chặn."""

    allowed: bool
    reason: str


class ActionGate(Protocol):
    """Interface nhỏ để thay rule gate bằng OOD model trong tương lai."""

    def check(self, text: str, intent: Intent, slots: dict[str, Any]) -> ActionGateDecision:
        """Kiểm tra câu có đủ dấu hiệu là command được hỗ trợ hay không."""
        ...


class CommandActionGate:
    """Rule gate precision-first cho đúng năm command domain của challenge."""

    def check(self, text: str, intent: Intent, slots: dict[str, Any]) -> ActionGateDecision:
        """Ưu tiên chặn phủ định/cancel, sau đó yêu cầu cue phù hợp intent."""
        folded_text = _fold_for_matching(text)
        if NON_ACTION_STATEMENT_FOLDED_PATTERN.search(folded_text):
            return ActionGateDecision(False, "non_action_statement")
        if (
            intent in SIDE_EFFECT_INTENTS
            and NON_COMMAND_QUESTION_FOLDED_PATTERN.search(folded_text)
        ):
            return ActionGateDecision(False, "status_or_choice_question")
        positive_dont_forget = POSITIVE_DONT_FORGET_FOLDED_PATTERN.search(folded_text)
        if (
            NEGATED_COMMAND_PATTERN.search(text)
            or NEGATED_COMMAND_FOLDED_PATTERN.search(folded_text)
        ) and not positive_dont_forget:
            return ActionGateDecision(False, "negated_command")
        unsupported = UNSUPPORTED_OPERATION_PATTERN.search(
            text
        ) or UNSUPPORTED_OPERATION_FOLDED_PATTERN.search(folded_text)
        if unsupported and not positive_dont_forget:
            return ActionGateDecision(False, "unsupported_operation")
        if OUT_OF_SCOPE_PATTERN.search(text) or OUT_OF_SCOPE_FOLDED_PATTERN.search(folded_text):
            return ActionGateDecision(False, "out_of_scope")
        cue = INTENT_CUES.get(intent)
        if cue is not None and cue.search(folded_text):
            return ActionGateDecision(True, "intent_cue")
        if intent is Intent.PLAY_MUSIC and any(
            slot_name in slots for slot_name in (SlotName.SONG.value, SlotName.ARTIST.value)
        ):
            return ActionGateDecision(True, "music_entity")
        return ActionGateDecision(False, "missing_intent_cue")
