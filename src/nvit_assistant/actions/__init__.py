"""Action adapter: core NLU chỉ gọi interface, challenge dùng implementation giả lập."""

from nvit_assistant.actions.router import ActionExecution, ActionRouter, MockActionRouter

__all__ = ["ActionExecution", "ActionRouter", "MockActionRouter"]
