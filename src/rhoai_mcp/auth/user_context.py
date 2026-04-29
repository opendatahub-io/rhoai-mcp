"""User context management with contextvars for per-session identity."""

import contextvars
from dataclasses import dataclass, field

_current_user: contextvars.ContextVar["UserContext | None"] = contextvars.ContextVar(
    "rhoai_current_user", default=None
)


@dataclass
class UserContext:
    """User identity context for per-session management."""

    username: str
    groups: list[str] = field(default_factory=list)
    uid: str | None = None

    @classmethod
    def current(cls) -> "UserContext | None":
        """Get the current user context.

        Returns:
            The current UserContext or None if not set.
        """
        return _current_user.get()

    @classmethod
    def set_current(cls, ctx: "UserContext") -> contextvars.Token["UserContext | None"]:
        """Set the current user context.

        Args:
            ctx: The UserContext to set as current.

        Returns:
            A token that can be used to reset the context later.
        """
        return _current_user.set(ctx)

    @classmethod
    def reset_current(cls, token: contextvars.Token["UserContext | None"]) -> None:
        """Reset the current user context using a token.

        Args:
            token: The token returned by set_current.
        """
        _current_user.reset(token)
