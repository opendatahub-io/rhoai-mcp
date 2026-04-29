"""Tests for UserContext with contextvars."""

from rhoai_mcp.auth.user_context import UserContext


class TestUserContext:
    def test_create_user_context(self):
        ctx = UserContext(username="alice", groups=["team-a"])
        assert ctx.username == "alice"
        assert ctx.groups == ["team-a"]
        assert ctx.uid is None

    def test_current_returns_none_when_unset(self):
        assert UserContext.current() is None

    def test_set_and_get_current(self):
        ctx = UserContext(username="alice", groups=["team-a"])
        token = UserContext.set_current(ctx)
        try:
            assert UserContext.current() is ctx
            assert UserContext.current().username == "alice"
        finally:
            UserContext.reset_current(token)

    def test_reset_clears_context(self):
        ctx = UserContext(username="alice", groups=["team-a"])
        token = UserContext.set_current(ctx)
        UserContext.reset_current(token)
        assert UserContext.current() is None

    def test_nested_contexts(self):
        ctx1 = UserContext(username="alice", groups=["team-a"])
        ctx2 = UserContext(username="bob", groups=["team-b"])
        token1 = UserContext.set_current(ctx1)
        assert UserContext.current().username == "alice"
        token2 = UserContext.set_current(ctx2)
        assert UserContext.current().username == "bob"
        UserContext.reset_current(token2)
        assert UserContext.current().username == "alice"
        UserContext.reset_current(token1)
        assert UserContext.current() is None
