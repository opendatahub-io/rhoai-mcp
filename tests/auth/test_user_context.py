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

    def test_token_field_default_is_none(self):
        ctx = UserContext(username="alice")
        assert ctx.token is None

    def test_token_field_with_secret_str(self):
        from pydantic import SecretStr
        ctx = UserContext(username="alice", token=SecretStr("my-secret-token"))
        assert ctx.token is not None
        assert ctx.token.get_secret_value() == "my-secret-token"

    def test_token_repr_does_not_leak_value(self):
        from pydantic import SecretStr
        ctx = UserContext(username="alice", token=SecretStr("my-secret-token"))
        assert "my-secret-token" not in repr(ctx)

    def test_token_str_does_not_leak_value(self):
        from pydantic import SecretStr
        ctx = UserContext(username="alice", token=SecretStr("my-secret-token"))
        assert "my-secret-token" not in str(ctx.token)

    async def test_cross_request_token_isolation(self):
        """Verify contextvars provides per-task isolation for tokens."""
        import asyncio

        from pydantic import SecretStr

        results = {}

        async def task(name: str, token_value: str):
            ctx = UserContext(username=name, token=SecretStr(token_value))
            reset = UserContext.set_current(ctx)
            try:
                await asyncio.sleep(0.01)  # Yield to other tasks
                current = UserContext.current()
                results[name] = current.token.get_secret_value()
            finally:
                UserContext.reset_current(reset)

        await asyncio.gather(
            task("alice", "alice-token"),
            task("bob", "bob-token"),
        )
        assert results["alice"] == "alice-token"
        assert results["bob"] == "bob-token"
