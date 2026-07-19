from __future__ import annotations

from typing import Any

import pytest

from mcp_server_qdrant.embeddings.openai import (
    OPENAI_CONNECT_TIMEOUT_SECONDS,
    OPENAI_KEEPALIVE_EXPIRY_SECONDS,
    OPENAI_MAX_CONNECTIONS,
    OPENAI_MAX_KEEPALIVE_CONNECTIONS,
    OPENAI_POOL_TIMEOUT_SECONDS,
    OPENAI_READ_TIMEOUT_SECONDS,
    OPENAI_WRITE_TIMEOUT_SECONDS,
    OpenAIProvider,
)


@pytest.mark.asyncio
async def test_openai_client_ignores_proxy_env_and_closes_idempotently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "synthetic-openai-provider-key"
    proxy = "http://proxy-attacker.invalid:8080"
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.setenv(name, proxy)

    provider = OpenAIProvider(
        api_key=secret,
        model_name="text-embedding-3-small",
    )
    client = await provider._get_client()  # noqa: SLF001
    http_client = provider._http_client  # noqa: SLF001

    assert http_client is not None
    assert client._client is http_client  # noqa: SLF001
    assert http_client._trust_env is False  # noqa: SLF001
    assert http_client.follow_redirects is False
    assert http_client.timeout.connect == OPENAI_CONNECT_TIMEOUT_SECONDS
    assert http_client.timeout.read == OPENAI_READ_TIMEOUT_SECONDS
    assert http_client.timeout.write == OPENAI_WRITE_TIMEOUT_SECONDS
    assert http_client.timeout.pool == OPENAI_POOL_TIMEOUT_SECONDS
    pool = http_client._transport._pool  # noqa: SLF001
    assert pool._proxy is None  # noqa: SLF001
    assert pool._max_connections == OPENAI_MAX_CONNECTIONS  # noqa: SLF001
    assert (  # noqa: SLF001
        pool._max_keepalive_connections == OPENAI_MAX_KEEPALIVE_CONNECTIONS
    )
    assert pool._keepalive_expiry == OPENAI_KEEPALIVE_EXPIRY_SECONDS  # noqa: SLF001
    assert secret not in repr(provider)
    assert secret not in str(provider)
    assert secret not in repr(vars(provider))

    await provider.aclose()
    await provider.aclose()

    assert provider.closed is True
    assert http_client.is_closed is True
    assert provider._client is None  # noqa: SLF001
    assert provider._http_client is None  # noqa: SLF001
    assert secret not in repr(vars(provider))


@pytest.mark.asyncio
async def test_openai_provider_sanitizes_network_errors_and_closes_client() -> None:
    secret = "synthetic-provider-key-in-upstream-error"

    class FailingEmbeddings:
        async def create(self, **_kwargs: Any) -> Any:
            raise RuntimeError(f"upstream authorization failed for {secret}")

    class FailingClient:
        def __init__(self) -> None:
            self.embeddings = FailingEmbeddings()
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    provider = OpenAIProvider(
        api_key=secret,
        model_name="text-embedding-3-small",
    )
    failing_client = FailingClient()
    provider._client = failing_client  # type: ignore[assignment]  # noqa: SLF001

    with pytest.raises(RuntimeError) as captured:
        await provider.embed_query("safe query")

    assert str(captured.value) == "OpenAI embedding request failed."
    assert captured.value.__cause__ is None
    assert secret not in str(captured.value)

    await provider.aclose()
    await provider.aclose()
    assert failing_client.close_calls == 1
