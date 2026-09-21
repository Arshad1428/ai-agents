import tools.search as search_module
from agent.config import _keep_alive
import server.app as server_app


class FakeDDGS:
    def __init__(self, timeout=None):
        self.timeout = timeout

    def text(self, query, max_results=5):
        return [
            {"title": f"T{i}", "body": "x" * 1000, "href": f"http://e/{i}"}
            for i in range(10)
        ][:max_results]


def test_search_limits_payload(monkeypatch):
    monkeypatch.setattr(search_module, "DDGS", FakeDDGS)
    out = search_module.web_search.invoke({"query": "anything"})
    assert out.count("Result ") == search_module.MAX_RESULTS
    # every snippet is truncated
    assert "x" * (search_module.MAX_BODY_CHARS + 1) not in out


def test_search_errors_are_returned_as_text(monkeypatch):
    class Broken:
        def __init__(self, timeout=None): pass
        def text(self, *a, **k): raise RuntimeError("network down")

    monkeypatch.setattr(search_module, "DDGS", Broken)
    assert "network down" in search_module.web_search.invoke({"query": "q"})


def test_keep_alive_parsing():
    assert _keep_alive("30m") == "30m"
    assert _keep_alive("-1") == -1
    assert _keep_alive("600") == 600


def test_warm_up_never_raises(monkeypatch):
    class Down:
        def invoke(self, *_):
            raise ConnectionError("ollama not running")

    monkeypatch.setattr(server_app, "plain_llm", Down())
    server_app.warm_up()  # must not raise
