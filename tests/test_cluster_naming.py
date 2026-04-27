from pathlib import Path
from types import SimpleNamespace

from sciscape.clustering import cluster_naming as cn
from sciscape.clustering.core_documents import ClusterDocument


class TestClusterNamingSettings:

    def test_gemini_api_key_promotes_gemini_defaults(self, monkeypatch):
        monkeypatch.setattr(cn, "_discover_env_overlay", lambda: {"GEMINI_API_KEY": "test-gemini-key"})
        monkeypatch.delenv(cn.ENV_BASE_URL, raising=False)
        monkeypatch.delenv(cn.ENV_API_KEY, raising=False)
        monkeypatch.delenv(cn.ENV_MODEL, raising=False)
        monkeypatch.delenv(cn._LEGACY_ENV_BASE_URL, raising=False)
        monkeypatch.delenv(cn._LEGACY_ENV_API_KEY, raising=False)
        monkeypatch.delenv(cn._LEGACY_ENV_MODEL, raising=False)

        settings = cn._determine_settings(base_url=None, api_key=None, model=None)

        assert settings["base_url"] == cn.GEMINI_OPENAI_BASE_URL
        assert settings["api_key"] == "test-gemini-key"
        assert settings["model"] == cn.GEMINI_DEFAULT_MODEL

    def test_explicit_sciscape_settings_beat_gemini_fallback(self, monkeypatch):
        monkeypatch.setattr(
            cn,
            "_discover_env_overlay",
            lambda: {
                "GEMINI_API_KEY": "test-gemini-key",
                "SCISCAPE_LLM_BASE_URL": "http://example.test/v1",
                "SCISCAPE_LLM_API_KEY": "example-key",
                "SCISCAPE_LLM_MODEL": "custom-model",
            },
        )

        settings = cn._determine_settings(base_url=None, api_key=None, model=None)

        assert settings["base_url"] == "http://example.test/v1"
        assert settings["api_key"] == "example-key"
        assert settings["model"] == "custom-model"


class TestClusterNamingEnvDiscovery:

    def test_discovers_workspace_env_when_repo_env_missing(self, monkeypatch, tmp_path):
        workspace = tmp_path / "workspace"
        repo_root = workspace / "1.4.4.Sciscape"
        module_path = repo_root / "sciscape" / "clustering" / "cluster_naming.py"
        module_path.parent.mkdir(parents=True, exist_ok=True)
        module_path.write_text("# stub\n", encoding="utf-8")
        (workspace / ".env").write_text("GEMINI_API_KEY=workspace-key\n", encoding="utf-8")
        monkeypatch.chdir(repo_root)
        monkeypatch.setattr(cn, "__file__", str(module_path))
        monkeypatch.delenv(cn.ENV_CONFIG_PATH, raising=False)

        overlay = cn._discover_env_overlay()

        assert overlay["GEMINI_API_KEY"] == "workspace-key"

    def test_repo_env_beats_workspace_env(self, monkeypatch, tmp_path):
        workspace = tmp_path / "workspace"
        repo_root = workspace / "1.4.4.Sciscape"
        module_path = repo_root / "sciscape" / "clustering" / "cluster_naming.py"
        module_path.parent.mkdir(parents=True, exist_ok=True)
        module_path.write_text("# stub\n", encoding="utf-8")
        (workspace / ".env").write_text("GEMINI_API_KEY=workspace-key\n", encoding="utf-8")
        (repo_root / ".env").write_text("GEMINI_API_KEY=repo-key\n", encoding="utf-8")
        monkeypatch.chdir(repo_root)
        monkeypatch.setattr(cn, "__file__", str(module_path))
        monkeypatch.delenv(cn.ENV_CONFIG_PATH, raising=False)

        overlay = cn._discover_env_overlay()

        assert overlay["GEMINI_API_KEY"] == "repo-key"


class TestSummariseClusterModelSelection:

    def test_summarise_cluster_uses_client_default_model_when_unspecified(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            cn,
            "_prepare_document_text",
            lambda client, document, model: {"uid": document.uid, "lang": "en", "text": document.title},
        )

        def fake_call(client, *, model, system_prompt, user_content):
            captured["model"] = model
            return (
                '{"name":"Example","description":"Desc","keywords":["k"],'
                '"sources":["W1"],"notes":""}'
            )

        monkeypatch.setattr(cn, "_call_chat_completion", fake_call)
        client = SimpleNamespace(_leiden_module_model="gemini-2.5-pro")

        summary = cn.summarise_cluster(
            client,
            "cluster-1",
            [ClusterDocument(uid="W1", title="Doc", abstract="Abstract")],
        )

        assert captured["model"] == "gemini-2.5-pro"
        assert summary.name == "Example"

    def test_summarise_cluster_explicit_model_overrides_client_default(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            cn,
            "_prepare_document_text",
            lambda client, document, model: {"uid": document.uid, "lang": "en", "text": document.title},
        )

        def fake_call(client, *, model, system_prompt, user_content):
            captured["model"] = model
            return (
                '{"name":"Example","description":"Desc","keywords":["k"],'
                '"sources":["W1"],"notes":""}'
            )

        monkeypatch.setattr(cn, "_call_chat_completion", fake_call)
        client = SimpleNamespace(_leiden_module_model="gemini-2.5-pro")

        cn.summarise_cluster(
            client,
            "cluster-1",
            [ClusterDocument(uid="W1", title="Doc", abstract="Abstract")],
            model="gpt-4o-mini",
        )

        assert captured["model"] == "gpt-4o-mini"
