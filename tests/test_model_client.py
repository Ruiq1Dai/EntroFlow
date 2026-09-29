from entroflow.model_client import _is_loopback_url


def test_loopback_model_urls_bypass_server_proxy():
    assert _is_loopback_url("http://127.0.0.1:8000/v1")
    assert _is_loopback_url("http://localhost:8000/v1")
    assert _is_loopback_url("http://[::1]:8000/v1")


def test_remote_model_urls_keep_normal_proxy_behavior():
    assert not _is_loopback_url("https://api.example.com/v1")
