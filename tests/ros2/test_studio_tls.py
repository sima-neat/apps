"""Real HTTPS trust checks; no ROS, model runtime or external network required."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
from pathlib import Path
import ssl
import subprocess
import threading
from urllib.error import URLError
from urllib.request import urlopen

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / 'examples/ros2/genai/neat-genai-studio/src/python/studio_tls.py'
spec = importlib.util.spec_from_file_location('studio_tls_test', ENTRY)
studio_tls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(studio_tls)


def make_certificate(directory, host='127.0.0.1'):
    directory.mkdir(parents=True, exist_ok=True)
    cert, key = directory / 'server.crt', directory / 'server.key'
    subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-sha256',
        '-nodes', '-days', '1', '-keyout', str(key), '-out', str(cert),
        '-subj', '/CN=localhost', '-addext', f'subjectAltName=IP:{host},DNS:localhost'],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return cert, key


@pytest.mark.parametrize('mode', ['local', 'installed', 'ca', 'env', 'untrusted',
                                  'wrong_ca', 'wrong_host', 'missing_ca'])
def test_verified_https_request(tmp_path, monkeypatch, mode):
    monkeypatch.delenv('NEAT_TLS_CERT', raising=False)
    root = tmp_path / 'backend'
    api = root / 'src/python/api/studio_api.py'
    api.parent.mkdir(parents=True)
    api.touch()
    config = root / ('common/config.yaml' if mode == 'installed' else 'src/common/config.yaml')
    config.parent.mkdir(parents=True, exist_ok=True)
    cert, key = make_certificate(root / '.local-certs',
                                '127.0.0.2' if mode == 'wrong_host' else '127.0.0.1')
    seen = []
    class Backend(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def do_GET(self):
            seen.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'HTTPS reached the backend')
    server = ThreadingHTTPServer(('127.0.0.1', 0), Backend)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(str(cert), str(key))
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    endpoint = f'https://127.0.0.1:{server.server_port}'
    settings = {}
    try:
        if mode == 'env':
            custom = tmp_path / 'configured.crt'
            cert.rename(custom)
            monkeypatch.setenv('NEAT_TLS_CERT', str(custom))
        elif mode in ('ca', 'untrusted', 'wrong_ca', 'missing_ca'):
            settings['studio_url'] = endpoint
            if mode == 'ca': settings['tls_ca_file'] = '../../.local-certs/server.crt'
            elif mode == 'wrong_ca': settings['tls_ca_file'] = str(make_certificate(tmp_path / 'other')[0])
            elif mode == 'missing_ca': settings['tls_ca_file'] = 'missing.pem'
        if mode == 'missing_ca':
            with pytest.raises(ValueError, match='certificate not found'):
                studio_tls.backend_ssl_context(config, settings, endpoint)
            return
        trusted = studio_tls.backend_ssl_context(config, settings, endpoint)
        assert trusted.verify_mode == ssl.CERT_REQUIRED and trusted.check_hostname
        if mode in ('untrusted', 'wrong_ca', 'wrong_host'):
            with pytest.raises(URLError) as exc:
                urlopen(endpoint, context=trusted, timeout=3)
            assert isinstance(exc.value.reason, ssl.SSLCertVerificationError)
            assert not seen
        else:
            with urlopen(endpoint, context=trusted, timeout=3) as response:
                assert response.read() == b'HTTPS reached the backend'
            assert seen == ['/']
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_http_does_not_require_a_certificate(tmp_path):
    assert studio_tls.backend_ssl_context(tmp_path / 'config.yaml',
        {'tls_ca_file': 'nonexistent.pem'}, 'http://127.0.0.1:5000') is None
