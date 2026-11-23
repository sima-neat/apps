"""Verified HTTPS trust for Studio's local backend and explicitly configured CAs."""
import os
from pathlib import Path
import ssl
from urllib.parse import urlsplit


def backend_ssl_context(config_path, settings, endpoint):
    """Trust Studio's local certificate or an explicit CA, without disabling verification."""
    if urlsplit(endpoint).scheme != "https":
        return None
    context = ssl.create_default_context()
    certificate = settings.get("tls_ca_file")
    if certificate:
        certificate = Path(certificate).expanduser()
        if not certificate.is_absolute():
            certificate = config_path.parent / certificate
    elif not settings.get("studio_url"):
        # Studio creates .local-certs beside its source/installed backend assets.
        # Config may be in src/common (checkout) or common (colcon installation).
        certificate = os.environ.get("NEAT_TLS_CERT")
        if certificate:
            certificate = Path(certificate).expanduser()
        else:
            candidates = (*config_path.parents, Path(__file__).resolve().parents[2])
            root = next((p for p in candidates
                         if (p / "src/python/api/studio_api.py").is_file()), None)
            if root is None:
                from ament_index_python.packages import get_package_share_directory
                root = Path(get_package_share_directory("neat_genai_studio"))
            certificate = root / ".local-certs/server.crt"
    if certificate:
        certificate = Path(certificate)
        if not certificate.is_file():
            raise ValueError(f"Studio HTTPS certificate not found: {certificate}; "
                             "start the local backend or set ros.tls_ca_file")
        context.load_verify_locations(cafile=str(certificate))
        # An explicitly trusted certificate may be a leaf or an intermediate CA.
        context.verify_flags |= ssl.VERIFY_X509_PARTIAL_CHAIN
    return context
