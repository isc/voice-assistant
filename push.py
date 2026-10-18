"""Web Push to the web app's browsers: a timer set from a phone rings there, the app open or not (AD-070).

A message is encrypted for the browser (RFC 8291, aes128gcm) and signed with this server's VAPID key
(RFC 8292), then posted to the browser's push service (Google's, Apple's, Mozilla's), which wakes the
app's service worker. The key and each browser's subscription, by the id it gives itself (AD-061), are
kept in one file, mode 600: a subscription's endpoint is all it takes to send to it."""

import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

import aiohttp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from web_ui import write_private

logger = logging.getLogger(__name__)

# Past this, a timer's message is dropped by the push service: a phone offline for longer hears of it late
# or never, rather than an hour after
TTL = 600
_RECORD_SIZE = 4096


def b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _point(key: ec.EllipticCurvePrivateKey) -> bytes:
    return key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)


def encrypt(
    payload: bytes,
    p256dh: bytes,
    auth: bytes,
    salt: Optional[bytes] = None,
    sender: Optional[ec.EllipticCurvePrivateKey] = None,
) -> bytes:
    """RFC 8291: one aes128gcm record readable with the browser's key (p256dh) and secret (auth)."""
    sender = sender or ec.generate_private_key(ec.SECP256R1())
    salt = salt or os.urandom(16)
    browser = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), p256dh)
    shared = sender.exchange(ec.ECDH(), browser)
    sender_point = _point(sender)
    ikm = HKDF(hashes.SHA256(), 32, auth, b"WebPush: info\x00" + p256dh + sender_point).derive(shared)
    key = HKDF(hashes.SHA256(), 16, salt, b"Content-Encoding: aes128gcm\x00").derive(ikm)
    nonce = HKDF(hashes.SHA256(), 12, salt, b"Content-Encoding: nonce\x00").derive(ikm)
    record = AESGCM(key).encrypt(nonce, payload + b"\x02", None)  # \x02: the last record
    return salt + _RECORD_SIZE.to_bytes(4, "big") + bytes([len(sender_point)]) + sender_point + record


def vapid(key: ec.EllipticCurvePrivateKey, endpoint: str, subject: str) -> str:
    """RFC 8292: the Authorization header proving this server holds the key the browser subscribed with."""
    url = urlsplit(endpoint)
    claims = {"aud": f"{url.scheme}://{url.netloc}", "exp": int(time.time()) + 12 * 3600, "sub": subject}
    signed = f"{b64(json.dumps({'typ': 'JWT', 'alg': 'ES256'}).encode())}.{b64(json.dumps(claims).encode())}"
    r, s = decode_dss_signature(key.sign(signed.encode(), ec.ECDSA(hashes.SHA256())))
    return f"vapid t={signed}.{b64(r.to_bytes(32, 'big') + s.to_bytes(32, 'big'))}, k={b64(_point(key))}"


def _valid(subscription) -> bool:
    """What PushSubscription.toJSON() gives: an https endpoint, a P-256 point and a 16-byte secret."""
    try:
        keys = subscription["keys"]
        return (
            subscription["endpoint"].startswith("https://")
            and len(subscription["endpoint"]) <= 1000
            and len(p256dh := unb64(keys["p256dh"])) == 65
            and p256dh[0] == 4
            and len(unb64(keys["auth"])) == 16
        )
    except (KeyError, TypeError, AttributeError, ValueError):
        return False


class Push:
    """This server's VAPID key and the browsers subscribed, in `path`; created on first use."""

    def __init__(self, path: Path):
        self.path = path
        try:
            data = json.loads(path.read_text())
            self.key = serialization.load_pem_private_key(data["key"].encode(), None)
            # Browser id -> {"endpoint", "keys": {"p256dh", "auth"}, "subject": the page's origin}
            self.subscriptions: dict[str, dict] = data["subscriptions"]
        except Exception as e:
            if not isinstance(e, FileNotFoundError):
                # Browsers subscribed with the lost key subscribe again with the new one at their next visit
                logger.warning(f"Push keys unreadable, starting anew: {e}")
            self.key, self.subscriptions = ec.generate_private_key(ec.SECP256R1()), {}
            self._save()

    @property
    def public_key(self) -> str:
        """For the page's subscribe(): its applicationServerKey."""
        return b64(_point(self.key))

    def _save(self) -> None:
        pem = self.key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )
        try:
            write_private(self.path, json.dumps({"key": pem.decode(), "subscriptions": self.subscriptions}, indent=1))
        except Exception as e:
            logger.warning(f"Failed to save push subscriptions: {e}")

    def subscribe(self, client: str, subscription, origin: Optional[str]) -> bool:
        """A browser's subscription, replacing the one it had. `origin`, the page's: the contact the push
        service gets (Apple's wants an https URL or a mailto)."""
        if not _valid(subscription):
            return False
        subject = origin if origin and origin.startswith("https://") else "mailto:root@localhost"
        entry = {"endpoint": subscription["endpoint"], "keys": subscription["keys"], "subject": subject}
        if self.subscriptions.get(client) != entry:
            self.subscriptions[client] = entry
            self._save()
        return True

    def unsubscribe(self, client: str) -> None:
        if self.subscriptions.pop(client, None):
            self._save()

    async def send(self, client: str, message: dict) -> bool:
        """Push a message (the service worker shows its title and body) to a browser; False if it has no
        subscription or the push service refused. A subscription it says is gone is forgotten."""
        if not (subscription := self.subscriptions.get(client)):
            return False
        keys = subscription["keys"]
        body = encrypt(json.dumps(message, ensure_ascii=False).encode(), unb64(keys["p256dh"]), unb64(keys["auth"]))
        headers = {
            "Authorization": vapid(self.key, subscription["endpoint"], subscription["subject"]),
            "Content-Encoding": "aes128gcm",
            "Content-Type": "application/octet-stream",
            "TTL": str(TTL),
            "Urgency": "high",
        }
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.post(subscription["endpoint"], data=body, headers=headers) as response:
                    if response.status < 300:
                        logger.info(f"Pushed to {client}: {message.get('body')}")
                        return True
                    if response.status in (404, 410):
                        logger.info(f"Push subscription of {client} gone ({response.status}), forgotten")
                        self.unsubscribe(client)
                    else:
                        logger.warning(f"Push to {client} refused: {response.status} {(await response.text())[:200]}")
        except (aiohttp.ClientError, TimeoutError) as e:
            logger.warning(f"Push to {client} failed: {e!r}")
        return False
