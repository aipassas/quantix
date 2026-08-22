"""Password hashing and policy. No Streamlit, no storage — just the crypto.

SCRYPT FROM THE STANDARD LIBRARY, deliberately. bcrypt and argon2 are the
usual choices and neither is installed here; adding a compiled dependency
to an app that ships by `pip install -r requirements.txt` on someone's
laptop is a real cost. hashlib.scrypt is memory-hard, is in the stdlib on
every Python this app supports, and at the parameters below costs ~140ms
and 64 MiB per attempt — which is the point. A fast hash is the whole
vulnerability.

PARAMETERS TRAVEL WITH THE HASH. Each stored value is
`scrypt$n$r$p$salt$key`, so raising the work factor later does not
invalidate every existing password: verify still reads the old cost from
the record, and needs_rehash() tells the caller to re-store the hash with
current parameters next time the user successfully signs in. Hard-coding
the cost at verify time is how an upgrade turns into a mass lockout.

EVERY COMPARISON IS CONSTANT-TIME. hmac.compare_digest, never ==. A
byte-by-byte comparison leaks how much of the hash matched, which is
enough to reconstruct it given enough attempts.

VERIFY_DUMMY EXISTS TO STOP USER ENUMERATION. Signing in with an unknown
address must take as long as signing in with a known one; otherwise the
response time answers "does this person have an account here?" for any
address someone cares to try. The caller runs verify_dummy() on the
no-such-user path.

NOTHING HERE LOGS. Not the password, not the hash, not the email. There
is no logger in this module on purpose — an exception with a password in
its frame is exactly what ends up in a bug report.
"""
import base64
import hashlib
import hmac
import os
import re
import secrets
import unicodedata
from typing import List, Optional, Tuple

# ~140ms and 64 MiB per attempt on a 2020-era laptop. The memory cost is
# the part that matters: it is what stops an attacker trading silicon for
# speed on stolen hashes.
SCRYPT_N = 2 ** 16
SCRYPT_R = 8
SCRYPT_P = 1
_DKLEN = 32
_SALT_BYTES = 16

# scrypt needs an explicit ceiling above n*r*128, or it refuses to run.
_MAXMEM = 256 * 1024 * 1024

_ALGORITHM = "scrypt"

MIN_LENGTH = 10
MAX_LENGTH = 1024          # a bound, so a huge body can't be a cheap DoS

# Deliberately short and about shape, not a banned-words list. A blocklist
# of "common passwords" that fits in a source file is theatre; length and
# variety are what actually help, and anything stronger belongs behind a
# breached-password API this app has no business calling.
_COMMON = frozenset("""
password password1 password123 qwerty qwerty123 123456 12345678 123456789
1234567890 letmein welcome admin admin123 iloveyou abc123 monkey dragon
football baseball trustno1 changeme passw0rd quantix quantix123
""".split())


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def normalise(password: str) -> bytes:
    """NFKC-normalise, then encode.

    Without this, a password typed with a composed accent on one keyboard
    and a decomposed one on another is a different byte string and simply
    fails to verify, with nothing in the UI to explain why.
    """
    return unicodedata.normalize("NFKC", password or "").encode("utf-8")


def hash_password(password: str,
                  n: int = SCRYPT_N, r: int = SCRYPT_R, p: int = SCRYPT_P) -> str:
    """A self-describing hash record for `password`."""
    salt = os.urandom(_SALT_BYTES)
    key = hashlib.scrypt(normalise(password), salt=salt, n=n, r=r, p=p,
                         dklen=_DKLEN, maxmem=_MAXMEM)
    return f"{_ALGORITHM}${n}${r}${p}${_b64(salt)}${_b64(key)}"


def _parse(encoded: str) -> Optional[Tuple[int, int, int, bytes, bytes]]:
    try:
        algorithm, n, r, p, salt, key = (encoded or "").split("$")
        if algorithm != _ALGORITHM:
            return None
        return int(n), int(r), int(p), _unb64(salt), _unb64(key)
    except Exception:
        # A malformed record is a failed verification, never a crash — a
        # corrupt line in the store must not take the login page down.
        return None


def verify_password(password: str, encoded: str) -> bool:
    """Constant-time check of `password` against a stored record."""
    parsed = _parse(encoded)
    if parsed is None:
        return False
    n, r, p, salt, expected = parsed
    try:
        actual = hashlib.scrypt(normalise(password), salt=salt, n=n, r=r, p=p,
                                dklen=len(expected), maxmem=_MAXMEM)
    except Exception:
        return False
    return hmac.compare_digest(actual, expected)


# A real record over a random secret. Verifying against this costs the
# same as verifying against a real user's, which is the entire point.
_DUMMY = hash_password(secrets.token_urlsafe(32))


def verify_dummy() -> bool:
    """Burn one verification's worth of time. Always False.

    Called on the no-such-account path so that a wrong address and a wrong
    password are indistinguishable by how long the answer took.
    """
    verify_password(secrets.token_urlsafe(16), _DUMMY)
    return False


def needs_rehash(encoded: str,
                 n: int = SCRYPT_N, r: int = SCRYPT_R, p: int = SCRYPT_P) -> bool:
    """Whether this record was made with weaker parameters than current.

    Only ever upgrades: a record already at or above the current cost is
    left alone, so lowering the constants in an emergency does not rewrite
    every strong hash down to the weaker setting.
    """
    parsed = _parse(encoded)
    if parsed is None:
        return True
    have_n, have_r, have_p, _, _ = parsed
    return (have_n, have_r, have_p) < (n, r, p)


def strength_problems(password: str, email: str = "", name: str = "") -> List[str]:
    """Everything wrong with `password`, phrased for the person typing it.

    Returns all problems rather than the first, because fixing one at a
    time and being rejected again is how people end up at "Password1!".
    """
    password = password or ""
    problems: List[str] = []

    if len(password) < MIN_LENGTH:
        problems.append(f"Use at least {MIN_LENGTH} characters.")
    if len(password) > MAX_LENGTH:
        problems.append(f"Keep it under {MAX_LENGTH} characters.")

    # Length carries most of the strength, so the variety rule only
    # applies below the point where length alone is doing the work. A long
    # passphrase of plain words should not be rejected for lacking a digit.
    if len(password) < 16:
        classes = sum(bool(pattern.search(password)) for pattern in (
            re.compile(r"[a-z]"), re.compile(r"[A-Z]"),
            re.compile(r"[0-9]"), re.compile(r"[^A-Za-z0-9]")))
        if classes < 3:
            problems.append(
                "Mix at least three of: lower case, upper case, digits, symbols — "
                "or use a longer passphrase of 16+ characters instead.")

    lowered = password.lower()
    if lowered in _COMMON:
        problems.append("That is one of the most common passwords in use. Pick another.")
    if re.fullmatch(r"(.)\1*", password or "x"):
        problems.append("Don't use a single repeated character.")

    local = (email or "").split("@")[0].strip().lower()
    if local and len(local) >= 3 and local in lowered:
        problems.append("Don't include your email address.")
    for part in (name or "").lower().split():
        if len(part) >= 4 and part in lowered:
            problems.append("Don't include your name.")
            break

    return problems


def is_strong(password: str, email: str = "", name: str = "") -> bool:
    return not strength_problems(password, email, name)


def new_token(bytes_of_entropy: int = 32) -> str:
    """A URL-safe random token for password resets."""
    return secrets.token_urlsafe(bytes_of_entropy)


def hash_token(token: str) -> str:
    """Reset tokens are stored HASHED, like passwords.

    A plain SHA-256 rather than scrypt: a 256-bit random token has nothing
    to brute-force, so the memory-hard cost buys nothing and would only
    slow every verification. What matters is that the store never holds
    the value that would let someone take over an account.
    """
    return hashlib.sha256((token or "").encode("utf-8")).hexdigest()


def tokens_match(token: str, hashed: str) -> bool:
    return hmac.compare_digest(hash_token(token), hashed or "")
