"""
tests/test_security.py
──────────────────────
Phase 3 unit tests: argon2id password hashing + JWT issuing/verification.
Hermetic — no DB, no Redis.
"""

from __future__ import annotations

import uuid

import pytest

from backend.core.security import (
    InvalidTokenError,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)


class TestPasswords:
    def test_hash_and_verify_roundtrip(self) -> None:
        h = hash_password("correct horse battery staple")
        assert h != "correct horse battery staple"
        assert verify_password("correct horse battery staple", h)

    def test_wrong_password_fails(self) -> None:
        h = hash_password("secret")
        assert not verify_password("wrong", h)

    def test_hashes_are_salted(self) -> None:
        assert hash_password("same") != hash_password("same")

    def test_malformed_hash_reads_as_failure(self) -> None:
        assert not verify_password("x", "not-a-hash")


class TestAccessTokens:
    def test_student_token_roundtrip(self) -> None:
        sid = uuid.uuid4()
        token, exp = create_access_token(subject=str(sid), token_type="student")
        claims = decode_token(token)
        assert claims["sub"] == str(sid)
        assert claims["principal_type"] == "student"

    def test_admin_token_carries_scope_claims(self) -> None:
        inst, dept = uuid.uuid4(), uuid.uuid4()
        token, _ = create_access_token(
            subject=str(uuid.uuid4()),
            token_type="admin",
            role="department_admin",
            institution_id=inst,
            department_id=dept,
        )
        claims = decode_token(token)
        assert claims["role"] == "department_admin"
        assert claims["institution_id"] == str(inst)
        assert claims["department_id"] == str(dept)

    def test_garbage_token_rejected(self) -> None:
        with pytest.raises(InvalidTokenError):
            decode_token("not.a.jwt")

    def test_tampered_signature_rejected(self) -> None:
        token, _ = create_access_token(subject=str(uuid.uuid4()))
        tampered = token[:-6] + ("aaaaaa" if not token.endswith("aaaaaa") else "bbbbbb")
        with pytest.raises(InvalidTokenError):
            decode_token(tampered)


class TestRefreshTokens:
    def test_refresh_roundtrip(self) -> None:
        token, _ = create_refresh_token(subject=str(uuid.uuid4()), principal_type="admin")
        claims = decode_token(token, expected_type="refresh")
        assert claims["type"] == "refresh"

    def test_access_token_is_not_a_refresh_token(self) -> None:
        token, _ = create_access_token(subject=str(uuid.uuid4()))
        with pytest.raises(InvalidTokenError):
            decode_token(token, expected_type="refresh")
