"""
Parses the Authentication-Results header that a receiving mail server
already attaches to every inbound email. This module does NOT perform
any cryptographic verification itself -- SPF/DKIM/DMARC checks are
already computed by the mail server before the message ever reaches
this pipeline. All this does is read the answer.

Why this matters: content analysis alone can be fooled by well-written
text. A message that impersonates a bank but FAILS DMARC is a stronger,
more deterministic signal than the LLM's read of the words -- an
attacker can write a convincing sentence, but can't as easily forge a
domain's cryptographic authentication result.

Real header example (Gmail-style, one line, wrapped here for readability):
    Authentication-Results: mx.google.com;
        spf=pass (google.com: domain of alerts@bank.com designates
            203.0.113.5 as permitted sender) smtp.mailfrom=alerts@bank.com;
        dkim=pass header.i=@bank.com header.s=selector1 header.b=abc123;
        dmarc=pass (p=REJECT sp=REJECT dis=NONE) header.from=bank.com

A failing example looks the same shape, just with fail/softfail/none
in place of pass, e.g.:
    Authentication-Results: mx.google.com;
        spf=softfail (google.com: domain of transitioning
            attacker@bank-verify-support.com does not designate
            198.51.100.9 as permitted sender);
        dkim=none (message not signed);
        dmarc=fail (p=REJECT sp=REJECT dis=NONE) header.from=bank.com
"""

import re
from typing import Dict, Optional, TypedDict

# Valid result values per the relevant RFCs (RFC 7208 for SPF, RFC 6376
# for DKIM, RFC 7489 for DMARC). Keeping this explicit means a value
# outside this set (a malformed or unexpected header) is caught rather
# than silently trusted.
_VALID_SPF = {"pass", "fail", "softfail", "neutral", "none", "temperror", "permerror"}
_VALID_DKIM = {"pass", "fail", "none", "neutral", "temperror", "permerror", "policy"}
_VALID_DMARC = {"pass", "fail", "none", "bestguesspass"}

# Matches e.g. "spf=pass", "dkim = fail", "dmarc=softfail" -- tolerant of
# the optional whitespace around '=' that different mail servers use,
# and stops at the next '(', ';', or whitespace so it doesn't accidentally
# swallow the parenthetical explanation that usually follows.
_SPF_RE = re.compile(r"\bspf\s*=\s*([a-zA-Z]+)")
_DKIM_RE = re.compile(r"\bdkim\s*=\s*([a-zA-Z]+)")
_DMARC_RE = re.compile(r"\bdmarc\s*=\s*([a-zA-Z]+)")


class AuthResults(TypedDict):
    spf_result: str
    dkim_result: str
    dmarc_result: str
    raw_header_present: bool


def _extract(pattern: re.Pattern, header_text: str, valid_values: set, field_name: str) -> str:
    match = pattern.search(header_text)
    if not match:
        return "none"
    value = match.group(1).lower()
    if value not in valid_values:
        # Don't silently accept something unrecognized as a pass/fail --
        # an unexpected token is safer treated as "we don't actually know"
        # than guessed at.
        return "none"
    return value


def parse_authentication_results(header_text: Optional[str]) -> AuthResults:
    """
    Parse a raw Authentication-Results header string into SPF/DKIM/DMARC
    verdicts. Safe to call with None or an empty string (e.g. a message
    with no such header at all, or a test payload that didn't supply one)
    -- returns "none" for each field rather than raising, since "no
    header present" is itself meaningful information (usually: this
    message didn't go through a mail server that performs these checks,
    or the header was stripped somewhere upstream).

    A single email can legitimately carry MULTIPLE Authentication-Results
    headers (one per receiving hop) or one header covering multiple
    authserv-ids. This deliberately takes the FIRST match of each
    spf=/dkim=/dmarc= token, which corresponds to the outermost (most
    recent, most trustworthy) receiving server's verdict -- the one
    closest to actually being your own infrastructure. If multiple
    hops disagree, this is a known limitation worth knowing about, not
    hidden: pass the full raw header through to the LLM prompt as well
    (see nodes.py) so a human reviewing the case can see the full text,
    not just the parsed summary.
    """
    if not header_text:
        return AuthResults(spf_result="none", dkim_result="none", dmarc_result="none", raw_header_present=False)

    return AuthResults(
        spf_result=_extract(_SPF_RE, header_text, _VALID_SPF, "spf"),
        dkim_result=_extract(_DKIM_RE, header_text, _VALID_DKIM, "dkim"),
        dmarc_result=_extract(_DMARC_RE, header_text, _VALID_DMARC, "dmarc"),
        raw_header_present=True,
    )


def get_deterministic_severity_floor(auth: AuthResults) -> Optional[str]:
    """
    Layer 2-equivalent for email: a deterministic override that doesn't
    depend on the LLM correctly weighing the auth result on its own.

    A DMARC failure means the message failed to authenticate as coming
    from the domain it claims to be from -- REGARDLESS of how benign
    the body text reads, a real institution's message should not be
    failing its own published DMARC policy. This mirrors the
    call-pipeline blocklist principle: a deterministic, cited signal
    should be able to set a floor under an LLM's severity judgment,
    not merely be one more thing the model might or might not weigh
    correctly on its own.

    Returns None if there's no reason to override the LLM's own
    judgment (no header present, or everything passed).
    """
    if not auth["raw_header_present"]:
        return None
    if auth["dmarc_result"] == "fail":
        return "medium"
    # SPF/DKIM failing while DMARC still passes is common and often
    # legitimate (e.g. a message forwarded through a mailing list) --
    # DMARC is deliberately the only field that sets a floor on its own,
    # matching real DMARC semantics: it's the field that ties SPF/DKIM
    # together with alignment and is what a mail server actually acts on.
    return None


def format_auth_context_block(auth: AuthResults) -> str:
    """Human-readable block to insert directly into the LLM prompt."""
    if not auth["raw_header_present"]:
        return "No Authentication-Results header was available for this message."
    return (
        f"SPF={auth['spf_result']}, DKIM={auth['dkim_result']}, "
        f"DMARC={auth['dmarc_result']}"
    )


if __name__ == "__main__":
    # Runnable without pytest -- quick manual verification against
    # realistic header shapes. Run: python -m backend.src.pipelines.text_fraud.auth_headers
    samples = {
        "clean, all-pass (Gmail-style)": (
            "mx.google.com; spf=pass (google.com: domain of alerts@bank.com "
            "designates 203.0.113.5 as permitted sender) smtp.mailfrom=alerts@bank.com; "
            "dkim=pass header.i=@bank.com header.s=selector1 header.b=abc123; "
            "dmarc=pass (p=REJECT sp=REJECT dis=NONE) header.from=bank.com"
        ),
        "spoofed domain, DMARC fail": (
            "mx.google.com; spf=softfail (google.com: domain of transitioning "
            "attacker@bank-verify-support.com does not designate 198.51.100.9 "
            "as permitted sender); dkim=none (message not signed); "
            "dmarc=fail (p=REJECT sp=REJECT dis=NONE) header.from=bank.com"
        ),
        "no header at all": None,
        "malformed / unexpected token": "spf=weird_value; dkim=pass; dmarc=fail",
    }
    for label, header in samples.items():
        result = parse_authentication_results(header)
        floor = get_deterministic_severity_floor(result)
        print(f"\n{label}:")
        print(f"  parsed: {result}")
        print(f"  severity floor: {floor}")
        print(f"  prompt context: {format_auth_context_block(result)}")
