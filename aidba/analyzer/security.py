"""Security anomaly detection."""
import re
import hashlib

SUSPICIOUS_PATTERNS = [
    (re.compile(r"\b1\s*=\s*1\b", re.I), "tautology"),
    (re.compile(r"\bOR\s+1\s*=\s*1\b", re.I), "tautology"),
    (re.compile(r"\bUNION\s+SELECT\b", re.I), "union_select"),
    (re.compile(r";\s*DROP\s+", re.I), "stacked_drop"),
    (re.compile(r"\bxp_cmdshell\b", re.I), "xp_cmdshell"),
    (re.compile(r"\bINTO\s+OUTFILE\b", re.I), "into_outfile"),
    (re.compile(r"\bpg_read_file\b", re.I), "pg_read_file"),
    (re.compile(r"information_schema\.tables", re.I), "schema_enumeration"),
    (re.compile(r"SELECT\s+\*\s+FROM\s+users", re.I), "full_users_select"),
]


class SecurityAnalyzer:
    def scan_query(self, db_name, query):
        if not query:
            return None
        for pat, kind in SUSPICIOUS_PATTERNS:
            if pat.search(query):
                return {
                    "rule": kind,
                    "query_hash": hashlib.sha1(query.encode()).hexdigest()[:16],
                    "snippet": query[:300],
                }
        return None
