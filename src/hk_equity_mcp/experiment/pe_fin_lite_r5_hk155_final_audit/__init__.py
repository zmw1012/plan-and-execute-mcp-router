"""PE-Fin Lite R5 HK-155 formal final audit (read-only, zero-network)."""

AUDIT_VERSION = "pe_fin_lite_r5_hk155_final_audit_v1"
RESULT_ROLE = "formal_offline_descriptive"
DEFAULT_AUDIT_SUBDIR = "final_audit_r5_v1"
DOCS_MIRROR_SUBDIR = "docs/pe_fin_lite_r5_hk155_final_audit_v1"
EXPECTED_N = 155
EXPECTED_DOMAIN_COUNTS = {
    "financial": 60,
    "market": 35,
    "news": 35,
    "cross_domain": 25,
}
EXPECTED_FORMULA_HASH = (
    "7414751814f6858a375d95da918921e3ee365dfc5b69cdaba0cee1bf9e1ecb3e"
)
EXPECTED_CANDIDATE_LOCK = (
    "6a2f6ec1ba4fcf70ab50d01fc21e7a1aabda95f1beb15981f32eda3dc21d28b4"
)
EXPECTED_INDEX_IDENTITY = (
    "ac4c109bed0edb21fb2b837dfdbd745680b24f33efa62e13ad6bca53f174b0df"
)

__all__ = [
    "AUDIT_VERSION",
    "RESULT_ROLE",
    "DEFAULT_AUDIT_SUBDIR",
    "EXPECTED_N",
]
