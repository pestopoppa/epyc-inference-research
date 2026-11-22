"""Install the comparison-receipt projector and its real gate in InstalledFeedBinding (AKU-08c4).

``receipt_feed_binding`` builds the application-installed binding whose finding owner is
``ReceiptBackedFindingProjector`` fed by the producer's ``comparison_receipt_journal``
(``lookup(path)``) and gated by ``registered_claim_grade_eligibility`` over the installed
``RegisteredClaimGradeVerifier``. Its scope/use/result verifiers re-resolve every Finding
against the durable journal at query time (no trust in the cached Finding), and each new
Finding carries the CURRENT invalidation generation of the opened feed instead of a
constant 0, so a Finding projected after a dependency invalidation is not born stale.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from . import comparison_receipt_journal as C
from . import feed_runtime as F
from . import scoped_evidence as evidence

SUPPORT_RULE = "autokernel:comparison-receipt-journal:v1"


class ReceiptFeedInstallation:
    """The installed finding owner plus its three query-time verifiers (one journal)."""

    def __init__(self, journal_path: str | Path, *, epoch: str, verifier: Any = None,
                 eligibility: Callable[[F.RegisteredComparisonReceipt], tuple[str, ...]] | None = None,
                 allow_fixture: bool = False) -> None:
        self.journal_path = str(journal_path)
        self.eligibility = eligibility or F.registered_claim_grade_eligibility(verifier)
        self.allow_fixture = allow_fixture
        self.projector = F.ReceiptBackedFindingProjector(
            C.lookup(self.journal_path), self.eligibility, epoch=epoch)
        self._feed = None

    # The feed owner attaches its opened feed: the generation source is that feed's index.
    def attach_feed(self, feed: Any) -> None:
        self._feed = feed

    def generations(self) -> Mapping[str, int]:
        if self._feed is None:
            raise F.FeedRuntimeRefused("receipt projector has no current invalidation generation owner")
        return self._feed.index().dependency_generations

    def __call__(self, projected: Any, grade: Any, row: Any, frontier: int) -> evidence.Finding | None:
        current = self.generations()
        finding = self.projector(projected, grade, row, frontier)
        if finding is None:
            return None
        body = finding.to_dict()
        body["dependency_generations"] = {
            dep: int(current.get(dep, 0)) for dep in finding.claim_key.dependency_identities}
        return evidence.Finding.from_dict(body)

    @property
    def refusals(self) -> Mapping[str, tuple[str, ...]]:
        return self.projector.refusals

    def _receipt(self, finding: evidence.Finding) -> F.RegisteredComparisonReceipt | str:
        locator = finding.source.locator.removeprefix("fixture:")
        try:
            receipt = C.replay(self.journal_path).receipt(locator)
        except C.ComparisonReceiptRefused as exc:
            return f"comparison_receipt_journal_refused: {exc}"
        if receipt is None:
            return "comparison_receipt_absent"
        if receipt.receipt_sha256 != finding.source.artifact_digest:
            return "comparison_receipt_digest_differs"
        if receipt.claim_key.digest != finding.claim_key.digest:
            return "comparison_receipt_claim_differs"
        if receipt.fixture_only and not self.allow_fixture:
            return "comparison_receipt_fixture_only"
        return receipt

    def scope_verifier(self, finding: evidence.Finding, claim: evidence.ClaimKey,
                       scope: Mapping[str, Any]) -> bool | str:
        receipt = self._receipt(finding)
        if isinstance(receipt, str):
            return receipt
        if dict(scope) != dict(receipt.claim_key.target_scope) or claim.digest != receipt.claim_key.digest:
            return "comparison_receipt_scope_differs"
        return True

    def use_verifier(self, finding: evidence.Finding, use: str) -> bool | str:
        receipt = self._receipt(finding)
        if isinstance(receipt, str):
            return receipt
        if receipt.intended_use != use:
            return "comparison_receipt_use_differs"
        codes = tuple(self.eligibility(receipt))  # eligibility is re-decided now, not cached
        return True if not codes else ",".join(codes)

    def result_verifier(self, selected: tuple, considered: tuple, claim: evidence.ClaimKey,
                        use: str) -> bool | str:
        for item in (*selected, *considered):
            receipt = self._receipt(item.finding)
            if isinstance(receipt, str):
                return receipt
        return True


def receipt_feed_binding(root_repo: Path, root_source_sha256: Mapping[str, str], current_epoch: str,
                         installation: ReceiptFeedInstallation) -> F.InstalledFeedBinding:
    """The complete installed verifier set; FeedRuntimeOwner attaches the generation owner."""
    if type(installation) is not ReceiptFeedInstallation or installation.projector.epoch != current_epoch:
        raise F.FeedRuntimeRefused("receipt feed installation differs from the installed epoch")
    return F.InstalledFeedBinding(
        root_repo, root_source_sha256, current_epoch, finding_projector=installation,
        scope_verifier=installation.scope_verifier, use_verifier=installation.use_verifier,
        result_verifier=installation.result_verifier, support_rule_identity=SUPPORT_RULE)
