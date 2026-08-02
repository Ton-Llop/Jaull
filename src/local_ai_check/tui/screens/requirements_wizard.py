from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    RadioButton,
    RadioSet,
    Static,
)

from local_ai_check.tui.widgets.warnings_panel import WarningsPanel
from local_ai_check.tui.widgets.workflow_header import WorkflowHeader
from local_ai_check.workflow.models import (
    CommercialUse,
    ConcurrencyLevel,
    DocumentScale,
    RecommendationPriority,
    UseCase,
    UserAnswers,
    WorkflowStep,
)
from local_ai_check.workflow.requirements import normalize_languages

if TYPE_CHECKING:
    from local_ai_check.tui.app import LocalAiCheckApp

# Every question is (internal value, user-facing label). No jargon on the right.
_USE_CASES: tuple[tuple[UseCase, str], ...] = (
    (UseCase.GENERAL_CHAT, "General chat and assistant"),
    (UseCase.CODING, "Programming and code"),
    (UseCase.DOCUMENT_QA, "Documents and company knowledge"),
    (UseCase.WRITING_TRANSLATION, "Writing and translation"),
)

_PRIORITIES: tuple[tuple[RecommendationPriority, str], ...] = (
    (RecommendationPriority.QUALITY, "Best quality"),
    (RecommendationPriority.BALANCED, "Balanced"),
    (RecommendationPriority.SPEED, "Fast responses"),
    (RecommendationPriority.MEMORY, "Lowest memory usage"),
)

_LANGUAGES: tuple[str, ...] = ("Spanish", "Catalan", "English", "Portuguese")

_CONCURRENCY: tuple[tuple[ConcurrencyLevel, str], ...] = (
    (ConcurrencyLevel.SINGLE, "One user"),
    (ConcurrencyLevel.SMALL, "2-5 users"),
    (ConcurrencyLevel.MEDIUM, "6-20 users"),
    (ConcurrencyLevel.LARGE, "More than 20 users"),
)

_DOCUMENT_SCALE: tuple[tuple[DocumentScale, str], ...] = (
    (DocumentScale.SHORT, "Short questions and snippets"),
    (DocumentScale.MEDIUM, "Medium reports"),
    (DocumentScale.LONG, "Long documents"),
    (DocumentScale.COLLECTION, "Large document collections"),
)

_COMMERCIAL: tuple[tuple[CommercialUse, str], ...] = (
    (CommercialUse.YES, "Yes"),
    (CommercialUse.NO, "No"),
    (CommercialUse.NOT_SURE, "Not sure"),
)


class RequirementsWizardScreen(Screen[None]):
    """Step 2: six plain-language questions, no technical terms."""

    BINDINGS = [("escape", "app.pop_screen", "Back"), ("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield WorkflowHeader(
            WorkflowStep.REQUIREMENTS,
            "What do you need?",
            "Answer in plain terms — the technical settings are derived for you.",
        )
        with VerticalScroll():
            with Vertical(classes="card"):
                yield Static("1. What will you use it for?", classes="card-title")
                yield RadioSet(
                    *[
                        RadioButton(label, value=index == 0, id=f"uc-{case.value}")
                        for index, (case, label) in enumerate(_USE_CASES)
                    ],
                    id="q-use-case",
                )
            with Vertical(classes="card"):
                yield Static("2. What matters most?", classes="card-title")
                yield RadioSet(
                    *[
                        RadioButton(
                            label,
                            value=priority is RecommendationPriority.BALANCED,
                            id=f"pr-{priority.value}",
                        )
                        for priority, label in _PRIORITIES
                    ],
                    id="q-priority",
                )
            with Vertical(classes="card"):
                yield Static("3. Which languages?", classes="card-title")
                for language in _LANGUAGES:
                    yield Checkbox(
                        language,
                        value=language == "English",
                        id=f"lang-{language.lower()}",
                    )
                yield Checkbox("Other", value=False, id="lang-other")
                yield Input(
                    placeholder="Other languages, comma separated (e.g. fr, italian)",
                    id="lang-other-input",
                )
            with Vertical(classes="card"):
                yield Static("4. How many people will use it at once?", classes="card-title")
                yield RadioSet(
                    *[
                        RadioButton(
                            label,
                            value=level is ConcurrencyLevel.SINGLE,
                            id=f"cc-{level.value}",
                        )
                        for level, label in _CONCURRENCY
                    ],
                    id="q-concurrency",
                )
            # Only meaningful for document work; hidden otherwise so the wizard
            # never asks a question the answer cannot influence.
            with Vertical(classes="card", id="q-documents-card"):
                yield Static("5. How much text at a time?", classes="card-title")
                yield Static(
                    "This sets the model's context window. It is not the size of a "
                    "document collection — a retrieval system feeds the model a few "
                    "chunks at a time.",
                    classes="text-muted",
                )
                yield RadioSet(
                    *[
                        RadioButton(
                            label,
                            value=scale is DocumentScale.MEDIUM,
                            id=f"ds-{scale.value}",
                        )
                        for scale, label in _DOCUMENT_SCALE
                    ],
                    id="q-documents",
                )
            with Vertical(classes="card"):
                yield Static(
                    "6. Must the model allow commercial use?", classes="card-title"
                )
                yield RadioSet(
                    *[
                        RadioButton(
                            label, value=choice is CommercialUse.YES, id=f"cu-{choice.value}"
                        )
                        for choice, label in _COMMERCIAL
                    ],
                    id="q-commercial",
                )
            yield Vertical(id="wizard-errors")
            yield Button("Find models", id="wizard-submit", classes="-primary")
        yield Footer()

    def on_mount(self) -> None:
        self._sync_document_question()

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.radio_set.id == "q-use-case":
            self._sync_document_question()

    def _sync_document_question(self) -> None:
        """Question 5 exists only for the document use case."""
        card = self.query_one("#q-documents-card", Vertical)
        card.display = self._selected_use_case() is UseCase.DOCUMENT_QA

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "wizard-submit":
            return
        answers = self.collect_answers()
        if answers is None:
            return
        self._app().start_discovery(answers)

    def collect_answers(self) -> UserAnswers | None:
        """Read the widgets into a :class:`UserAnswers`, or report why not.

        Public so tests can drive the wizard without pressing buttons.
        """
        errors = self.query_one("#wizard-errors", Vertical)
        errors.remove_children()

        languages = [
            language
            for language in _LANGUAGES
            if self.query_one(f"#lang-{language.lower()}", Checkbox).value
        ]

        other: list[str] = []
        if self.query_one("#lang-other", Checkbox).value:
            raw = self.query_one("#lang-other-input", Input).value
            entries = [part.strip() for part in raw.split(",") if part.strip()]
            codes, rejected = normalize_languages(entries)
            if rejected:
                errors.mount(
                    WarningsPanel(
                        [
                            "These entries are not recognised languages: "
                            + ", ".join(rejected)
                        ]
                    )
                )
                return None
            if not codes:
                errors.mount(
                    WarningsPanel(["Enter at least one language, or untick 'Other'."])
                )
                return None
            other = codes

        if not languages and not other:
            errors.mount(WarningsPanel(["Select at least one language."]))
            return None

        use_case = self._selected_use_case()
        return UserAnswers(
            use_case=use_case,
            priority=self._selected(
                "q-priority", "pr-", _PRIORITIES, RecommendationPriority.BALANCED
            ),
            languages=languages,
            other_languages=other,
            concurrency=self._selected(
                "q-concurrency", "cc-", _CONCURRENCY, ConcurrencyLevel.SINGLE
            ),
            document_scale=(
                self._selected(
                    "q-documents", "ds-", _DOCUMENT_SCALE, DocumentScale.MEDIUM
                )
                if use_case is UseCase.DOCUMENT_QA
                else None
            ),
            commercial_use=self._selected(
                "q-commercial", "cu-", _COMMERCIAL, CommercialUse.YES
            ),
        )

    def _selected_use_case(self) -> UseCase:
        return self._selected("q-use-case", "uc-", _USE_CASES, UseCase.GENERAL_CHAT)

    def _selected[T](
        self,
        radio_set_id: str,
        prefix: str,
        options: tuple[tuple[T, str], ...],
        default: T,
    ) -> T:
        radio_set = self.query_one(f"#{radio_set_id}", RadioSet)
        pressed = radio_set.pressed_button
        if pressed is None or pressed.id is None:
            return default
        wanted = pressed.id.removeprefix(prefix)
        for value, _ in options:
            if str(value) == wanted:
                return value
        return default

    def _app(self) -> LocalAiCheckApp:
        from local_ai_check.tui.app import LocalAiCheckApp

        assert isinstance(self.app, LocalAiCheckApp)
        return self.app


__all__ = ["RequirementsWizardScreen"]
