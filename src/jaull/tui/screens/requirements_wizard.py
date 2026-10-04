from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    Digits,
    Footer,
    Input,
    RadioButton,
    RadioSet,
    Static,
)

from jaull.application.requirements import normalize_languages
from jaull.domain.requirements import (
    CommercialUse,
    ConcurrencyLevel,
    DocumentScale,
    RecommendationPriority,
    UseCase,
    UserAnswers,
    WorkloadMode,
)
from jaull.tui.palette import INK_3
from jaull.tui.widgets.warnings_panel import WarningsPanel
from jaull.tui.widgets.workflow_header import WorkflowHeader
from jaull.workflow.models import WorkflowStep

if TYPE_CHECKING:
    from jaull.tui.app import JaullApp

# Every question is (internal value, user-facing label). No jargon on the right.
_USE_CASES: tuple[tuple[UseCase, str], ...] = (
    (UseCase.GENERAL_CHAT, "General chat and assistant"),
    (UseCase.CODING, "Programming and code"),
    (UseCase.DOCUMENT_QA, "Documents and company knowledge"),
    (UseCase.SUMMARIZATION_EXTRACTION, "Summarization and extraction"),
    (UseCase.REASONING, "Reasoning and problem solving"),
    (UseCase.WRITING_TRANSLATION, "Writing and translation"),
)

# Question titles carry no number of their own. One of them is hidden unless
# the use case is document work, so a literal "6" left the user looking at
# 1-2-3-4-5-7. `_renumber_questions` assigns the ordinals to whatever is
# actually on screen.
_QUESTION_TITLES: tuple[tuple[str, str], ...] = (
    ("q-title-use-case", "What will you use it for?"),
    ("q-title-workload-mode", "How will requests be processed?"),
    ("q-title-priority", "What matters most?"),
    ("q-title-languages", "Which languages?"),
    ("q-title-concurrency", "How many people will use it at once?"),
    ("q-title-documents", "How much text at a time?"),
    ("q-title-commercial", "Must the model allow commercial use?"),
)


_WORKLOAD_MODES: tuple[tuple[WorkloadMode, str], ...] = (
    (WorkloadMode.INTERACTIVE, "Interactive — someone is waiting for the answer"),
    (WorkloadMode.BATCH, "Batch processing — jobs run offline or asynchronously"),
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
    (ConcurrencyLevel.MEDIUM, "6-10 users"),
    (ConcurrencyLevel.LARGE, "10+ users"),
)

_DOCUMENT_SCALE: tuple[tuple[DocumentScale, str], ...] = (
    (DocumentScale.SHORT, "Short questions and snippets"),
    (DocumentScale.MEDIUM, "Medium reports"),
    (DocumentScale.LONG, "Long documents"),
    (DocumentScale.COLLECTION, "Large document collections"),
)

_COMMERCIAL: tuple[tuple[CommercialUse, str], ...] = (
    (CommercialUse.YES, "Commercial required"),
    (CommercialUse.NO, "Non-commercial acceptable"),
    (CommercialUse.NO_PREFERENCE, "No preference"),
    (CommercialUse.NOT_SURE, "Not sure"),
)



def _option_label(text: str) -> str:
    """Separate an option's name from the sentence that explains it.

    "Interactive — someone is waiting for the answer" is two things wearing one
    weight: the name you are choosing, and the reason it exists. Dimming the
    tail keeps the exact wording while letting the eye land on the name first,
    and costs no extra row, which matters in a column holding six questions.
    """
    name, separator, detail = text.partition(" — ")
    if not separator:
        return text
    return f"{name}  [{INK_3}]{detail}[/]"


class RequirementsWizardScreen(Screen[None]):
    """Step 2: task and workload are separate questions."""

    BINDINGS = [("escape", "app.pop_screen", "Back"), ("q", "quit", "Quit")]

    def on_resize(self) -> None:
        self._apply_columns()

    def _apply_columns(self) -> None:
        """One column below 100 cells: two would each be too narrow to read."""
        columns = self.query(".wizard-columns")
        if columns:
            columns.first().set_class(self.size.width < 100, "-narrow")

    def compose(self) -> ComposeResult:
        yield WorkflowHeader(
            WorkflowStep.REQUIREMENTS,
            "Your needs",
        )
        # Numbered questions separated by whitespace, not by six boxes: the
        # heading and the indent already say where one question ends.
        with VerticalScroll(id="wizard-body"):
            # Seven questions stacked in one column ran past the fold, which put
            # the submit button somewhere you had to go looking for. Two columns
            # halve the height; below 100 cells they fold back into one, where
            # stacking is the only honest option.
            with Horizontal(classes="wizard-columns"):
                with Vertical(classes="wizard-column"):
                    with Horizontal(classes="question"):
                        yield Digits("1", id="q-num-use-case", classes="question-number")
                        with Vertical(classes="question-body"):
                            yield Static(id="q-title-use-case", classes="question-title")
                            yield RadioSet(
                                *[
                                    RadioButton(label, value=index == 0, id=f"uc-{case.value}")
                                    for index, (case, label) in enumerate(_USE_CASES)
                                ],
                                id="q-use-case",
                            )
                    with Horizontal(classes="question"):
                        yield Digits("1", id="q-num-workload-mode", classes="question-number")
                        with Vertical(classes="question-body"):
                            yield Static(id="q-title-workload-mode", classes="question-title")
                            yield RadioSet(
                                *[
                                    RadioButton(
                                        _option_label(label),
                                        value=mode is WorkloadMode.INTERACTIVE,
                                        id=f"wm-{mode.value}",
                                    )
                                    for mode, label in _WORKLOAD_MODES
                                ],
                                id="q-workload-mode",
                            )
                    with Horizontal(classes="question"):
                        yield Digits("1", id="q-num-priority", classes="question-number")
                        with Vertical(classes="question-body"):
                            yield Static(id="q-title-priority", classes="question-title")
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
                with Vertical(classes="wizard-column"):
                    with Horizontal(classes="question"):
                        yield Digits("1", id="q-num-languages", classes="question-number")
                        with Vertical(classes="question-body"):
                            yield Static(id="q-title-languages", classes="question-title")
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
                    with Horizontal(classes="question"):
                        yield Digits("1", id="q-num-concurrency", classes="question-number")
                        with Vertical(classes="question-body"):
                            yield Static(id="q-title-concurrency", classes="question-title")
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
                    with Horizontal(classes="question", id="q-documents-card"):
                        yield Digits("1", id="q-num-documents", classes="question-number")
                        with Vertical(classes="question-body"):
                            yield Static(id="q-title-documents", classes="question-title")
                            yield Static(
                                "Sets the context window — not the size of a document "
                                "collection; retrieval feeds the model a few chunks at a time.",
                                classes="question-note",
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
                    with Horizontal(classes="question"):
                        yield Digits("1", id="q-num-commercial", classes="question-number")
                        with Vertical(classes="question-body"):
                            yield Static(id="q-title-commercial", classes="question-title")
                            yield RadioSet(
                                *[
                                    RadioButton(
                                        label,
                                        value=choice is CommercialUse.YES,
                                        id=f"cu-{choice.value}",
                                    )
                                    for choice, label in _COMMERCIAL
                                ],
                                id="q-commercial",
                            )
            yield Vertical(id="wizard-errors")
            with Horizontal(id="wizard-actions", classes="actions-right"):
                yield Button("Find models", id="wizard-submit", classes="-primary")
        yield Footer()

    def on_mount(self) -> None:
        self._sync_document_question()

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.radio_set.id == "q-use-case":
            self._sync_document_question()

    def _sync_document_question(self) -> None:
        """The text-size question exists only for the document use case."""
        card = self.query_one("#q-documents-card", Horizontal)
        card.display = self._selected_use_case() is UseCase.DOCUMENT_QA
        self._renumber_questions()

    def _renumber_questions(self) -> None:
        """Number the questions the user can actually see, in order.

        Hiding a question in the middle used to leave a gap in the sequence,
        because each title carried its ordinal as a literal string.
        """
        documents_visible = self.query_one("#q-documents-card", Horizontal).display
        ordinal = 0
        for widget_id, text in _QUESTION_TITLES:
            if widget_id == "q-title-documents" and not documents_visible:
                continue
            ordinal += 1
            self.query_one(f"#{widget_id}", Static).update(text)
            number = widget_id.replace("q-title-", "q-num-")
            self.query_one(f"#{number}", Digits).update(str(ordinal))

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
            workload_mode=self._selected(
                "q-workload-mode", "wm-", _WORKLOAD_MODES, WorkloadMode.INTERACTIVE
            ),
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

    def _app(self) -> JaullApp:
        from jaull.tui.app import JaullApp

        assert isinstance(self.app, JaullApp)
        return self.app


__all__ = ["RequirementsWizardScreen"]
