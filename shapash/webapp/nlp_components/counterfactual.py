"""Counterfactual component: generate what-if flips and apply them into the editor.

When the explainer binds more than one generator (e.g. HotFlip *and* AblationFlip on a gradient model)
a **method selector** lets the user switch between them live; with a single generator the selector is
hidden. Each generator's controls are rendered automatically from its ``config_spec()`` (so a new knob
or a new generator needs no UI change) into its own visibility-toggled group. Generated counterfactuals
are shown as a table; an *Apply* button publishes the chosen text to a shared store the data editor
subscribes to.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback_context, dcc, html
from dash.exceptions import PreventUpdate

from shapash.compute.generators.base import Field, IntField, TokenListField
from shapash.compute.generators.cf_utils import format_edit
from shapash.webapp.nlp_components.base import CAP_COUNTERFACTUAL, CAP_PREDICT, WebappComponent
from shapash.webapp.nlp_components.datapoint import datapoint_from_contributions

_CARD_STYLE = {"border": "1px solid #dee2e6", "borderRadius": "4px", "padding": "12px", "height": "100%"}

# Inline flex styles (not the Bootstrap ``d-flex`` class, whose ``!important`` would defeat the
# ``display: none`` visibility toggle on the per-generator control groups).
_INLINE = {"display": "flex", "alignItems": "center", "gap": "0.4rem"}
_GROUP_STYLE = {"display": "flex", "flexWrap": "wrap", "alignItems": "center", "gap": "0.75rem"}
_HIDDEN = {"display": "none"}


class CounterfactualComponent(WebappComponent):
    """Spec-driven counterfactual generation panel."""

    id = "counterfactual"
    name = "Counterfactual Suggestions"
    scope = "local"
    # Requires the editor too (it reads the editor's text and applies flips back into it).
    requires = frozenset({CAP_COUNTERFACTUAL, CAP_PREDICT})

    def _generators(self, engine) -> list[tuple[str, str]]:
        """Ordered ``(name, display_name)`` pairs the engine offers (empty when none)."""
        return engine.available_cf_generators() if engine is not None else []

    def _fields(self, engine, generator: str) -> list[tuple[str, Field]]:
        """Ordered ``(name, field)`` pairs from one generator's config spec."""
        return list(engine.cf_config_spec(generator).items())

    def _config_controls(self, engine, generator: str) -> list:
        """Render each config field of ``generator`` as an inline label+control (IntField → number)."""
        items = []
        for name, fld in self._fields(engine, generator):
            cid = f"{self.id}-cfg-{generator}-{name}"
            if isinstance(fld, IntField):
                control = dcc.Input(
                    id=cid,
                    type="number",
                    min=fld.minimum,
                    max=fld.maximum,
                    step=1,
                    value=fld.default,
                    style={"width": "80px"},
                )
            elif isinstance(fld, TokenListField):
                control = dbc.Input(
                    id=cid,
                    type="text",
                    value=",".join(fld.default),
                    placeholder="comma,separated",
                    style={"width": "150px"},
                )
            else:  # pragma: no cover - future field types
                control = dbc.Input(id=cid, type="text", value=str(fld.default))
            items.append(
                html.Div(
                    [html.Label(fld.label, className="small fw-bold mb-0"), control],
                    style=_INLINE,
                )
            )
        return items

    def layout(self, explanation, engine=None) -> html.Div:
        """Return the counterfactual card with a method selector and per-generator config controls.

        The selector and *every* generator's controls are built here (not injected by a callback) so
        their ids exist in the initial layout — otherwise the ``generate`` callback's ``State`` would
        reference not-yet-created objects. Only the active generator's control group is visible; the
        selector callback toggles the rest. The selector is hidden when a single generator is bound.
        """
        generators = self._generators(engine)
        default_gen = generators[0][0] if generators else None

        # Method selector — kept in the layout even for a single generator (so its id/State exists),
        # just hidden. Its config controls live in per-generator groups toggled by the selector.
        selector = html.Div(
            [
                html.Label("Method", className="small fw-bold mb-0"),
                dbc.Select(
                    id=f"{self.id}-generator",
                    options=[{"label": label, "value": name} for name, label in generators],
                    value=default_gen,
                    size="sm",
                    style={"width": "170px"},
                ),
            ],
            style=_INLINE if len(generators) > 1 else _HIDDEN,
        )
        groups = [
            html.Div(
                self._config_controls(engine, name),
                id=f"{self.id}-cfg-group-{name}",
                style=_GROUP_STYLE if name == default_gen else _HIDDEN,
            )
            for name, _ in generators
        ]
        # Selector + the active generator's controls all flow on one line to save vertical space.
        controls_row = html.Div(
            [selector, *groups],
            id=f"{self.id}-controls",
            className="d-flex flex-wrap align-items-center gap-3 mb-2",
        )
        return html.Div(
            [
                html.H6("Counterfactual Suggestions", className="fw-bold mb-2"),
                html.Small(
                    "Find minimal token edits that flip the prediction of the edited text.",
                    className="text-muted d-block mb-2",
                ),
                controls_row,
                dbc.Button("Generate", id=f"{self.id}-generate-btn", color="warning", size="sm", className="mb-2"),
                dcc.Loading(html.Div(id=f"{self.id}-results")),
                dcc.Store(id=f"{self.id}-store", data=[]),
            ],
            style=_CARD_STYLE,
        )

    def register_callbacks(self, app, explanation, engine, stores) -> None:
        """Wire the method selector, Generate, and per-row Apply (→ shared editor store)."""
        apply_store = stores["apply"]
        current_store = stores["current"]

        generators = self._generators(engine)
        gen_names = [name for name, _ in generators]
        # Flat, ordered list of every (generator, field name, field) — one State per rendered control.
        all_fields = [(gen, name, fld) for gen in gen_names for name, fld in self._fields(engine, gen)]
        config_states = [State(f"{self.id}-cfg-{gen}-{name}", "value") for gen, name, _ in all_fields]

        # Show only the selected generator's control group. Skipped for a single-method explainer
        # (nothing to toggle); config controls themselves live in `layout()` so their ids always exist.
        if len(gen_names) > 1:

            @app.callback(
                [Output(f"{self.id}-cfg-group-{gen}", "style") for gen in gen_names],
                Input(f"{self.id}-generator", "value"),
            )
            def toggle_controls(selected):
                return [_GROUP_STYLE if gen == selected else _HIDDEN for gen in gen_names]

        # Reads the current datapoint (a selected dataset row *or* an edited/predicted text) so
        # counterfactuals can be generated for both, not only hand-typed editor text.
        @app.callback(
            Output(f"{self.id}-results", "children"),
            Output(f"{self.id}-store", "data"),
            Input(f"{self.id}-generate-btn", "n_clicks"),
            State(current_store, "data"),
            State(f"{self.id}-generator", "value"),
            *config_states,
        )
        def generate(n_clicks, datapoint, selected_gen, *config_values):
            text = (datapoint or {}).get("text", "")
            if not n_clicks or not text or not text.strip():
                raise PreventUpdate
            gen_name = selected_gen or (gen_names[0] if gen_names else None)
            config = {}
            for (gen, name, fld), value in zip(all_fields, config_values, strict=True):
                if gen != gen_name:
                    continue  # only the active generator's controls feed its config
                if isinstance(fld, IntField):
                    config[name] = int(value) if value is not None else fld.default
                elif isinstance(fld, TokenListField):
                    config[name] = [t.strip() for t in (value or "").split(",") if t.strip()]
                else:  # pragma: no cover
                    config[name] = value
            cfs = engine.generate_counterfactuals(text, config=config, generator=gen_name)
            if not cfs:
                return html.Div("No counterfactual found within these limits.", className="text-muted"), []
            return _results_table(cfs, self.id), [cf.new_text for cf in cfs]

        # "Inspect" loads the chosen counterfactual into the editor (apply_store → textarea) *and*
        # makes it the current datapoint, so its token contributions render immediately in the shared
        # Sentence/Waterfall panels without a separate Predict click.
        @app.callback(
            Output(apply_store, "data"),
            Output(current_store, "data", allow_duplicate=True),
            Input({"type": f"{self.id}-apply", "index": ALL}, "n_clicks"),
            State(f"{self.id}-store", "data"),
            prevent_initial_call=True,
        )
        def apply(n_clicks_list, cf_texts):
            if not any(n_clicks_list or []):
                raise PreventUpdate
            triggered = callback_context.triggered_id
            if not triggered:
                raise PreventUpdate
            index = triggered["index"]
            if not cf_texts or not (0 <= index < len(cf_texts)):
                raise PreventUpdate
            text = cf_texts[index]
            contributions, label, _probs = engine.explain_text(text)
            return text, datapoint_from_contributions(text, contributions, label)


def _results_table(cfs, component_id: str):
    """Render counterfactuals as a table with per-row Apply buttons."""
    header = html.Thead(
        html.Tr([html.Th("→ Label"), html.Th("Δ prob"), html.Th("Substitutions"), html.Th("Text"), html.Th("")])
    )
    rows = []
    for i, cf in enumerate(cfs):
        subs = ", ".join(format_edit(old, new) for _, old, new in cf.substitutions)
        rows.append(
            html.Tr(
                [
                    html.Td(html.B(cf.new_label)),
                    html.Td(f"{cf.prob_delta:.2f}"),
                    html.Td(subs, style={"fontSize": "0.85em"}),
                    html.Td(cf.new_text, style={"fontSize": "0.85em"}),
                    html.Td(
                        dbc.Button(
                            "Inspect",
                            id={"type": f"{component_id}-apply", "index": i},
                            color="link",
                            size="sm",
                            className="p-0",
                            title="Load into the editor and show its token contributions on the right",
                        )
                    ),
                ]
            )
        )
    return dbc.Table([header, html.Tbody(rows)], bordered=False, hover=True, size="sm", className="mb-0")
