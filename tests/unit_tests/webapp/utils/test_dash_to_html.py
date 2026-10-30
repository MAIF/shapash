import unittest

from dash import html

from shapash.webapp.utils.dash_to_html import DashHtmlPreview, dash_component_to_html


class TestDashComponentToHtml(unittest.TestCase):
    def test_renders_tag_and_text_child(self):
        self.assertEqual(dash_component_to_html(html.Div("hello")), "<div>hello</div>")

    def test_renders_nested_children_in_order(self):
        component = html.Div([html.Span("a"), html.Span("b")])
        self.assertEqual(dash_component_to_html(component), "<div><span>a</span><span>b</span></div>")

    def test_renders_style_dict_as_kebab_case_css(self):
        component = html.Span("x", style={"backgroundColor": "rgb(1,2,3)", "fontSize": "1em"})
        result = dash_component_to_html(component)
        self.assertIn('style="background-color:rgb(1,2,3);font-size:1em"', result)

    def test_renders_title_and_class_name(self):
        component = html.Span("x", title="tip", className="fw-bold")
        result = dash_component_to_html(component)
        self.assertIn('title="tip"', result)
        self.assertIn('class="fw-bold"', result)

    def test_escapes_text_content_and_attribute_values(self):
        component = html.Span("<script>", title='say "hi"')
        result = dash_component_to_html(component)
        self.assertNotIn("<script>", result)
        self.assertIn("&lt;script&gt;", result)
        self.assertIn("&quot;hi&quot;", result)

    def test_none_children_renders_as_empty(self):
        self.assertEqual(dash_component_to_html(html.Div()), "<div></div>")


class TestDashHtmlPreview(unittest.TestCase):
    def test_repr_html_wraps_the_component_markup_in_a_shrink_to_fit_container(self):
        component = html.Div([html.Span("hi")])
        preview = DashHtmlPreview(component)
        rendered = preview._repr_html_()
        self.assertIn(dash_component_to_html(component), rendered)
        self.assertIn("display:inline-block", rendered)
        self.assertIn("max-width:100%", rendered)

    def test_repr_html_pins_an_explicit_light_background_and_dark_text(self):
        """Ambient-inherited text colour is invisible against these panels in a dark IDE theme."""
        preview = DashHtmlPreview(html.Div("hi"))
        rendered = preview._repr_html_()
        self.assertIn("background-color:#ffffff", rendered)
        self.assertIn("color:#212529", rendered)

    def test_max_width_is_customizable(self):
        component = html.Div("hi")
        preview = DashHtmlPreview(component, max_width="500px")
        self.assertIn("max-width:500px", preview._repr_html_())

    def test_component_attribute_exposes_the_underlying_dash_component(self):
        component = html.Div("hi")
        preview = DashHtmlPreview(component)
        self.assertIs(preview.component, component)

    def test_repr_stays_the_dash_component_repr(self):
        component = html.Div("hi")
        preview = DashHtmlPreview(component)
        self.assertEqual(repr(preview), repr(component))


if __name__ == "__main__":
    unittest.main()
