"""
Unit test smart plotter
"""
import types
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from shapash.explainer.smart_explainer import SmartExplainer

class TestSmartPlotter(unittest.TestCase):
    """
    Unit test Smart plotter class
    check the different plots available

    Parameters
    ----------
    unittest : [type]
        [description]
    """
    def predict_proba(self, arg1, X):
        """
        predict_proba method
        """
        classes = [1, 3]
        proba = 1 / len(classes)
        vect = [proba] * len(classes)
        matrx = np.repeat(np.array([vect]), X.shape[0], axis=0)

        return matrx

    def predict(self, arg1, arg2):
        """
        predict method
        """
        matrx = np.array(
            [1, 3]
        )
        return matrx

    def setUp(self):
        """
        SetUp
        """
        self.x_pred = pd.DataFrame(
            data=np.array(
                [['PhD', 34],
                 ['Master', 27]]),
            columns=['X1', 'X2'],
            index=['person_A', 'person_B']
        )
        self.contrib0 = pd.DataFrame(
            data=np.array(
                [[-3.4, 0.78],
                 [1.2, 3.6]]),
            columns=['X1', 'X2'],
            index=['person_A', 'person_B']
        )
        self.contrib1 = pd.DataFrame(
            data=np.array(
                [[-0.3, 0.89],
                 [4.7, 0.6]]),
            columns=['X1', 'X2'],
            index=['person_A', 'person_B']
        )
        self.contrib_sorted = pd.DataFrame(
            data=np.array(
                [[-3.4, 0.78],
                 [3.6, 1.2]]),
            columns=['contrib_0', 'contrib_1'],
            index=['person_A', 'person_B']
        )
        self.x_sorted = pd.DataFrame(
            data=np.array(
                [['PhD', 34],
                 [27, 'Master']]),
            columns=['feature_0', 'feature_1'],
            index=['person_A', 'person_B']
        )
        self.var_dict = pd.DataFrame(
            data=np.array(
                [[0, 1],
                 [1, 0]]),
            columns=['feature_0', 'feature_1'],
            index=['person_A', 'person_B']
        )
        self.mask = pd.DataFrame(
            data=np.array(
                [[True, False],
                 [True, True]]),
            columns=['feature_0', 'feature_1'],
            index=['person_A', 'person_B']
        )
        model = lambda: None
        model._classes = np.array([1, 3])
        model.predict = types.MethodType(self.predict, model)
        model.predict_proba = types.MethodType(self.predict_proba, model)
        # Declare explainer object
        self.feature_dictionary = {'X1': 'Education', 'X2': 'Age'}
        self.smart_explainer = SmartExplainer(features_dict=self.feature_dictionary)
        self.smart_explainer.data = dict()
        self.smart_explainer.data['contrib_sorted'] = self.contrib_sorted
        self.smart_explainer.data['x_sorted'] = self.x_sorted
        self.smart_explainer.data['var_dict'] = self.var_dict
        self.smart_explainer.x_init = self.x_pred
        self.smart_explainer.x_pred = self.x_pred
        self.smart_explainer.postprocessing_modifications = False
        #self.smart_explainer.x_contrib_plot = self.x_contrib_plot
        self.smart_explainer.columns_dict = {i: col for i, col in enumerate(self.smart_explainer.x_pred.columns)}
        self.smart_explainer.inv_columns_dict = {v: k for k, v in self.smart_explainer.columns_dict.items()}
        self.smart_explainer.mask = self.mask
        self.smart_explainer.contributions = [self.contrib0, self.contrib1]
        self.smart_explainer.features_imp = None
        self.smart_explainer.model = model
        self.smart_explainer._case, self.smart_explainer._classes = self.smart_explainer.check_model()
        self.smart_explainer.state = self.smart_explainer.choose_state(self.smart_explainer.contributions)
        self.smart_explainer.y_pred = None
        self.smart_explainer.features_desc = self.smart_explainer.check_features_desc()

    @patch('shapash.explainer.smart_explainer.SmartExplainer.filter')
    @patch('shapash.explainer.smart_plotter.SmartPlotter.local_pred')
    def test_local_plot_1(self, local_pred, filter):
        """
        Unit test Local plot 1
        Parameters
        ----------
        select_lines : [type]
            [description]
        """
        local_pred.return_value = 12.88
        filter.return_value = None
        self.smart_explainer._case = "regression"
        output = self.smart_explainer.plot.local_plot(index='person_B')
        output_data = output.data

        feature_values = ['<b>Age :</b><br />27', '<b>Education :</b><br />Master']
        contributions = [3.6, 1.2]

        bars = []
        bars.append(go.Bar(
            x=[contributions[1]],
            y=[feature_values[1]],
            orientation='h'
        ))
        bars.append(go.Bar(
            x=[contributions[0]],
            y=[feature_values[0]],
            orientation='h'
        ))
        expected_output = go.Figure(
            data=bars,
            layout=go.Layout(yaxis=dict(type='category')))

        for part in list(zip(output_data, expected_output.data)):
            assert part[0].x == part[1].x
            assert part[0].y == part[1].y


    @patch('shapash.explainer.smart_plotter.select_lines')
    def test_local_plot_2(self, select_lines):
        """
        Unit test local plot 2
        Parameters
        ----------
        select_lines : [type]
            [description]
        """
        select_lines.return_value = [0, 1]
        self.smart_explainer._case = "regression"

        with self.assertRaises(ValueError):
            condition = ''
            output = self.smart_explainer.plot.local_plot(query=condition)
            expected_output = go.Figure(
                data=go.Bar(
                    x=[],
                    y=[],
                    orientation='h'),
                layout=go.Layout(yaxis=dict(type='category')))
            assert output == expected_output


    @patch('shapash.explainer.smart_plotter.select_lines')
    def test_local_plot_3(self, select_lines):
        """
        Unit test local plot 3
        Parameters
        ----------
        select_lines : [type]
            [description]
        """
        select_lines.return_value = []
        condition = ''
        output = self.smart_explainer.plot.local_plot(query=condition)
        expected_output = go.Figure()
        assert output.data == expected_output.data
        assert output.layout.title.text == "Local Explanation - <b>No Matching Entry</b>"

    @patch('shapash.explainer.smart_explainer.SmartExplainer.filter')
    @patch('shapash.explainer.smart_plotter.select_lines')
    @patch('shapash.explainer.smart_plotter.SmartPlotter.local_pred')
    def test_local_plot_4(self, local_pred, select_lines, filter):
        """
        Unit test local plot 4
        Parameters
        ----------
        select_lines : [type]
            [description]
        """
        filter.return_value = None
        local_pred.return_value = 0.58
        select_lines.return_value = ['B']
        index = ['A', 'B']
        x_pred = pd.DataFrame(
            data=np.array(
                [['PhD', 34],
                 ['Master', 27]]
            ),
            columns=['X1', 'X2'],
            index=index
        )
        contrib_sorted1 = pd.DataFrame(
            data=np.array(
                [[-3.4, 0.78],
                 [3.6, 1.2]]
            ),
            columns=['contrib_0', 'contrib_1'],
            index=index
        )
        contrib_sorted2 = pd.DataFrame(
            data=np.array(
                [[-0.4, 0.78],
                 [0.6, 0.2]]
            ),
            columns=['contrib_0', 'contrib_1'],
            index=index
        )
        x_sorted1 = pd.DataFrame(
            data=np.array(
                [['PhD', 34],
                 [27, 'Master']]
            ),
            columns=['feature_0', 'feature_1'],
            index=index
        )
        x_sorted2 = pd.DataFrame(
            data=np.array(
                [['PhD', 34],
                 [27, 'Master']]
            ),
            columns=['feature_0', 'feature_1'],
            index=index
        )
        var_dict1 = pd.DataFrame(
            data=np.array(
                [[0, 1],
                 [1, 0]]
            ),
            columns=['feature_0', 'feature_1'],
            index=index
        )
        var_dict2 = pd.DataFrame(
            data=np.array(
                [[0, 1],
                 [1, 0]]
            ),
            columns=['feature_0', 'feature_1'],
            index=index
        )
        mask1 = pd.DataFrame(
            data=np.array(
                [[True, False],
                 [True, True]]
            ),
            columns=['feature_0', 'feature_1'],
            index=index)
        mask2 = pd.DataFrame(
            data=np.array(
                [[True, True],
                 [True, True]]
            ),
            columns=['feature_0', 'feature_1'],
            index=index)
        feature_dictionary = {'X1': 'Education', 'X2': 'Age'}
        smart_explainer_mi = SmartExplainer(features_dict=feature_dictionary)
        smart_explainer_mi.data = dict()
        smart_explainer_mi.contributions = [contrib_sorted1, contrib_sorted2]
        smart_explainer_mi.data['contrib_sorted'] = [contrib_sorted1, contrib_sorted2]
        smart_explainer_mi.data['x_sorted'] = [x_sorted1, x_sorted2]
        smart_explainer_mi.data['var_dict'] = [var_dict1, var_dict2]
        smart_explainer_mi.x_pred = x_pred
        smart_explainer_mi.columns_dict = {i: col for i, col in enumerate(smart_explainer_mi.x_pred.columns)}
        smart_explainer_mi.mask = [mask1, mask2]
        smart_explainer_mi._case = "classification"
        smart_explainer_mi._classes = [0, 1]
        smart_explainer_mi.state = smart_explainer_mi.choose_state(smart_explainer_mi.contributions)
        condition = "index == 'B'"
        output = smart_explainer_mi.plot.local_plot(query=condition)
        feature_values = ['<b>Age :</b><br />27', '<b>Education :</b><br />Master']
        contributions = [0.6, 0.2]
        bars = []
        bars.append(go.Bar(
            x=[contributions[1]],
            y=[feature_values[1]],
            orientation='h'
        ))
        bars.append(go.Bar(
            x=[contributions[0]],
            y=[feature_values[0]],
            orientation='h'
        ))
        expected_output = go.Figure(
            data=bars,
            layout=go.Layout(yaxis=dict(type='category')))
        for part in list(zip(output.data, expected_output.data)):
            assert part[0].x == part[1].x
            assert part[0].y == part[1].y
        tit = "Local Explanation - Id: <b>B</b><span style='font-size: 12px;'><br />" + \
              "Response: <b>1</b> - Proba: <b>0.5800</b></span>"
        assert output.layout.title.text == tit

    @patch('shapash.explainer.smart_explainer.SmartExplainer.filter')
    @patch('shapash.explainer.smart_plotter.select_lines')
    @patch('shapash.explainer.smart_plotter.SmartPlotter.local_pred')
    def test_local_plot_5(self, local_pred, select_lines, filter):
        """
        Unit test local plot 5
        Parameters
        ----------
        select_lines : [type]
            [description]
        """
        local_pred.return_value = 0.58
        select_lines.return_value = ['B']
        filter.return_value = None
        index = ['A', 'B']
        x_pred = pd.DataFrame(
            data=np.array(
                [['PhD', 34],
                 ['Master', 27]]
            ),
            columns=['X1', 'X2'],
            index=index
        )
        contrib_sorted1 = pd.DataFrame(
            data=np.array(
                [[-3.4, 0.78],
                 [3.6, 1.2]]
            ),
            columns=['contrib_0', 'contrib_1'],
            index=index
        )
        contrib_sorted2 = pd.DataFrame(
            data=np.array(
                [[-0.4, 0.78],
                 [0.6, 0.2]]
            ),
            columns=['contrib_0', 'contrib_1'],
            index=index
        )
        x_sorted1 = pd.DataFrame(
            data=np.array(
                [['PhD', 34],
                 [27, 'Master']]
            ),
            columns=['feature_0', 'feature_1'],
            index=index
        )
        x_sorted2 = pd.DataFrame(
            data=np.array(
                [['PhD', 34],
                 [27, 'Master']]
            ),
            columns=['feature_0', 'feature_1'],
            index=index
        )
        var_dict1 = pd.DataFrame(
            data=np.array(
                [[0, 1],
                 [1, 0]]
            ),
            columns=['feature_0', 'feature_1'],
            index=index
        )
        var_dict2 = pd.DataFrame(
            data=np.array(
                [[0, 1],
                 [1, 0]]
            ),
            columns=['feature_0', 'feature_1'],
            index=index
        )
        mask1 = pd.DataFrame(
            data=np.array(
                [[True, False],
                 [True, True]]
            ),
            columns=['feature_0', 'feature_1'],
            index=index)
        mask2 = pd.DataFrame(
            data=np.array(
                [[False, True],
                 [False, True]]
            ),
            columns=['feature_0', 'feature_1'],
            index=index)
        mask_contrib1 = pd.DataFrame(
            data=np.array(
                [[0.0, 0.78],
                 [0.0, 1.20]]
            ),
            columns=['masked_neg', 'masked_pos'],
            index=index)
        mask_contrib2 = pd.DataFrame(
            data=np.array(
                [[0.0, 0.78],
                 [0.0, 0.20]]
            ),
            columns=['masked_neg', 'masked_pos'],
            index=index)

        feature_dictionary = {'X1': 'Education', 'X2': 'Age'}
        smart_explainer_mi = SmartExplainer(features_dict=feature_dictionary)
        smart_explainer_mi.data = dict()
        smart_explainer_mi.contributions = [contrib_sorted1, contrib_sorted2]
        smart_explainer_mi.data['contrib_sorted'] = [contrib_sorted1, contrib_sorted2]
        smart_explainer_mi.data['x_sorted'] = [x_sorted1, x_sorted2]
        smart_explainer_mi.data['var_dict'] = [var_dict1, var_dict2]
        smart_explainer_mi.x_pred = x_pred
        smart_explainer_mi.columns_dict = {i: col for i, col in enumerate(smart_explainer_mi.x_pred.columns)}
        smart_explainer_mi.mask = [mask1, mask2]
        smart_explainer_mi.masked_contributions = [mask_contrib1, mask_contrib2]
        smart_explainer_mi.mask_params = {'features_to_hide': None,
                                          'threshold': None,
                                          'positive': None,
                                          'max_contrib': 1}
        smart_explainer_mi._case = "classification"
        smart_explainer_mi._classes = [0, 1]

        smart_explainer_mi.state = smart_explainer_mi.choose_state(smart_explainer_mi.contributions)
        condition = "index == 'B'"
        output = smart_explainer_mi.plot.local_plot(query=condition)
        feature_values = ['<i>Hidden Positive Contributions</i>',
                          '<b>Education :</b><br />Master']
        contributions = [0.2, 0.2]
        bars = []
        for elem in list(zip(feature_values, contributions)):
            bars.append(go.Bar(
                x=[elem[1]],
                y=[elem[0]],
                orientation='h'
            ))
        expected_output = go.Figure(
            data=bars,
            layout=go.Layout(yaxis=dict(type='category')))

        assert len(expected_output.data) == len(output.data)
        for part in list(zip(output.data, expected_output.data)):
            assert part[0].x == part[1].x
            assert part[0].y == part[1].y
        tit = "Local Explanation - Id: <b>B</b><span style='font-size: 12px;'><br />" + \
                          "Response: <b>1</b> - Proba: <b>0.5800</b></span>"
        assert output.layout.title.text == tit

        output2 = smart_explainer_mi.plot.local_plot(query=condition, show_masked=False)
        assert len(output2.data) == 1
        assert expected_output.data[-1].x == output2.data[0].x
        smart_explainer_mi.mask_params = {'features_to_hide': None,
                                          'threshold': None,
                                          'positive': True,
                                          'max_contrib': 1}
        output3 = smart_explainer_mi.plot.local_plot(row_num=1)
        assert len(output3.data) == 2
        assert expected_output.data[-1].x == output3.data[-1].x
        assert expected_output.data[-2].x == output3.data[-2].x

    @patch('shapash.explainer.smart_explainer.SmartExplainer.filter')
    @patch('shapash.explainer.smart_plotter.select_lines')
    @patch('shapash.explainer.smart_plotter.SmartPlotter.local_pred')
    def test_local_plot_multi_index(self, local_pred, select_lines, filter):
        """
        Unit test local plot multi index
        Parameters
        ----------
        select_lines : [type]
            [description]
        """
        local_pred.return_value = 12.78
        select_lines.return_value = [('C', 'A')]
        filter.return_value = None

        index = pd.MultiIndex.from_tuples(
            [('A', 'A'), ('C', 'A')],
            names=('col1', 'col2')
        )

        x_pred_multi_index = pd.DataFrame(
            data=np.array(
                [['PhD', 34],
                 ['Master', 27]]
            ),
            columns=['X1', 'X2'],
            index=index
        )

        contrib_sorted_multi_index = pd.DataFrame(
            data=np.array(
                [[-3.4, 0.78],
                 [3.6, 1.2]]
            ),
            columns=['contrib_0', 'contrib_1'],
            index=index
        )

        x_sorted_multi_index = pd.DataFrame(
            data=np.array(
                [['PhD', 34],
                 [27, 'Master']]
            ),
            columns=['feature_0', 'feature_1'],
            index=index
        )

        var_dict_multi_index = pd.DataFrame(
            data=np.array(
                [[0, 1],
                 [1, 0]]
            ),
            columns=['feature_0', 'feature_1'],
            index=index
        )
        mask_multi_index = pd.DataFrame(
            data=np.array(
                [[True, False],
                 [True, True]]
            ),
            columns=['feature_0', 'feature_1'],
            index=index)

        feature_dictionary = {'X1': 'Education', 'X2': 'Age'}

        smart_explainer_mi = SmartExplainer(features_dict=feature_dictionary)
        smart_explainer_mi.data = dict()
        smart_explainer_mi.contributions = contrib_sorted_multi_index
        smart_explainer_mi.data['contrib_sorted'] = contrib_sorted_multi_index
        smart_explainer_mi.data['x_sorted'] = x_sorted_multi_index
        smart_explainer_mi.data['var_dict'] = var_dict_multi_index
        smart_explainer_mi.x_pred = x_pred_multi_index
        smart_explainer_mi.columns_dict = {i: col for i, col in enumerate(smart_explainer_mi.x_pred.columns)}
        smart_explainer_mi.mask = mask_multi_index
        smart_explainer_mi._case = "regression"
        smart_explainer_mi.state = smart_explainer_mi.choose_state(smart_explainer_mi.contributions)
        smart_explainer_mi.y_pred = None

        condition = "index == 'person_B'"

        output = smart_explainer_mi.plot.local_plot(query=condition)

        feature_values = ['<b>Age :</b><br />27', '<b>Education :</b><br />Master']
        contributions = [3.6, 1.2]

        bars = []
        bars.append(go.Bar(
            x=[contributions[1]],
            y=[feature_values[1]],
            orientation='h'
        ))
        bars.append(go.Bar(
            x=[contributions[0]],
            y=[feature_values[0]],
            orientation='h'
        ))
        expected_output = go.Figure(
            data=bars,
            layout=go.Layout(yaxis=dict(type='category')))
        for part in list(zip(output.data, expected_output.data)):
            assert part[0].x == part[1].x
            assert part[0].y == part[1].y

    def test_get_selection(self):
        """
        Unit test get selection
        """
        line = ['person_A']
        output = self.smart_explainer.plot.get_selection(
            line,
            self.var_dict,
            self.x_sorted,
            self.contrib_sorted
        )
        expected_output = np.array([0, 1]), np.array(['PhD', 34]), np.array([-3.4, 0.78])
        assert len(output) == 3
        assert np.array_equal(output[0], expected_output[0])
        assert np.array_equal(output[1], expected_output[1])
        assert np.array_equal(output[2], expected_output[2])

    def test_apply_mask_one_line(self):
        """
        Unit test apply mask one line
        """
        line = ['person_A']
        var_dict = np.array([0, 1])
        x_sorted = np.array(['PhD', 34])
        contrib_sorted = np.array([-3.4, 0.78])
        output = self.smart_explainer.plot.apply_mask_one_line(
            line,
            var_dict,
            x_sorted,
            contrib_sorted
        )
        expected_output = np.array([0]), np.array(['PhD']), np.array([-3.4])
        assert len(output) == 3
        assert np.array_equal(output[0], expected_output[0])
        assert np.array_equal(output[1], expected_output[1])
        assert np.array_equal(output[2], expected_output[2])

    def test_check_masked_contributions_1(self):
        """
        Unit test check masked contributions 1
        """
        line = ['person_A']
        var_dict =  ['X1', 'X2']
        x_val = ['PhD', 34]
        contrib = [-3.4, 0.78]
        var_dict, x_val, contrib = self.smart_explainer.plot.check_masked_contributions(line, var_dict, x_val, contrib)
        expected_var_dict = ['X1', 'X2']
        expected_x_val = ['PhD', 34]
        expected_contrib = [-3.4, 0.78]
        self.assertListEqual(var_dict,expected_var_dict)
        self.assertListEqual(x_val, expected_x_val)
        self.assertListEqual(contrib, expected_contrib)

    def test_check_masked_contributions_2(self):
        """
        Unit test check masked contributions 2
        """
        line = ['person_A']
        var_dict =  ['X1', 'X2']
        x_val = ['PhD', 34]
        contrib = [-3.4, 0.78]
        self.smart_explainer.masked_contributions = pd.DataFrame(
            data=[[0.0, 2.5], [0.0, 1.6]],
            columns=['masked_neg', 'masked_pos'],
            index=['person_A', 'person_B']
        )
        var_dict, x_val, contrib = self.smart_explainer.plot.check_masked_contributions(line, var_dict, x_val, contrib)
        expected_var_dict = ['X1', 'X2', 'Hidden Positive Contributions']
        expected_x_val = ['PhD', 34, '' ]
        expected_contrib = [-3.4, 0.78, 2.5]
        self.assertListEqual(var_dict,expected_var_dict)
        self.assertListEqual(x_val, expected_x_val)
        self.assertListEqual(contrib, expected_contrib)

    def test_plot_bar_chart_1(self):
        """
        Unit test plot bar chart 1
        """
        var_dict = ['X1', 'X2']
        x_val = ['PhD', 34]
        contributions = [-3.4, 0.78]
        bars = []
        for num, elem in enumerate(var_dict):
            bars.append(go.Bar(
                x=[contributions[num]],
                y=['<b>{} :</b><br />{}'.format(elem, x_val[num])],
                orientation='h'
            ))
        expected_output_fig = go.Figure(
            data=bars,
            layout=go.Layout(yaxis=dict(type='category')))
        self.smart_explainer._case = "regression"
        fig_output = self.smart_explainer.plot.plot_bar_chart('ind', var_dict, x_val, contributions)
        for part in list(zip(fig_output.data, expected_output_fig.data)):
            assert part[0].x == part[1].x
            assert part[0].y == part[1].y

    def test_plot_bar_chart_2(self):
        """
        Unit test plot bar chart 2
        """
        var_dict = ['X1', 'X2', 'Hidden Positive Contributions']
        x_val = ['PhD', 34, '']
        order = [3, 1, 2]
        contributions = [-3.4, 0.78, 2.5]
        ylabel = ['<b>X1 :</b><br />PhD', '<b>X2 :</b><br />34', '<i>Hidden Positive Contributions</i>']
        self.smart_explainer.masked_contributions = pd.DataFrame()
        bars = []
        comblist = list(zip(order, contributions, ylabel))
        comblist.sort(reverse=True)
        for elem in comblist:
            bars.append(go.Bar(
                x=[elem[1]],
                y=[elem[2]],
                orientation='h'
            ))
        expected_output_fig = go.Figure(
            data=bars,
            layout=go.Layout(yaxis=dict(type='category')))

        self.smart_explainer._case = "regression"
        fig_output = self.smart_explainer.plot.plot_bar_chart('ind', var_dict, x_val, contributions)
        for part in list(zip(fig_output.data, expected_output_fig.data)):
            assert part[0].x == part[1].x
            assert part[0].y == part[1].y

    def test_contribution_plot_1(self):
        """
        Classification
        """
        col = 'X1'
        output = self.smart_explainer.plot.contribution_plot(col, violin_maxf=0, proba=False)
        expected_output = go.Scatter(x=self.smart_explainer.x_pred[col],
                                     y=self.smart_explainer.contributions[-1][col],
                                     mode='markers',
                                     hovertext=[f"Id: {x}<br />" for x in self.smart_explainer.x_pred.index])
        assert np.array_equal(output.data[0].x, expected_output.x)
        assert np.array_equal(output.data[0].y, expected_output.y)
        assert len(np.unique(output.data[0].marker.color)) == 1
        assert output.layout.xaxis.title.text == self.smart_explainer.features_dict[col]

    def test_contribution_plot_2(self):
        """
        Regression
        """
        col = 'X2'
        xpl = self.smart_explainer
        xpl.contributions = self.contrib1
        xpl._case = "regression"
        xpl.state = xpl.choose_state(xpl.contributions)
        output = xpl.plot.contribution_plot(col, violin_maxf=0)
        expected_output = go.Scatter(x=xpl.x_pred[col],
                                     y=xpl.contributions[col],
                                     mode='markers',
                                     hovertext=[f"Id: {x}<br />" for x in xpl.x_pred.index])

        assert np.array_equal(output.data[0].x, expected_output.x)
        assert np.array_equal(output.data[0].y, expected_output.y)
        assert np.array_equal(output.data[0].hovertext, expected_output.hovertext)
        assert len(np.unique(output.data[0].marker.color)) == 1
        assert output.layout.xaxis.title.text == self.smart_explainer.features_dict[col]

    def test_contribution_plot_3(self):
        """
        Color Plot classification
        """
        col = 'X2'
        xpl = self.smart_explainer
        xpl.y_pred = pd.DataFrame([0, 1], columns=['pred'], index=xpl.x_pred.index)
        xpl._classes = [0, 1]
        output = xpl.plot.contribution_plot(col, violin_maxf=0, proba=False)
        expected_output = go.Scatter(x=xpl.x_pred[col],
                                     y=xpl.contributions[-1][col],
                                     mode='markers',
                                     hovertext=[f"Id: {x}<br />Predict: {y}" for x, y in zip(xpl.x_pred.index, xpl.y_pred.iloc[:, 0].tolist())])

        assert np.array_equal(output.data[0].x, expected_output.x)
        assert np.array_equal(output.data[0].y, expected_output.y)
        assert np.array_equal(output.data[0].hovertext, expected_output.hovertext)
        assert len(np.unique(output.data[0].marker.color)) == 2
        assert output.layout.xaxis.title.text == self.smart_explainer.features_dict[col]

    def test_contribution_plot_4(self):
        """
        Regression Color Plot
        """
        col = 'X2'
        xpl = self.smart_explainer
        xpl.contributions = self.contrib1
        xpl._case = "regression"
        xpl.state = xpl.choose_state(xpl.contributions)
        xpl.y_pred = pd.DataFrame([0.46989877093, 12.749302948], columns=['pred'], index=xpl.x_pred.index)
        output = xpl.plot.contribution_plot(col, violin_maxf=0)
        expected_output = go.Scatter(x=xpl.x_pred[col],
                                     y=xpl.contributions[col],
                                     mode='markers',
                                     hovertext=[f"Id: {x}<br />Predict: {round(y,3)}" for x, y in zip(xpl.x_pred.index, xpl.y_pred.iloc[:, 0].tolist())])

        assert np.array_equal(output.data[0].x, expected_output.x)
        assert np.array_equal(output.data[0].y, expected_output.y)
        assert len(np.unique(output.data[0].marker.color)) >= 2
        assert np.array_equal(output.data[0].hovertext, expected_output.hovertext)
        assert output.layout.xaxis.title.text == self.smart_explainer.features_dict[col]

    def test_contribution_plot_5(self):
        """
        Regression Color Plot with pred
        """
        col = 'X2'
        xpl = self.smart_explainer
        xpl.contributions = pd.concat([self.contrib1]*10, ignore_index=True)
        xpl._case = "regression"
        xpl.state = xpl.choose_state(xpl.contributions)
        xpl.x_pred = pd.concat([xpl.x_pred]*10, ignore_index=True)
        xpl.postprocessing_modifications = False
        xpl.y_pred = pd.concat([pd.DataFrame([0.46989877093, 12.749302948])]*10, ignore_index=True)
        output = xpl.plot.contribution_plot(col)
        np_hv = np.array([f"Id: {x}<br />Predict: {round(y,2)}" for x, y in zip(xpl.x_pred.index, xpl.y_pred.iloc[:, 0].tolist())])
        assert len(output.data) == 3
        assert output.data[0].type == 'violin'
        assert output.data[-1].type == 'scatter'
        assert len(np.unique(output.data[-1].marker.color)) >= 2
        assert np.array_equal(output.data[-1].hovertext, np_hv)
        assert output.layout.xaxis.title.text == xpl.features_dict[col]

    def test_contribution_plot_6(self):
        """
        Regression without pred
        """
        col = 'X2'
        xpl = self.smart_explainer
        xpl.contributions = pd.concat([self.contrib1]*10, ignore_index=True)
        xpl._case = "regression"
        xpl.state = xpl.choose_state(xpl.contributions)
        xpl.x_pred = pd.concat([xpl.x_pred]*10, ignore_index=True)
        xpl.postprocessing_modifications = False
        output = xpl.plot.contribution_plot(col)
        np_hv = [f"Id: {x}" for x in xpl.x_pred.index]
        np_hv.sort()
        annot_list = []
        for data_plot in output.data:
            annot_list.extend(data_plot.hovertext.tolist())
        annot_list.sort()
        assert len(output.data) == 2
        for elem in output.data:
            assert elem.type == 'violin'
        assert output.data[-1].marker.color == output.data[-2].marker.color
        self.assertListEqual(annot_list, np_hv)
        assert output.layout.xaxis.title.text == xpl.features_dict[col]

    def test_contribution_plot_7(self):
        """
        Classification without pred
        """
        col = 'X1'
        xpl = self.smart_explainer
        xpl.contributions[0] = pd.concat([xpl.contributions[0]] * 10, ignore_index=True)
        xpl.contributions[1] = pd.concat([xpl.contributions[1]] * 10, ignore_index=True)
        xpl.x_pred = pd.concat([xpl.x_pred] * 10, ignore_index=True)
        xpl.postprocessing_modifications = False
        np_hv = [f"Id: {x}" for x in xpl.x_pred.index]
        np_hv.sort()
        output = xpl.plot.contribution_plot(col, proba=False)
        annot_list = []
        for data_plot in output.data:
            annot_list.extend(data_plot.hovertext.tolist())
        annot_list.sort()
        assert len(output.data) == 2
        for elem in output.data:
            assert elem.type == 'violin'
        assert output.data[-1].marker.color == output.data[-2].marker.color
        self.assertListEqual(annot_list, np_hv)
        assert output.layout.xaxis.title.text == xpl.features_dict[col]

    def test_contribution_plot_8(self):
        """
        Classification with pred
        """
        col = 'X1'
        xpl = self.smart_explainer
        xpl.x_pred = pd.concat([xpl.x_pred] * 10, ignore_index=True)
        xpl.x_pred.index = [i for i in range(xpl.x_pred.shape[0])]
        xpl.postprocessing_modifications = False
        xpl.contributions[0] = pd.concat([xpl.contributions[0]] * 10, ignore_index=True)
        xpl.contributions[1] = pd.concat([xpl.contributions[1]] * 10, ignore_index=True)
        xpl.contributions[0].index = xpl.x_pred.index
        xpl.contributions[1].index = xpl.x_pred.index
        xpl.y_pred = pd.DataFrame([0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                                  columns=['pred'], index=xpl.x_pred.index)
        model = lambda: None
        model.classes_ = np.array([0, 1])
        xpl.model = model
        np_hv = [f"Id: {x}<br />Predict: {y}" for x, y in zip(xpl.x_pred.index, xpl.y_pred.iloc[:, 0].tolist())]
        np_hv.sort()
        output = xpl.plot.contribution_plot(col)
        annot_list = []
        for data_plot in output.data:
            annot_list.extend(data_plot.hovertext.tolist())
        annot_list.sort()
        assert len(output.data) == 4
        for elem in output.data:
            assert elem.type == 'violin'
        assert output.data[0].side == 'negative'
        assert output.data[1].side == 'positive'
        assert output.data[-1].line.color == output.data[-3].line.color
        assert output.data[-1].line.color != output.data[-2].line.color
        self.assertListEqual(annot_list, np_hv)
        assert output.layout.xaxis.title.text == xpl.features_dict[col]

    def test_contribution_plot_9(self):
        """
        Classification with pred and sampling
        """
        col = 'X1'
        xpl = self.smart_explainer
        xpl.x_pred = pd.concat([xpl.x_pred] * 20, ignore_index=True)
        xpl.x_pred.index = [i for i in range(xpl.x_pred.shape[0])]
        xpl.postprocessing_modifications = False
        xpl.contributions[0] = pd.concat([xpl.contributions[0]] * 20, ignore_index=True)
        xpl.contributions[1] = pd.concat([xpl.contributions[1]] * 20, ignore_index=True)
        xpl.contributions[0].index = xpl.x_pred.index
        xpl.contributions[1].index = xpl.x_pred.index
        xpl.y_pred = pd.DataFrame([0, 1, 1, 0, 0]*8,
                                  columns=['pred'], index=xpl.x_pred.index)
        model = lambda: None
        model.classes_ = np.array([0, 1])
        xpl.model = model
        output = xpl.plot.contribution_plot(col, max_points=39)
        assert len(output.data) == 4
        for elem in output.data:
            assert elem.type == 'violin'
        assert output.data[0].side == 'negative'
        assert output.data[1].side == 'positive'
        assert output.data[-1].line.color == output.data[-3].line.color
        assert output.data[-1].line.color != output.data[-2].line.color
        assert output.layout.xaxis.title.text == xpl.features_dict[col]
        total_row = 0
        for data in output.data:
            total_row = total_row + data.x.shape[0]
        assert total_row == 39
        expected_title = "<b>Education</b> - Feature Contribution<span style='font-size: 12px;'><br />Response: <b>3</b>" \
                + " - Length of random Subset : 39 (98%)</span>"
        assert output.layout.title['text'] == expected_title

    def test_contribution_plot_10(self):
        """
        Regression with pred and subset
        """
        col = 'X2'
        xpl = self.smart_explainer
        xpl.x_pred = pd.concat([xpl.x_pred] * 4, ignore_index=True)
        xpl.x_pred.index = [i for i in range(xpl.x_pred.shape[0])]
        xpl.postprocessing_modifications = False
        xpl.contributions = pd.concat([self.contrib1] * 4, ignore_index=True)
        xpl._case = "regression"
        xpl.state = xpl.choose_state(xpl.contributions)
        xpl.y_pred = pd.DataFrame([0.46989877093, 12.749302948]*4, columns=['pred'], index=xpl.x_pred.index)
        subset = [1, 2, 6, 7]
        output = xpl.plot.contribution_plot(col, selection=subset, violin_maxf=0)
        expected_output = go.Scatter(x=xpl.x_pred[col].loc[subset],
                                     y=xpl.contributions[col].loc[subset],
                                     mode='markers',
                                     hovertext=[f"Id: {x}<br />Predict: {y:.2f}"
                                                for x, y in zip(xpl.x_pred.loc[subset].index,
                                                                xpl.y_pred.loc[subset].iloc[:, 0].tolist())])

        assert np.array_equal(output.data[0].x, expected_output.x)
        assert np.array_equal(output.data[0].y, expected_output.y)
        assert len(np.unique(output.data[0].marker.color)) >= 2
        assert np.array_equal(output.data[0].hovertext, expected_output.hovertext)
        assert output.layout.xaxis.title.text == self.smart_explainer.features_dict[col]
        expected_title = "<b>Age</b> - Feature Contribution<span style='font-size: 12px;'><br />" \
            + "Length of user-defined Subset : 4 (50%)</span>"
        assert output.layout.title['text'] == expected_title

    def test_contribution_plot_11(self):
        """
        classification with proba
        """
        col = 'X1'
        xpl =self.smart_explainer
        xpl.proba_values = pd.DataFrame(
            data = np.array(
                [[0.4, 0.6],
                [0.3, 0.7]]),
            columns = ['class_1','class_2'],
            index = xpl.x_init.index.values)
        output = self.smart_explainer.plot.contribution_plot(col)
        assert str(type(output.data[-1])) == "<class 'plotly.graph_objs._scatter.Scatter'>"
        self.assertListEqual(list(output.data[-1]['marker']['color']), [0.6, 0.7])
        self.assertListEqual(list(output.data[-1]['y']), [-0.3, 4.7])

    def test_plot_features_import_1(self):
        """
        Unit test plot features import 1
        """
        serie1 = pd.Series([0.131, 0.51], index=['col1', 'col2'])
        output = self.smart_explainer.plot.plot_features_import(serie1)
        data = go.Bar(
            x=serie1,
            y=serie1.index,
            name='Global',
            orientation='h'
        )

        expected_output = go.Figure(data=data)
        assert np.array_equal(output.data[0].x, expected_output.data[0].x)
        assert np.array_equal(output.data[0].y, expected_output.data[0].y)
        assert output.data[0].name == expected_output.data[0].name
        assert output.data[0].orientation == expected_output.data[0].orientation

    def test_plot_features_import_2(self):
        """
        Unit test plot features import 2
        """
        serie1 = pd.Series([0.131, 0.51], index=['col1', 'col2'])
        serie2 = pd.Series([0.33, 0.11], index=['col1', 'col2'])
        output = self.smart_explainer.plot.plot_features_import(serie1, serie2)
        data1 = go.Bar(
            x=serie1,
            y=serie1.index,
            name='Global',
            orientation='h'
        )
        data2 = go.Bar(
            x=serie2,
            y=serie2.index,
            name='Subset',
            orientation='h'
        )
        expected_output = go.Figure(data=[data2, data1])
        assert np.array_equal(output.data[0].x, expected_output.data[0].x)
        assert np.array_equal(output.data[0].y, expected_output.data[0].y)
        assert output.data[0].name == expected_output.data[0].name
        assert output.data[0].orientation == expected_output.data[0].orientation
        assert np.array_equal(output.data[1].x, expected_output.data[1].x)
        assert np.array_equal(output.data[1].y, expected_output.data[1].y)
        assert output.data[1].name == expected_output.data[1].name
        assert output.data[1].orientation == expected_output.data[1].orientation

    def test_features_importance_1(self):
        """
        Unit test features importance 1
        """
        xpl = self.smart_explainer
        output = xpl.plot.features_importance(selection=['person_A', 'person_B'])

        data1 = go.Bar(
            x=np.array([0.2296, 0.7704]),
            y=np.array(['Age', 'Education']),
            name='Subset',
            orientation='h')

        data2 = go.Bar(
            x=np.array([0.2296, 0.7704]),
            y=np.array(['Age', 'Education']),
            name='Global',
            orientation='h')

        expected_output = go.Figure(data=[data1, data2])

        assert np.array_equal(output.data[0].x, expected_output.data[0].x)
        assert np.array_equal(output.data[0].y, expected_output.data[0].y)
        assert output.data[0].name == expected_output.data[0].name
        assert output.data[0].orientation == expected_output.data[0].orientation
        assert np.array_equal(output.data[1].x, expected_output.data[1].x)
        assert np.array_equal(output.data[1].y, expected_output.data[1].y)
        assert output.data[1].name == expected_output.data[1].name
        assert output.data[1].orientation == expected_output.data[1].orientation

    def test_features_importance_2(self):
        """
        Unit test features importance 2
        """
        xpl = self.smart_explainer
        #regression
        xpl.contributions = self.contrib1
        xpl._case = "regression"
        xpl.state = xpl.choose_state(xpl.contributions)
        output = xpl.plot.features_importance(selection=['person_A', 'person_B'])

        data1 = go.Bar(
            x=np.array([0.2296, 0.7704]),
            y=np.array(['Age', 'Education']),
            name='Subset',
            orientation='h')

        data2 = go.Bar(
            x=np.array([0.2296, 0.7704]),
            y=np.array(['Age', 'Education']),
            name='Global',
            orientation='h')

        expected_output = go.Figure(data=[data1, data2])

        assert np.array_equal(output.data[0].x, expected_output.data[0].x)
        assert np.array_equal(output.data[0].y, expected_output.data[0].y)
        assert output.data[0].name == expected_output.data[0].name
        assert output.data[0].orientation == expected_output.data[0].orientation
        assert np.array_equal(output.data[1].x, expected_output.data[1].x)
        assert np.array_equal(output.data[1].y, expected_output.data[1].y)
        assert output.data[1].name == expected_output.data[1].name
        assert output.data[1].orientation == expected_output.data[1].orientation

    def test_local_pred_1(self):
        xpl = self.smart_explainer
        output = xpl.plot.local_pred('person_A',label=0)
        assert output == 0.5

    def test_plot_line_comparison_1(self):
        """
        Unit test 1 for plot_line_comparison
        """
        xpl = self.smart_explainer
        index = ['person_A', 'person_B']
        data = pd.DataFrame(
            data=np.array(
                [['PhD', 34],
                 ['Master', 27]]
            ),
            columns=['X1', 'X2'],
            index=index
        )
        features_dict = {'X1': 'X1', 'X2': 'X2'}
        xpl.inv_features_dict = {'X1': 'X1', 'X2': 'X2'}
        colors = ['rgba(244, 192, 0, 1.0)', 'rgba(74, 99, 138, 0.7)']
        var_dict = ['X1', 'X2']
        contributions = [[-3.4, 0.78], [1.2, 3.6]]
        title = 'Compare plot - index : <b>person_A</b> ; <b>person_B</b>'
        predictions = [data.loc[id] for id in index]

        fig = list()
        for i in range(2):
            fig.append(go.Scatter(
                x=[contributions[0][i], contributions[1][i]],
                y=['<b>' + feat + '</b>' for feat in var_dict],
                mode='lines+markers',
                name=f'Id: <b>{index[i]}</b>',
                hovertext=[f'Id: <b>{index[i]}</b><br /><b>X1</b> <br />Contribution: {contributions[0][i]:.4f} <br />'
                           + f'Value: {data.iloc[i][0]}',
                           f'Id: <b>{index[i]}</b><br /><b>X2</b> <br />Contribution: {contributions[1][i]:.4f} <br />'
                           + f'Value: {data.iloc[i][1]}'
                           ],
                marker={'color': colors[i]}
                )
            )
        expected_output = go.Figure(data=fig)
        output = xpl.plot.plot_line_comparison(['person_A', 'person_B'], var_dict, contributions,
                                               predictions=predictions, dict_features=features_dict)

        for i in range(2):
            assert np.array_equal(output.data[i]['x'], expected_output.data[i]['x'])
            assert np.array_equal(output.data[i]['y'], expected_output.data[i]['y'])
            assert output.data[i].name == expected_output.data[i].name
            assert output.data[i].hovertext == expected_output.data[i].hovertext
            assert output.data[i].marker == expected_output.data[i].marker
        assert title == output.layout.title.text

    def test_plot_line_comparison_2(self):
        """
        Unit test 2 for plot_line_comparison
        """
        xpl = self.smart_explainer
        index = ['person_A', 'person_B']
        data = pd.DataFrame(
            data=np.array(
                [['PhD', 34],
                 ['Master', 27]]
            ),
            columns=['X1', 'X2'],
            index=index
        )
        xpl.inv_features_dict = {'X1': 'X1', 'X2': 'X2'}
        var_dict = ['X1', 'X2']
        contributions = [[-3.4, 0.78], [1.2, 3.6]]
        subtitle = 'This is a good test'
        title = 'Compare plot - index : <b>person_A</b> ; <b>person_B</b>' \
                + "<span style='font-size: 12px;'><br />This is a good test</span>"
        predictions = [data.loc[id] for id in index]

        output = xpl.plot.plot_line_comparison(index, var_dict, contributions,
                                               subtitle=subtitle, predictions=predictions,
                                               dict_features=xpl.inv_features_dict)

        assert title == output.layout.title.text

    def test_compare_plot_1(self):
        """
        Unit test 1 for compare_plot
        """
        xpl = self.smart_explainer
        xpl.contributions = pd.DataFrame(
            data=[[-3.4, 0.78], [1.2, 3.6]],
            index=['person_A', 'person_B'],
            columns=['X1', 'X2']
        )
        xpl.inv_features_dict = {'Education': 'X1', 'Age': 'X2'}
        xpl._case = "regression"
        output = xpl.plot.compare_plot(row_num=[1], show_predict=False)
        title = "Compare plot - index : <b>person_B</b><span style='font-size: 12px;'><br /></span>"
        data = [go.Scatter(
            x=[1.2, 3.6],
            y=['<b>Education</b>', '<b>Age</b>'],
            name='Id: <b>person_B</b>',
            hovertext=['Id: <b>person_B</b><br /><b>Education</b> <br />Contribution: 1.2000 <br />Value: Master',
                       'Id: <b>person_B</b><br /><b>Age</b> <br />Contribution: 3.6000 <br />Value: 27']
            )
        ]
        expected_output = go.Figure(data=data)
        assert np.array_equal(expected_output.data[0].x, output.data[0].x)
        assert np.array_equal(expected_output.data[0].y, output.data[0].y)
        assert np.array_equal(expected_output.data[0].hovertext, output.data[0].hovertext)
        assert expected_output.data[0].name == output.data[0].name
        assert title == output.layout.title.text

    def test_compare_plot_2(self):
        """
        Unit test 2 for compare_plot
        """
        xpl = self.smart_explainer
        xpl.inv_features_dict = {'Education': 'X1', 'Age': 'X2'}
        index = ['person_A', 'person_B']
        contributions = [[-3.4, 0.78], [1.2, 3.6]]
        xpl.contributions = pd.DataFrame(
            data=contributions,
            index=index,
            columns=['X1', 'X2']
        )
        data = np.array(
                    [['PhD', 34],
                     ['Master', 27]]
                )
        xpl._case = "regression"
        output = xpl.plot.compare_plot(index=index, show_predict=True)
        title_and_subtitle = "Compare plot - index : <b>person_A</b> ;" \
                " <b>person_B</b><span style='font-size: 12px;'><br />" \
                "Predictions: person_A: <b>1</b> ; person_B: <b>1</b></span>"
        fig = list()
        for i in range(2):
            fig.append(go.Scatter(
                x=contributions[i][::-1],
                y=['<b>Age</b>', '<b>Education</b>'],
                name=f'Id: <b>{index[i]}</b>',
                hovertext=[
                    f'Id: <b>{index[i]}</b><br /><b>Age</b> <br />Contribution: {contributions[i][1]:.4f}'
                    f' <br />Value: {data[i][1]}',
                    f'Id: <b>{index[i]}</b><br /><b>Education</b> <br />Contribution: {contributions[i][0]:.4f}'
                    f' <br />Value: {data[i][0]}'
                ]
            )
            )

        expected_output = go.Figure(data=fig)
        for i in range(2):
            assert np.array_equal(expected_output.data[i].x, output.data[i].x)
            assert np.array_equal(expected_output.data[i].y, output.data[i].y)
            assert np.array_equal(expected_output.data[i].hovertext, output.data[i].hovertext)
            assert expected_output.data[i].name == output.data[i].name
        assert title_and_subtitle == output.layout.title.text

    def test_compare_plot_3(self):
        """
        Unit test 3 for compare_plot classification
        """
        index = ['A', 'B']
        x_pred = pd.DataFrame(
            data=np.array(
                [['PhD', 34],
                 ['Master', 27]]
            ),
            columns=['X1', 'X2'],
            index=index
        )
        contributions1 = pd.DataFrame(
            data=np.array(
                [[-3.4, 0.78],
                 [1.2, 3.6]]
            ),
            columns=['X1', 'X2'],
            index=index
        )
        contributions2 = pd.DataFrame(
            data=np.array(
                [[-0.4, 0.78],
                 [0.2, 0.6]]
            ),
            columns=['X1', 'X2'],
            index=index
        )
        feature_dictionary = {'X1': 'Education', 'X2': 'Age'}
        smart_explainer_mi = SmartExplainer(features_dict=feature_dictionary)
        smart_explainer_mi.contributions = [
            pd.DataFrame(
                data=contributions1,
                index=index,
                columns=['X1', 'X2']
            ),
            pd.DataFrame(
                data=contributions2,
                index=index,
                columns=['X1', 'X2']
            )
        ]
        smart_explainer_mi.inv_features_dict = {'Education': 'X1', 'Age': 'X2'}
        smart_explainer_mi.data = dict()
        smart_explainer_mi.x_pred = x_pred
        smart_explainer_mi.columns_dict = {i: col for i, col in enumerate(smart_explainer_mi.x_pred.columns)}
        smart_explainer_mi._case = "classification"
        smart_explainer_mi._classes = [0, 1]
        smart_explainer_mi.model = 'predict_proba'

        output_label0 = smart_explainer_mi.plot.compare_plot(index=['A', 'B'], label=0, show_predict=False)
        output_label1 = smart_explainer_mi.plot.compare_plot(index=['A', 'B'], show_predict=False)

        title_0 = "Compare plot - index : <b>A</b> ; <b>B</b><span style='font-size: 12px;'><br /></span>"
        title_1 = "Compare plot - index : <b>A</b> ; <b>B</b><span style='font-size: 12px;'><br /></span>"

        fig_0 = list()
        x0 = contributions1.to_numpy()
        x0.sort(axis=1)
        for i in range(2):
            fig_0.append(go.Scatter(
                x=x0[i][::-1],
                y=['<b>Age</b>', '<b>Education</b>'],
                name=f'Id: <b>{index[i]}</b>',
                hovertext=[
                    f'Id: <b>{index[i]}</b><br /><b>Age</b> <br />Contribution: {x0[i][1]:.4f}'
                    f' <br />Value: {x_pred.to_numpy()[i][1]}',
                    f'Id: <b>{index[i]}</b><br /><b>Education</b> <br />Contribution: {x0[i][0]:.4f}'
                    f' <br />Value: {x_pred.to_numpy()[i][0]}'
                ]
                )
            )

        fig_1 = list()
        x1 = contributions2.to_numpy()
        x1.sort(axis=1)
        for i in range(2):
            fig_1.append(go.Scatter(
                x=x1[i][::-1],
                y=['<b>Age</b>', '<b>Education</b>'],
                name=f'Id: <b>{index[i]}</b>',
                hovertext=[
                    f'Id: <b>{index[i]}</b><br /><b>Age</b> <br />Contribution: {x1[i][1]:.4f}'
                    f' <br />Value: {x_pred.to_numpy()[i][1]}',
                    f'Id: <b>{index[i]}</b><br /><b>Education</b> <br />Contribution: {x1[i][0]:.4f}'
                    f' <br />Value: {x_pred.to_numpy()[i][0]}'
                ]
                )
            )

        expected_output_0 = go.Figure(data=fig_0)
        expected_output_1 = go.Figure(data=fig_1)

        assert title_0 == output_label0.layout.title.text
        assert title_1 == output_label1.layout.title.text
        for i in range(2):
            assert np.array_equal(output_label1.data[i].x, expected_output_1.data[i].x)
            assert np.array_equal(output_label1.data[i].y, expected_output_1.data[i].y)
            assert np.array_equal(output_label1.data[i].hovertext, expected_output_1.data[i].hovertext)

            assert np.array_equal(output_label0.data[i].x, expected_output_0.data[i].x)
            assert np.array_equal(output_label0.data[i].y, expected_output_0.data[i].y)
            assert np.array_equal(output_label0.data[i].hovertext, expected_output_0.data[i].hovertext)
