import plotly.offline as offline
import plotly.graph_objs as go
import matplotlib.pyplot as plt


class Projection():
    """Contains all important attributes and actions related to projection."""

    def __init__(self, x_positions, y_positions, data, model=None):
        """Initialization of projection class.

        Parameters
        ----------
        x_positions : list of x-positions of words
        y_positions : list of y-positions of words
        data : DataFrame
            All data (words) in the same order as their respective positions.
        model : model used
        """
        self.x_positions = x_positions
        self.y_positions = y_positions
        self.data = data
        self.model = model

    def simple_scatterplot(self, figsize=(30,20), export=False, title='default', save_path='visualizations/vis.png'):
        """Matplotlib simple scatterplot without labeling."""

        plt.gcf().set_size_inches(figsize[0], figsize[1])
        plt.title(title)
        plt.xlabel('x_positions')
        plt.ylabel('y_positions')
        plt.scatter(self.x_positions, self.y_positions)
        if export and save_path.startswith('visualizations/'):
            plt.savefig('visualizations/matplotlib/%s.png' % title)
        plt.show()

    def matplotlib_plot_with_manual_labels(self, figsize=(30,20), annotate=True, export=True, markersize=20,
                                           save_path='visualizations/vis.png', title='default', titlesize=25, unlabeled_markersize):
        """Creates visualization by Matplotlib.

        Words are colored and divided into groups (according to manual labeling), which you can filter.

        Parameters
        ----------
        figsize : tuple of (int, int)
            Defines size of the figure
        annotate : bool
            True if you want annotations in your visualization, False if you want just a scatter plot
        export : bool
            True if you want to save your visualization
        markersize: int
            Size of the marker for labeled items
        save_path : str
            Path where the visualization will be saved, if the export value is True
        title : str
            Title of the visualization
        titlesize : int
            Size of the title in Figure.
        """
        if 'manual_label' not in self.data.columns:
            raise ValueError("Data are not labeled into groups.")
        colors = ['blue', 'green', 'red', 'darkcyan', 'magenta', 'orange', 'darkgray', 'purple']
        self.data.reset_index(drop=True, inplace=True)
        for i, row in self.data.iterrows():
            if row['manual_label'] != 0:
                plt.scatter(self.x_positions[i], self.y_positions[i], c=colors[row['manual_label'] - 1], s=markersize)
                if annotate:
                    plt.annotate(row['question'], xy=(self.x_positions[i]+0.5, self.y_positions[i]),
                                 color=colors[row['manual_label'] - 1])
            else:
                plt.scatter(self.x_positions[i], self.y_positions[i], c='black', s=unlabeled_markersize)
                if annotate:
                    plt.annotate(row['question'], xy=(self.x_positions[i], self.y_positions[i]), color='black')
        plt.gcf().set_size_inches(figsize[0], figsize[1])
        plt.title(title, fontsize=titlesize)
        if export and save_path.startswith('visualizations/'):
            plt.savefig(save_path)
        plt.show()

    def plotly_with_manual_labels(self, annotate=True, save_path='visualizations/vis.html', title='default', unlabeled_markersize):
        """Creates visualization by Plotly.

        Words are colored and divided into groups (according to manual labeling), which you can filter.

        Parameters
        ----------
        annotate : bool
            True if you want annotations in your visualization, False if you want just a scatter plot
        save_path : str
            Path where the visualization will be saved
        title : str
            Title of the visualization
        """
        plotly_data = self.data.copy()
        plotly_data['x_position'] = self.x_positions
        plotly_data['y_position'] = self.y_positions

        manual_labels = plotly_data['manual_label'].unique()
        traces = []
        
        colors = { 0: 'rgb(0,0,0)', 1: 'rgb(0,0,255)', 2: 'rgb(50,205,50)', 3: 'rgb(255,0,0)', 
                  4: 'rgb(0,139,139)', 5: 'rgb(255,0,255)', 6: 'rgb(255,165,0)', 7: 'rgb(128,128,128)', 8: 'rgb(128,0,128)' }

        for label in manual_labels:
            current_trace = plotly_data[plotly_data['manual_label'] == label]
            trace = go.Scatter(
                x=current_trace['x_position'],
                y=current_trace['y_position'],
                mode='markers+text' if annotate else 'markers',
                name='GROUP {0}'.format(label) if label != 0 else 'UNLABELED DATA',
                text=current_trace['question'],
                hoverinfo='text',
                textposition='middle right',
                marker = dict(
                    size = 5 if label == 0 else 15,
                    color = colors[label],
                )
            )
            traces.append(trace)

        layout = go.Layout(
            title=title,
            titlefont={'family': 'Arial', 'size': 20},
            showlegend=True,
            xaxis=dict(
                autorange=True,
                showgrid=False,
                zeroline=False,
                showline=True,
                autotick=True,
                ticks="outside",
                showticklabels=True
            ),
            yaxis=dict(
                autorange=True,
                showgrid=False,
                zeroline=False,
                showline=True,
                autotick=True,
                ticks="outside",
                showticklabels=True
            ),
            hovermode='closest'
        )
        fig = go.Figure(data=traces, layout=layout)
        plot_url = offline.plot(fig, filename=save_path)
