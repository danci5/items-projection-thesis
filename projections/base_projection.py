import matplotlib.pyplot as plt


class Projection():
    
    def __init__(self, x_positions, y_positions, data, data_name=None, model=None):
        self.x_positions = x_positions
        self.y_positions = y_positions
        self.data = data
        self.data_name = data_name
        self.model = model

    def simple_scatterplot(figsize=(40,30),export=False, title='default', save_path='visualizations/vis.png'):
        """Matplotlib simple scatterplot from dataframe for visualization."""

        plt.gcf().set_size_inches(figsize[0], figsize[1])
        plt.title(title)
        plt.xlabel('x_positions')
        plt.ylabel('y_positions')
        plt.scatter(self.x_positions, self.y_positions)
        if export and save_path.startswith('visualizations/'):
            plt.savefig('visualizations/matplotlib/%s.png' % title)
        plt.show()

    def matplotlib_plot_with_manual_labels(self, figsize=(40,30), annotate=True, export=True, save_path='visualizations/vis.png'):
        """Scatterplot with annotations colored based on the item's manual label.
        You can set your preferred figure size and if you want to export the visualization.
        """

        # TODO: generate list containing random colors, not hardcoded
        if 'manual_label' not in self.data.columns:
            raise ValueError("Data are not labeled into groups.")
        colors = ['blue', 'green', 'red', 'darkcyan', 'magenta', 'yellow', 'darkgray', 'purple']
        self.data.reset_index(drop=True, inplace=True)
        for i, row in self.data.iterrows():
            if row['manual_label'] != 0:
                plt.scatter(self.x_positions[i], self.y_positions[i], c=colors[row['manual_label'] - 1])
                if annotate:
                    plt.annotate(row['question'], xy=(self.x_positions[i], self.y_positions[i]),
                                 color=colors[row['manual_label'] - 1])
            else:
                plt.scatter(self.x_positions[i], self.y_positions[i], c='black')
                if annotate:
                    plt.annotate(row['question'], xy=(self.x_positions[i], self.y_positions[i]), color='black')
        plt.gcf().set_size_inches(figsize[0], figsize[1])
        plt.title("Used: {0}, {1}".format(self.data_name, self.model), color='black')
        if export and save_path.startswith('visualizations/'):
            plt.savefig(save_path)
        plt.show()

    def plotly_with_manual_labels(self, figsize=(40,30), annotate=True, export=True, save_path='visualizations/vis.png'):
        # TODO: add implementation for plotly
        pass