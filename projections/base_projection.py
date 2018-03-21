class Projection():
    
    def __init__(self, x_positions, y_positions, labels, practice_sets):
        self.x_positions = x_positions
        self.y_positions = y_positions
        self.labels = labels
        self.practice_sets = practice_sets

    
    def simple_scatterplot(figure_size=(30,20),export=False, title='default'):
        """Matplotlib simple scatterplot from dataframe for visualization."""
        x, y = figure_size
        plt.gcf().set_size_inches(x, y)
        plt.title(title)
        plt.xlabel('x_positions')
        plt.ylabel('y_positions')
        plt.scatter(df['x'], df['y'])
        if export:
            plt.savefig('visualizations/matplotlib/%s.png' % title)
        plt.show()
        

    # this will be deleted
    def create_dataframe_for_visualization(x_positions, y_positions, labels, practice_sets):
        """Creates dataframe with x,y positions and their labels, practice sets."""
        df = {'x': x_positions, 'y': y_positions, 'label': labels, 'ps': practice_sets}
        df = pd.DataFrame(data=d)
        df.index.name = 'id'
        return df
