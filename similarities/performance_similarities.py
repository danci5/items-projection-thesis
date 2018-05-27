def drop_nans(similarity_matrix):
    """Removes NaN values from matrix and returns the matrix.

    NOTE: Not using pandas dropna because it would drop the label if any NA values are present.
    Therefore, row that has the most NaNs is found and dropped (gradually removes all NaNs).
    """
    while similarity_matrix.isnull().sum().sum() > 0:
        most_nans = similarity_matrix.isnull().sum().idxmax()
        similarity_matrix = similarity_matrix.drop(most_nans, axis=0)
        similarity_matrix = similarity_matrix.drop(most_nans, axis=1)

    return similarity_matrix


def replace_nans_with_zero(similarity_matrix):
    """Replace NaN values with 0."""
    return similarity_matrix.fillna(0)


def pearson_similarity(matrix, no_nans=False):
    """Returns similarity matrix created by applying Pearson on learners' performance data."""
    if no_nans:
        return drop_nans(matrix.corr())
    else:
        return matrix.corr()


def doublepearson_similarity(matrix, no_nans=False):
    """Returns similarity matrix created by applying Pearson twice on learners' performance data."""
    if no_nans:
        return pearson_similarity(pearson_similarity(matrix, True), True)
    else:
        return pearson_similarity(pearson_similarity(matrix))
