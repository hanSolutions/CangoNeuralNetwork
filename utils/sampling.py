import numpy as np


def undo_oversampling(prob, orginal_fraction, oversampled_fraction):
    # https://yiminwu.wordpress.com/2013/12/03/how-to-undo-oversampling-explained/
    # http://www.data-mining-blog.com/tips-and-tutorials/overrepresentation-oversampling/
    # adjust_prob = 1 / ( 1 + (1/orginal_fraction)/(1/oversampled_fraction -1) * (1 / prob - 1))

    original_odds = np.subtract(np.divide(1.0, orginal_fraction), 1.0)
    oversampled_odds = np.subtract(np.divide(1.0, oversampled_fraction), 1.0)
    scoring_odds = np.subtract(np.divide(1.0, prob), 1.0)
    adjust_odds = np.multiply(np.divide(original_odds, oversampled_odds), scoring_odds)
    adjust_prob = np.divide(1.0, np.add(1.0, adjust_odds))

    return adjust_prob