from scipy.stats import wilcoxon, friedmanchisquare

def wilcoxon_test(scores_model_1, scores_model_2):
    stat, p_value = wilcoxon(scores_model_1, scores_model_2)
    return stat, p_value

def friedman_test(*model_scores):
    stat, p_value = friedmanchisquare(*model_scores)
    return stat, p_value
