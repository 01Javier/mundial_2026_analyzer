from scipy.stats import poisson


def dixon_coles_tau(home_goals, away_goals, lambda_home, lambda_away, rho):
    """
    Ajuste Dixon-Coles para marcadores bajos: 0-0, 0-1, 1-0, 1-1.
    """
    if home_goals == 0 and away_goals == 0:
        return 1 - (lambda_home * lambda_away * rho)
    if home_goals == 0 and away_goals == 1:
        return 1 + (lambda_home * rho)
    if home_goals == 1 and away_goals == 0:
        return 1 + (lambda_away * rho)
    if home_goals == 1 and away_goals == 1:
        return 1 - rho
    return 1.0


def poisson_score_matrix_dc(lambda_home, lambda_away, rho=-0.05, max_goals=8):
    """
    Genera matriz de marcadores con correccion Dixon-Coles y normaliza a 1.
    """
    matrix = {}
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            base = poisson.pmf(hg, lambda_home) * poisson.pmf(ag, lambda_away)
            tau = dixon_coles_tau(hg, ag, lambda_home, lambda_away, rho)
            matrix[(hg, ag)] = max(0.0, base * tau)

    total = sum(matrix.values())
    if total <= 0:
        return matrix, 0.0

    normalized = {score: prob / total for score, prob in matrix.items()}
    return normalized, total
