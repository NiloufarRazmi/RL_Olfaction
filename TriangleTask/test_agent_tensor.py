import pandas as pd
import torch
from .agent_tensor import EpsilonGreedy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_decrease_epsilon():
    explorer = EpsilonGreedy(
        epsilon=1,
        epsilon_min=0.1,
        epsilon_max=1,
        decay_rate=0.01,
        epsilon_warmup=100,
    )
    episodes = torch.arange(600, device=DEVICE)
    epsilons = torch.empty_like(episodes) * torch.nan
    for eps_i, epsi in enumerate(epsilons):
        epsilons[eps_i] = explorer.epsilon
        explorer.epsilon = explorer.update_epsilon(episodes[eps_i])
    assert epsilons[0] == 1
    torch.testing.assert_close(
        actual=epsilons[-1], expected=torch.tensor(0.1), atol=6e-3, rtol=1e-2
    )
    assert epsilons[102] < 1
    assert epsilons[200] > 0.4 and epsilons[200] < 0.5
    assert epsilons[300] > 0.2 and epsilons[300] < 0.4
    assert epsilons[400] > 0.1 and epsilons[400] < 0.2
    df = pd.DataFrame({"episodes": episodes, "epsilons": epsilons})
    assert df.epsilons.is_monotonic_decreasing
