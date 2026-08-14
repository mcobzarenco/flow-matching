"""Generate tests/fixtures/molmo_flow_parity/port_outputs.npz by
EXECUTING THE PORT (bijou/molmoact2/) — run at tag
pre-molmoact2-retirement, the last commit where the port exists.

Mirrors tests/test_molmo_flow.py's live-pair test bodies byte-exactly:
_tiny_pair() (manual_seed(0), decoder weights copied into the port),
_inputs() (manual_seed(1)), the state draw continuing the seed-1
stream, and the seeded Euler loop. The saved tensors are the PORT's
outputs; the converted tests compare OUR decoder against them, so the
parity oracle survives the port's deletion.
"""

import numpy as np
import torch

from bijou.decoders.molmo_flow import MolmoFlowConfig, SamplingMethod
from bijou.molmoact2.action_expert import ActionExpertConfig
from bijou.molmoact2.wiring import generate_actions

TINY = MolmoFlowConfig(
    max_horizon=6,
    max_action_dim=8,
    hidden_size=64,
    num_layers=2,
    num_heads=4,
    mlp_ratio=4.0,
    ffn_multiple_of=32,
    timestep_embed_dim=16,
    dropout=0.0,
    attn_dropout=0.0,
    context_layer_norm=True,
    qk_norm=True,
    qk_norm_eps=1e-6,
    rope=True,
    causal_attn=False,
    llm_kv_dim=40,
)

# _tiny_pair(), with the integration fixture's deterministic
# perturbation: adaLN-Zero init outputs an exactly-zero velocity field,
# which would make the forward fixture VACUOUS (any zero-outputting
# implementation would pass). Perturb into a non-degenerate field
# BEFORE copying weights into the port, so the fixture pins real math
# through every block.
torch.manual_seed(0)
decoder = TINY.build()
generator = torch.Generator().manual_seed(42)
with torch.no_grad():
    for block in decoder.iter_blocks():
        block.modulation.linear.bias.fill_(0.1)
    for parameter in (
        decoder.final_layer.linear.weight,
        decoder.final_layer.linear.bias,
        decoder.final_layer.modulation.linear.bias,
    ):
        parameter.add_(0.05 * torch.randn(parameter.shape, generator=generator))
fields = {
    f.name: getattr(TINY, f.name)
    for f in __import__("dataclasses").fields(ActionExpertConfig)
}
port = ActionExpertConfig(**fields).build(llm_kv_dim=TINY.llm_kv_dim)
port.load_state_dict(decoder.state_dict(), strict=True)
decoder.eval()
port.eval()

# _inputs(), verbatim (batch=2, ctx_len=5).
torch.manual_seed(1)
actions = torch.randn(2, TINY.max_horizon, TINY.max_action_dim)
timesteps = torch.rand(2)
kv_states = [
    (torch.randn(2, 5, TINY.llm_kv_dim), torch.randn(2, 5, TINY.llm_kv_dim))
    for _ in range(TINY.num_layers)
]
enc_mask = torch.ones(2, 5, dtype=torch.bool)
enc_mask[1, -2:] = False

with torch.no_grad():
    forward = port(actions, timesteps, kv_states, encoder_attention_mask=enc_mask)
    # The state draw continues the seed-1 stream exactly as the test did.
    states = torch.randn(2, TINY.hidden_size)
    forward_state = port(actions, timesteps, kv_states, state_embeddings=states)
    pad = torch.zeros(TINY.max_action_dim, dtype=torch.bool)
    pad[-2:] = True
    euler = generate_actions(
        port,
        encoder_kv_states=kv_states,
        encoder_attention_mask=enc_mask,
        action_dim_is_pad=pad,
        num_steps=4,
        generator=torch.Generator().manual_seed(7),
    )

# Non-vacuity gates: a fixture of zeros pins nothing.
assert float(forward.abs().sum()) > 1.0, "vacuous forward fixture"
assert float(euler.abs().sum()) > 1.0, "vacuous euler fixture"
assert not torch.equal(forward, forward_state), "state path inert"
np.savez_compressed(
    "tests/fixtures/molmo_flow_parity/port_outputs.npz",
    forward=forward.numpy(),
    states=states.numpy(),
    forward_state=forward_state.numpy(),
    euler=euler.numpy(),
)
print("forward", forward.shape, float(forward.abs().sum()))
print("euler", euler.shape, float(euler.abs().sum()))
print("written tests/fixtures/molmo_flow_parity/port_outputs.npz")
