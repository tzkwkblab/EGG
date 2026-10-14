# reconstruct_game

A general attribute-value reconstruction game trained with REINFORCE.

A state is a tuple of categorical attributes. Sender receives an input describing the state
(by default the concatenation of one-hot vectors, one per attribute), sends a fixed-length message,
and Receiver outputs, for every attribute, a distribution over its values. The game is
`egg.zoo.compo_vs_generalization` reorganized so that the following can be chosen from the command line:

| | options |
|---|---|
| state space | `--n_values` (one number per attribute; they may differ), `--n_train`, `--n_test`, `--split_seed` |
| agents | `--sender_cell`, `--receiver_cell` (rnn, gru, lstm), `--sender_hidden`, `--receiver_hidden`, `--sender_emb`, `--receiver_emb`, `--sender_input_layer` (linear, mlp), `--receiver_dropout` |
| losses | `--receiver_loss` (ce, gce, sce), `--sender_loss` (same, ce, gce, sce: the loss whose negative is Sender's reward), `--receiver_weight_mode` (weighted-CE surrogates), `--attr_reduction` (mean, sum), `--gce_kappa`, `--sce_alpha`, `--sce_beta`, `--sce_A` |
| REINFORCE | `--baseline` (no, mean, builtin, value), `--value_network` (separate, shared, shared_detach), `--value_loss_coeff`, `--sender_entropy_coeff` |
| optimization | `--optimizer` (adam, adamw, sgd, adagrad), `--lr`, `--sender_lr`, `--receiver_lr`, `--weight_decay`, `--batch_size`, `--n_steps`, `--steps_per_epoch`, `--shuffle_train` / `--no_shuffle_train` |
| randomness | `--random_seed` (initial parameters), `--split_seed` (train/test split), `--data_seed` (order of training batches), `--train_noise_seed` (Sender's sampling and dropout) |
| recording | `--out_dir`, `--record_dynamics`, `--record_steps`, `--save_checkpoint` |

Defaults follow `compo_vs_generalization` (RNN cells, hidden size 50, Adam, running-mean baseline),
except that training is counted in steps and training batches are reshuffled at every pass.

```bash
python -m egg.zoo.reconstruct_game.train \
    --n_values 10 10 10 10 --max_len 5 --vocab_size 10 --batch_size 5120 --n_steps 20000 \
    --sender_cell gru --receiver_cell gru --sender_hidden 500 --receiver_hidden 500 \
    --receiver_emb 30 --receiver_dropout 0.5 --optimizer adamw --lr 0.001 \
    --baseline value --receiver_loss gce --record_dynamics --out_dir runs/gce_seed1 --random_seed 1
```

Messages consist of `--max_len` symbols from `1..vocab_size` followed by a constant symbol
(as in `compo_vs_generalization`, Sender is wrapped in `PlusOneWrapper`, so the eos symbol 0 is
never produced and all messages have the same length).

## Outputs (`--out_dir`)

- `config.json`: all options, EGG revision, library versions
- `final.json`: metrics at the last step (TopSim, PosDis, BosDis, accuracy on the test states and on
  a fixed subset of the training states) and timings
- `dynamics.jsonl`: one line per recorded step (see `egg.core.dynamics.DynamicsRecorder`)
- `protocol.npz`: greedy messages and the attribute-position mutual information matrix at every
  recorded step, on a fixed probe set
- `train_log.jsonl`: light-weight training log
- `checkpoint_final.pt` (with `--save_checkpoint`)

## Using other inputs

`train.main(params, sender_input_fn=..., batch_transform=...)` lets a script replace Sender's input:
`sender_input_fn(states, n_values)` maps integer states `[N, n_attributes]` to inputs `[N, d]`
(e.g. pre-computed image features), and `batch_transform(sender_input, labels)` is applied to every
training batch (e.g. per-presentation noise). Receiver's targets and all metrics always use the
ground-truth attribute values. `game.build_game` and `game.build_optimizer` build the same game
from Python.

## Library components used by this game

- `egg.core.losses`: `gce_loss`, `sce_loss`, `weighted_ce_loss`, `AttributeReconstructionLoss`
- `egg.core.SenderReceiverRnnReinforce`: `sender_loss`, `value_fn`, `value_loss_coeff`
- `egg.core.baselines`: `MLPCritic`, `SharedFeatureCritic`
- `egg.core.util.build_game_optimizer`: separate learning rates for Sender and Receiver, AdamW
- `egg.core.datasets`: `CyclicBatchSampler`, `IndexedTensorLoader`
- `egg.core.dynamics.DynamicsRecorder`, and functional metrics in `egg.core.language_analysis`
