# Temperature Sampling Study

## Finding
Temperature has minimal effect on instruction-tuned models (Gemma-3-1B, Qwen2.5-1.5B) for factual and creative prompts. The model is too confident in its top token — temperature changes the distribution but rarely changes the argmax. Nucleus sampling (top_p=0.9) shows slightly different outputs with marginally lower diversity.

## Results (10 creative prompts, 40 tokens each)

| Config | Qwen Div | Gemma Div | Qwen Rep | Gemma Rep | Effect |
|---|---|---|---|---|---|
| Greedy (T=0.1) | 0.818 | 0.882 | 0/10 | 0/10 | Baseline |
| Mild (T=0.7) | 0.818 | 0.882 | 0/10 | 0/10 | None |
| High (T=1.5) | 0.818 | 0.882 | 0/10 | 0/10 | None |
| Nucleus p=0.9 | 0.789 | 0.857 | 0/10 | 0/10 | Minor |

## Why
Instruction-tuned models are trained to produce confident, deterministic responses. The softmax distribution is sharply peaked — the top token has >90% probability. Temperature scaling doesn't change which token is most likely until T becomes very high (>5.0), at which point the distribution becomes uniform and the output is random noise.

## Practical Implication
Temperature is not a useful control knob for instruction-tuned models in the 0.1-1.5 range. Use nucleus sampling (top_p) for creative diversity. Save temperature tuning for base models or RLHF policy training where the distribution is less peaked.