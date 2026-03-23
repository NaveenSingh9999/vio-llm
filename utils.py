import torch
import torch.nn.functional as F

def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, top_p=0.9, sample=True, device='cpu'):
    model.eval()
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        idx_cond = input_ids[:, -128:]
        with torch.no_grad():
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
        if sample:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        input_ids = torch.cat((input_ids, idx_next), dim=1)
        if idx_next.item() in [tokenizer.sep_token_id, tokenizer.pad_token_id]: break
    return tokenizer.decode(input_ids[0])