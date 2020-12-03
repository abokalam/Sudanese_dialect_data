import os
import sys
import torch
import random
import argparse
import numpy as np
import torch.nn.functional as F
from GPT2.encoder import *
from GPT2.model import *
from GPT2.utils import *
 
 


class GPT2Config(object):
    def __init__(
            self,
            vocab_size_or_config_json_file=50257,
            n_positions=1024,
            n_ctx=1024,
            n_embd=1600,
            n_layer=48,
            n_head=25,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
    ):
        self.vocab_size = vocab_size_or_config_json_file
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        

 
def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)
 
def sample_sequence(model, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, device='cuda', sample=True,):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    prev = context
    output = context
    past = None
    enc = get_encoder()
    end_type = ""
    resp_name = None
    with torch.no_grad():
        count = 0
        current_token = enc.decode(output[:, len(context):].tolist()[-1]).split(" ")[-1]
        while count < 1 or (not ":" in current_token and not "\n" in current_token):
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
            count += 1
            if count > 100:
                end_type = "max"
                break
            current_token = enc.decode(output[:, len(context):].tolist()[-1]).split(" ")[-1]
        if ":" in current_token:
            end_type = "colon"
        elif "\n" in current_token:
            end_type = "newline"

    return output, end_type
 
def load_model(state_dict):
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=False)
    parser.add_argument("--quiet", type=bool, default=False)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--length", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args([])
 
    if args.quiet is False:
        print(args)
 
    if args.batch_size == -1:
        args.batch_size = 1
    assert args.nsamples % args.batch_size == 0

    seed = random.randint(0, 2147483647)
    print("Random seed: ", seed)

    np.random.seed(seed)
    torch.random.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
    enc = get_encoder()
    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    model = load_weight(model, state_dict)
    model.to(device)
    model.eval()
    return [model, config, enc, args, device], seed
 
def generate_reply(model, config, enc, args, device):
    if args.length == -1:
        args.length = config.n_ctx // 2
    elif args.length > config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % config.n_ctx)

    context_tokens = enc.encode(args.text)
    output = ""
    resp_name = None
    generated = 0
    for _ in range(args.nsamples // args.batch_size):
        out, end_type = sample_sequence(
            model=model,
            context=context_tokens  if not  args.unconditional else None,
            start_token=enc.encoder['<|endoftext|>'] if args.unconditional else None,
            batch_size=args.batch_size,
            temperature=args.temperature, top_k=args.top_k, device=device
        )
        out = out[:, len(context_tokens):].tolist()
        for i in range(args.batch_size):
            generated += 1
            text = enc.decode(out[i])
            #if args.quiet is False:
                #print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            #print(text)
            output += text
    if end_type == "newline":
    	output = output.rstrip('\n')
    elif end_type == "colon":
        split = output.rstrip().rsplit(' ', 1)
        resp_name = split[1].rstrip(':')
        output = split[0]
    elif end_type == "max":
    	output = output.rsplit('.|?|!', 1)[0]
    return output, resp_name


if os.path.exists('gpt2-xl-pytorch_model.bin'):
    state_dict = torch.load('gpt2-xl-pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)
    model, seed = load_model(state_dict)
else:
    print('Please download gpt2-xl-pytorch_model.bin')
    sys.exit()

def main():
    conv_so_far = ""
    name = "samantha"
    my_name = "me"
    in_conv = [name]
    print("Seed: ", seed)
    print(f"Start a conversation with {name} or say \"Bye\" at any point to reset personality.\n")
    while True:
        msg = input(f"{my_name}: ")
        if msg.lower() == "bye":
            conv_so_far = ""
            in_conv = [name]
            # clear_screen()
            print("Enter a name (No Spaces.):")
            name = "samantha"
            in_conv = [name]
            print("Seed: ", seed)
            print(f"Start a conversation with {name} or say \"Bye\" at any point to reset personality.\n")
            continue
        who_talks = random.choice(in_conv)
        conv_so_far += f" {my_name}: " + msg + " " + who_talks + ":"
        model[3].text = conv_so_far
        answer, resp_name = generate_reply(model[0], model[1], model[2], model[3], model[4]) #.rstrip().rstrip(f"{my_name}:")
        conv_so_far += answer
        print(who_talks + ": " + answer.strip(' '))
    return 0



if __name__ == "__main__":
    main()
