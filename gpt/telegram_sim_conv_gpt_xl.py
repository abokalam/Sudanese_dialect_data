import os
import sys
import torch
import random
import argparse
import numpy as np
import torch.nn.functional as F
import pandas as pd
from GPT2.encoder import *
from GPT2.model import *
from GPT2.utils import *


import telegram
from telegram.ext import (Updater, CommandHandler, MessageHandler, Filters)

import timeit
 
 
TOKEN = "1394004879:AAFHSGuiit6HBZdyv2QvZsVuZOqS4hz4F4s"
PORT = int(os.environ.get('PORT', '5000'))
SERVER = '34.90.109.168/'
CERT = '/etc/nginx/public6.pem'

print("i am here")

conv_so_far = ""





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
    parser.add_argument("--top_k", type=int, default=80)
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

    print("!!!!! input:" + args.text)
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
        print('!!!!!!!!output is:' +output)
        split = output.rstrip().rsplit(' ', 1)
       # print("split :" + str(split))
        resp_name = "me"
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


def findWholeWord(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search


def model_output(text , sender_id, sender_first_name):
    global conv_so_far
    global model
    global seed
    is_entry = sender_id in conv_so_far.sender_id.values
   # personality  = 'me: hi samantha: hello me: tell me about yourself samantha: my name is samantha'
    personality = ''
    if is_entry == False:
        new_entry = {'sender_id': sender_id,'sender_first_name' : sender_first_name, 'conv': personality, 'old_conv':'', 'seed':seed}
        conv_so_far = conv_so_far.append(new_entry, ignore_index=True)
        print("new entry added")
        
    print("Seed: ", seed)
    print("conv so far:" + conv_so_far.loc[conv_so_far['sender_id'] == sender_id ].conv + "\n")
    name = "samantha"
    my_name = 'me' 
    in_conv = [name]
    # print(f"Start a conversation with {name} or say \"Bye\" at any point to reset personality.\n")
    msg = text
    if findWholeWord('hello')(msg.lower()) or findWholeWord('hey')(msg.lower()) or findWholeWord('hi')(msg.lower()) or findWholeWord('good evening')(msg.lower()) or findWholeWord('good morning')(msg.lower()) or findWholeWord('good afternoon')(msg.lower()):
        answer = "hello " + sender_first_name
       # print("greetings detected")
       # who_talks = random.choice(in_conv)
       # conv_so_far.loc[conv_so_far['sender_id'] == sender_id , "conv"] += f" {my_name}: " + msg + " " + who_talks + ":"
        #conv_so_far.loc[conv_so_far['sender_id'] == sender_id , "conv"] += answer
        print("//////////////////////greetings")
        return answer
    if msg.lower() == "bye":
        print("resetting the conversation...")
        conv_so_far.loc[conv_so_far['sender_id'] == sender_id , "old_conv"] += conv_so_far.loc[conv_so_far['sender_id'] == sender_id , "conv"]
        conv_so_far.loc[conv_so_far['sender_id'] == sender_id , "conv"] = ""
        in_conv = [name]
       # clear_screen()
        name = "samantha"
        in_conv = [name]
        print("Seed: ", seed)
        return "The conversation has been reset"
    who_talks = random.choice(in_conv)
    conv_so_far.loc[conv_so_far['sender_id'] == sender_id , "conv"] += f" {my_name}: " + msg + " " + who_talks + ":"
    print(str(conv_so_far.loc[conv_so_far['sender_id'] == sender_id , "conv"].values).strip('[').strip(']').strip('\'') + "//////////////////////////")
    model[3].text =str(conv_so_far.loc[conv_so_far['sender_id'] == sender_id , "conv"].values).strip('[').strip(']').strip('\'')
    answer, resp_name = generate_reply(model[0], model[1], model[2], model[3], model[4]) #.rstrip().rstrip(f"{my_name}:")
    if len(answer.strip(' ')) == 1:
        print("!?!?!?!?!? I !?!?!?!?!?!?!?!")
        del model
        model, seed = load_model(state_dict)
        model[3].text = str(conv_so_far.loc[conv_so_far['sender_id'] == sender_id , "conv"].values).strip('[').strip(']').strip('\'')
        answer, resp_name = generate_reply(model[0], model[1], model[2], model[3], model[4])
        conv_so_far.loc[conv_so_far['sender_id'] == sender_id , "conv"] += answer
        return answer
    conv_so_far.loc[conv_so_far['sender_id'] == sender_id , "conv"] += answer
    print("client:" + msg)
    print(who_talks + ": " + answer.strip(' '))
    if answer == "":
        answer = ""
        return "empty"
    return answer

def htel(update, context):
  update.message.reply_text('init samantha bot')
def echo(update,context):
  global conv_so_far  
  text = update.message.text
  sender = update.message.from_user
  sender_id = update.message.from_user.id
  sender_first_name = update.message.from_user.first_name
  print("message recieved")
  print("processing input...")
 # is_start = (text == '/start')
 # if is_start == False:
  output = model_output(text, sender_id, sender_first_name)
  print("sending model output")
  update.message.reply_text(output)
  start = timeit.default_timer()

  conv_so_far.to_csv('telegram_bot_history.csv',index=False)

  stop = timeit.default_timer()

  print('Time to save history: ', stop - start)



def tbot():
  updater = Updater(TOKEN, use_context=True)
  global conv_so_far
  columns = ['sender_id','sender_first_name', 'conv','old_conv', 'seed']
  conv_so_far = pd.DataFrame(columns=columns)
  dp = updater.dispatcher
  dp.add_handler(CommandHandler("start", htel))
  dp.add_handler(MessageHandler(Filters.text, echo))
  updater.start_webhook(listen="0.0.0.0", port=5000, url_path=TOKEN)
  updater.bot.setWebhook(SERVER+TOKEN,certificate=open(CERT, 'rb'))
  updater.idle()
tbot()
