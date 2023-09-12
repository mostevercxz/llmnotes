## chat 类应用

github 上开源的 chat 类应用:
* [ChatDev,代表不同公司角色（如CTO、软件工程师和设计师）的LLM智能体](https://github.com/OpenBMB/ChatDev)
* [chat莎士比亚,十几条数据集+gpt3.5 fine tune](https://github.com/steven-tey/shooketh/tree/main)
* [将gpt3.5 ft成 NL-to-SQL](https://medium.com/dataherald/fine-tuning-gpt-3-5-turbo-for-natural-language-to-sql-4445c1d37f7c)

## agent 探索

[Agently,用于快速构建LLM Agent应用程序的轻量级框架](https://github.com/Maplemx/Agently)

## papers

[llm agent 调查相关paper搜集](https://github.com/Paitesanshi/LLM-Agent-Survey)

## prompt engineering

[claude提示词工程师的建议](https://twitter.com/op7418/status/1696868059770925391)

## 评测数据集

[LongChat数据集格式](https://github.com/DachengLi1/LongChat),参考下这个 https://github.com/DachengLi1/LongChat/blob/longeval/longeval/evaluation/topics/predictions/chatglm2-6b/5_response.txt
看下怎么评估数据集质量的

## prompts

### replicate 训练 Homer Simpson
reference: https://github.com/daanelson/homerbot_errata/blob/main/simpsons_analytics.ipynb
1. 从Kaggle的《辛普森一家》数据集中拿到 simpsons_script_lines.csv 。内有辛普森一家第 27 季之前所有剧集的剧本。
2. 提取1-12季数据，因为数据质量更高。The resulting dataset has 61k lines of dialog and 1.1M tokens.
3. 数据包含：辛普森前面的对话，辛普森，辛普森说的话，比如{'previous': 'Marge Simpson: Ooo, careful, Homer.',
 'character': 'Homer Simpson',
 'line': "There's no time to be careful."}
4. 提示词是:"Below is a script from the American animated sitcom The Simpsons.\
 Write a response that completes {character}'s last line in the \
conversation. \n\n{previous}\n{character}:";

```python3
good_season_episodes = ep_df[ep_df['season'] <= 12]
tok = AutoTokenizer.from_pretrained('google/flan-t5-xl')
for samp in range(2):
    n = np.random.randint(good_df.shape[0])
    orig_n = n
    n_tokens = 0
    while n_tokens < 1500:
        text = good_df.iloc[n].spoken_words
        #print(f'text: {text}')
        tokenized = tok(text, return_tensors='pt').input_ids
        #print(f'tokenized {tokenized}')
        n_tokens += tokenized.shape[1]
        n+=1
    print(f'tokens: {n_tokens}')
    print(f'n lines: {n - orig_n}')

def max_scenes(df):
    cur_scene = []
    big_tuples = []
    for ind, line in df.iterrows():
        if line.raw_character_text == 'Nobody':
            n_tokens = len(tok(' '.join(cur_scene))['input_ids'])
            big_tuples.append((len(cur_scene), n_tokens))
            cur_scene = []
        else:
            cur_scene.append(line.spoken_words)
    return big_tuples

lines_per_scene = [val[0] for val in res if val[0] > 0 ]

def generate_data(df):
    cur_scene = []
    for ind, line in df.iterrows():
        if line.raw_character_text == 'Nobody':
            process_scene(cur_scene)
            cur_scene = []
        else:
            cur_scene.append((line.raw_character_text, line.spoken_words))

def build_text(preamble):
    return '\n'.join([f"{char}: {text}" for char, text in preamble])

def process_scene(cur_scene):
    if len(cur_scene) <= 1:
        return
    for ind in range(1, len(cur_scene)):
        prompt = build_text(cur_scene[max(0, ind-CONTEXT_LINES):ind])
        cur_char, cur_text = cur_scene[ind]
        data.append({'p': prompt, 'c': cur_char, 'l': cur_text})

generate_data(good_df)

with open('processed_conversations/simpsons_data_two.json', 'w') as f:
    json.dump(data, f)
```

('Marge Simpson: Ooo, careful, Homer.\nHomer Simpson:',
 "There's no time to be careful.")
