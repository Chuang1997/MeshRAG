"""Question decomposition -> sequential hash-graph retrieval -> answers."""
import ast
import argparse
import json
import os
import re
from pathlib import Path
from prompts import DECOMPOSE_HEAD, DECOMPOSE_INSTRUCTION, CHAIN_REASONING_HEAD, CHAIN_REASONING_INSTRUCTION
from retrieve import GraphRetriever


class LLM:
    def __init__(self, model="deepseek-flash", provider="deepseek"):
        from openai import OpenAI
        self.provider = provider
        if provider == 'deepseek':
            self.client = OpenAI(api_key=os.environ['DEEPSEEK_API_KEY'],
                                 base_url=os.getenv('DEEPSEEK_BASE_URL','https://api.deepseek.com'), max_retries=3, timeout=90)
        else:
            self.client = OpenAI(max_retries=3, timeout=90)
        self.model = model

    def __call__(self, prompt, json_mode=False):
        options = {'response_format': {'type':'json_object'}} if json_mode else {}
        if self.provider == 'deepseek':
            options.update(max_tokens=2000, extra_body={'thinking': {'type':'disabled'}})
        else:
            options['max_completion_tokens'] = 2000
        result = self.client.chat.completions.create(
            model=self.model, messages=[{'role':'user','content':prompt}],
            temperature=0, **options)
        if result.choices[0].finish_reason != 'stop':
            raise RuntimeError('Incomplete LLM response')
        text = result.choices[0].message.content
        if not text:
            raise RuntimeError('Empty LLM response')
        return json.loads(text) if json_mode else text.strip()


def decompose_question(question, llm):
    # Preserve the original heuristic and example; parse list or numbered output.
    prompt = (f'{DECOMPOSE_HEAD}\n'
              "Example question: what is one of the stars of The Newcomers known for?\n"
              "Decomposition: ['1.What is The Newcomers (e.g., a film, TV series, etc.)?', "
              "'2.Who are the leading actors/actresses in The Newcomers?', "
              "'3.What is one of those stars renowned or famous for?']\n"
              f'The given question Q: {question}\n{DECOMPOSE_INSTRUCTION}')
    raw = llm(prompt).strip()
    if raw.lower().rstrip('.') == 'end':
        return [question]
    raw = re.sub(r'^```(?:json|python)?\s*|\s*```$', '', raw).strip()
    try:
        items = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        items = [line.strip() for line in raw.splitlines() if re.match(r'^\s*\d+[.)]',line)]
    if not isinstance(items,list) or not items or not all(isinstance(x,str) and x.strip() for x in items):
        raise ValueError(f'Invalid decomposition: {raw}')
    return [re.sub(r'^\s*\d+[.)]\s*','',x).strip() for x in items]


def chain_reasoning_with_retrieval(original_question, sub_questions, retriever, llm, top_k=3):
    sub_answers, previous = [], []
    for step, subq in enumerate(sub_questions,1):
        query = f'{previous[-1]}, {subq}' if previous and previous[-1].lower() != 'unknown' else subq
        hits = retriever.retrieve(query,top_k)
        context = '\n'.join(f"[{p['id']}] {p['text']}" for p in hits)
        answer = llm(f'{CHAIN_REASONING_HEAD}\nContext: {context}\n'
                     f'Previous answers: {json.dumps(previous)}\nSub-question: {subq}\n'
                     f'{CHAIN_REASONING_INSTRUCTION}')
        previous.append(answer)
        sub_answers.append(dict(step=step, sub_question=subq, retrieved=hits, answer=answer))
    evidence = json.dumps(sub_answers,ensure_ascii=False)
    final_prompt = (f'Original Question: {original_question}\n{evidence}\n'
                    'Based on the sub-questions, answers and retrieved information, provide a direct answer '
                    'to the original question. Output only the answer itself.')
    return sub_answers,final_prompt


def answer_question(question, retriever, llm, top_k=5):
    subqs = decompose_question(question, llm)
    steps, prompt = chain_reasoning_with_retrieval(question, subqs, retriever, llm, top_k)
    return dict(question=question, answer=llm(prompt), steps=steps)


def load_corpus(path):
    path = Path(path)
    if path.suffix == '.jsonl':
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        rows = json.loads(path.read_text())
    corpus = [dict(id=str(p.get('id', p.get('idx'))),
                   text=(str(p['title']) + '\n' if p.get('title') else '') + p['text']) for p in rows]
    if not corpus or len({p['id'] for p in corpus}) != len(corpus):
        raise ValueError('Corpus IDs must be nonempty and unique')
    if any(p['id'] in ('None', '') or not isinstance(p['text'], str) or not p['text'].strip() for p in corpus):
        raise ValueError('Each paragraph requires an ID and nonempty text')
    return corpus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--question', required=True)
    parser.add_argument('--state', default='state')
    parser.add_argument('--model', default='deepseek-flash')
    parser.add_argument('--encoder', default='all-MiniLM-L6-v2')
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args()
    retriever = GraphRetriever(load_corpus(args.corpus), args.state, args.encoder)
    result = answer_question(args.question, retriever, LLM(args.model), args.top_k)
    Path(args.state, 'answer.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(result['answer'])


if __name__ == '__main__':
    main()
