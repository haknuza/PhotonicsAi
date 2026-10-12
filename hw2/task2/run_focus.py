#!/usr/bin/env python3
"""Run the existing four-role agent architecture on autonomous DMD focusing."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
from datetime import datetime
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'agent'))
from agent_system.llm import build_llm
from agent_system.graph import clean, content_text, parse_decision
from agent_system.state import DialogueState, render_dialogue, last_decision
from agent_system.coding_tool import CodingTool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, START, END
from bench import Bench, Settings

CONTEXT = '''Simulated focusing bench: 16x16 binary amplitude DMD; fixed 32x32 complex Gaussian
transmission matrix camera; I=|T Ein|^2. Each shot has Poisson noise (1000 photons per intensity unit)
and Gaussian read noise sigma .002. Focus at camera (16,16), square odd width focus_size.
Proximal annulus starts at focus_size//2+2 and ends at background_radius, split into angular regions.
Enhancement = mean focus intensity / unweighted mean of regional background means.
Random binary start; flip random macropixels and accept if a new noisy candidate measurement beats a
fresh current measurement. Final enhancement uses 20 independent shots; no hidden T information
is supplied to the search. Available settings: macropixel_size in [1,2,4,8], focus_size in [1,3,5],
background_radius 4..15, background_regions 2..8, iterations 10..2000, flips 1..16 (at most macro count),
seed nonnegative integer. Keep seed fixed for comparisons. Choose at least three experiments,
including different macropixel sizes. Aim for a bright localized single-pixel focus.
For the three-run controlled comparison, choose one focus geometry and one background
geometry and hold them fixed across all runs; vary macropixel size. Include a fine grid
so that enough independently controlled contributions can form a strong focus.
Choose modest iterations, at most 500 initially. All settings are yours to choose without human input.
Only measured tool results are authoritative. Do not claim results before a tool returns.'''


def build_focus_graph(llm, output, max_decisions=8):
    experiments = []
    output = Path(output)
    coding = CodingTool(output/'work')
    def focus(**kwargs):
        settings = Settings(**kwargs)
        directory = output/f'experiment_{len(experiments)+1:02d}'
        print('[tool] focus_beam '+json.dumps(settings.model_dump()), flush=True)
        result = Bench(settings).run(directory)
        experiments.append(result)
        (output/'experiments.json').write_text(json.dumps(experiments,indent=2))
        print(f"[measurement] enhancement={result['enhancement']:.6f}", flush=True)
        return json.dumps(result)
    tool = StructuredTool.from_function(func=focus, name='focus_beam', args_schema=Settings,
                                       description='Run noisy simulated focusing search and save camera images. '+CONTEXT)
    def write_and_run_python(filename: str, code: str):
        return coding.write_and_run(filename, code).report()
    code_tool = StructuredTool.from_function(write_and_run_python,
                                             description='Write and run Python for auxiliary analysis.')
    def orchestrator(state):
        feedback = ''
        for _ in range(4):
            system = ('You are the ORCHESTRATOR. Read shared dialogue and choose scientist (plan), '
                      'general (concept explanation), executor (tools), or finish. No human pauses. '
                      'Start with scientist. Send plans to executor. Finish only after at least three '
                      'successful experiments. You write no code. Return JSON only: '
                      '{"next_role":"scientist|general|executor|finish","instruction":"concrete instruction",'
                      '"reason":"one sentence"}. '+CONTEXT)
            response = llm.invoke([SystemMessage(content=system),HumanMessage(content=
                render_dialogue(state['messages'],limit=3500)+f'\nVerified experiments: {len(experiments)}'+feedback)])
            try:
                d = parse_decision(content_text(response))
                if not any(m['role']=='scientist' for m in state['messages']) and d.next_role!='scientist':
                    raise ValueError('First route must be scientist for an experimental plan')
                if d.next_role=='human' or (d.next_role=='finish' and len(experiments)<3):
                    raise ValueError('no human destination; finish requires three verified experiments')
                break
            except ValueError as e:
                feedback = '\nRejected response: '+str(e)
        else:
            raise RuntimeError('No valid model routing decision')
        print(f'[orchestrator] -> {d.next_role}: {d.reason}',flush=True)
        return {'messages':[dict(role='orchestrator',text=d.instruction+'\n'+d.reason,decision=d.model_dump())]}
    def specialist(role):
        def node(state):
            response=llm.invoke([SystemMessage(content=f'You are the {role.upper()}. No tools. '+CONTEXT+
                (' Give a short experimental plan with three explicit settings configurations; no code.' if role=='scientist' else 'Explain briefly.')),
                HumanMessage(content=render_dialogue(state['messages'],limit=3000))])
            text=clean(content_text(response)); print(f'[{role}] {text}',flush=True)
            return {'messages':[dict(role=role,text=text)]}
        return node
    def executor(state):
        convo=[SystemMessage(content='You are the EXECUTOR. Use focus_beam to implement the scientist plan. '
                    'Run all three planned experiments now. You choose arguments; no human input. '
                    'Summarize real results after tool calls. '+CONTEXT),
               HumanMessage(content=render_dialogue(state['messages'],limit=4500))]
        bound=llm.bind_tools([tool,code_tool]); records=[]
        for _ in range(5):
            response=bound.invoke(convo); convo.append(response)
            calls=getattr(response,'tool_calls',[]) or []
            if not calls:
                text=clean(content_text(response)); break
            for call in calls:
                if len(records)>=4:
                    result='Tool limit reached.'
                else:
                    try:
                        result= {'focus_beam':tool,'write_and_run_python':code_tool}[call['name']].invoke(call['args'])
                    except Exception as e:
                        result='Tool error: '+str(e)
                    records.append(dict(name=call['name'],args=call['args'],result=result))
                convo.append(ToolMessage(content=result,tool_call_id=call['id']))
        else:
            text='Executor tool budget reached.'
        print('[executor] '+text,flush=True)
        return {'messages':[dict(role='executor',text=text+'\n'+json.dumps(records),tool_results=records)]}
    g=StateGraph(DialogueState)
    g.add_node('orchestrator',orchestrator)
    for role in ('scientist','general'): g.add_node(role,specialist(role))
    g.add_node('executor',executor); g.add_edge(START,'orchestrator')
    g.add_conditional_edges('orchestrator',lambda s: END if last_decision(s['messages'])['next_role']=='finish'
                            else last_decision(s['messages'])['next_role'])
    for role in ('scientist','general','executor'):
        g.add_conditional_edges(role,lambda s: END if sum(m['role']=='orchestrator' for m in s['messages'])>=max_decisions else 'orchestrator')
    return g.compile(),experiments


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',default=os.environ.get('LLM_MODEL','qwen2.5:3b'))
    ap.add_argument('--base-url'); ap.add_argument('--output')
    ap.add_argument('--request',default='Focus the beam on the simulated bench. Choose the settings yourself, compare at least three experiments, and keep the best measured enhancement.')
    args=ap.parse_args()
    output=Path(args.output or Path(__file__).parent/'runs'/datetime.now().strftime('%Y%m%d-%H%M%S'))
    output.mkdir(parents=True,exist_ok=False)
    llm=build_llm(args.model,args.base_url)
    llm.max_tokens=900
    graph,experiments=build_focus_graph(llm,output)
    messages=[dict(role='user',text=args.request)]
    status='failed'
    try:
        for state in graph.stream({'messages':messages},stream_mode='values',config={'recursion_limit':40}):
            messages=state['messages']
            (output/'transcript.json').write_text(json.dumps(messages,indent=2))
            (output/'transcript.md').write_text(render_dialogue(messages))
        status='complete' if len(experiments)>=3 else 'incomplete'
    finally:
        best=max(experiments,key=lambda r:r['enhancement']) if experiments else None
        (output/'summary.json').write_text(json.dumps(dict(status=status,model=args.model,best=best),indent=2))
        if experiments:
            with (output/'experiments.csv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(experiments[0]['settings'])+['enhancement','initial_enhancement','enhancement_shot_std'])
                writer.writeheader()
                for r in experiments: writer.writerow({**r['settings'],**{k:r[k] for k in ('enhancement','initial_enhancement','enhancement_shot_std')}})
    print('Saved to '+str(output),flush=True)
    return 0 if status=='complete' else 1

if __name__=='__main__': sys.exit(main())
