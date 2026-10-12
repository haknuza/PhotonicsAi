import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from langchain_core.messages import AIMessage
from run_focus import build_focus_graph

class Model:
    def __init__(self): self.decisions=0; self.executor_calls=0; self.tools=[]
    def bind_tools(self,tools): self.tools=tools; return self
    def invoke(self,messages):
        system=messages[0].content
        if system.startswith('You are the ORCHESTRATOR'):
            next_role=['scientist','executor','finish'][self.decisions]
            self.decisions+=1
            return AIMessage(content=json.dumps(dict(next_role=next_role,instruction='Run planned experiments',reason='test model decision')))
        if system.startswith('You are the SCIENTIST'):
            return AIMessage(content='Compare macropixels 1, 2, 4 with a one-pixel target.')
        self.executor_calls+=1
        if self.executor_calls==1:
            return AIMessage(content='',tool_calls=[dict(name='focus_beam',id=str(m),args=dict(
                macropixel_size=m,focus_size=1,background_radius=10,background_regions=4,iterations=30))
                for m in (1,2,4)])
        return AIMessage(content='Experiments completed.')


def test_executor_tool_results_and_model_routing(tmp_path):
    model=Model(); graph,experiments=build_focus_graph(model,tmp_path)
    final=graph.invoke({'messages':[dict(role='user',text='Focus the beam')]})
    assert len(experiments)==3
    assert [r['settings']['macropixel_size'] for r in experiments]==[1,2,4]
    assert [m['role'] for m in final['messages']]==['user','orchestrator','scientist','orchestrator','executor','orchestrator']
    executor=next(m for m in final['messages'] if m['role']=='executor')
    assert json.loads(executor['tool_results'][0]['result'])['enhancement']==experiments[0]['enhancement']
    assert [t.name for t in model.tools]==['focus_beam','write_and_run_python']
