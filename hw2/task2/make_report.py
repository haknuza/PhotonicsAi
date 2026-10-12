"""Build a short report using only persisted, measured experiment results."""
import argparse
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('run',type=Path)
    ap.add_argument('--lecture-enhancement',type=float)
    args=ap.parse_args()
    results=json.loads((args.run/'experiments.json').read_text())
    summary=json.loads((args.run/'summary.json').read_text())
    styles=getSampleStyleSheet(); story=[]
    def para(text): story.extend([Paragraph(text,styles['BodyText']),Spacer(1,10)])
    story.append(Paragraph('Homework 2 - Task 2: Simulated beam focusing',styles['Title']))
    para('Code: <link href="https://github.com/haknuza/PhotonicsAi/tree/main/hw2/task2">github.com/haknuza/PhotonicsAi - hw2/task2</link> (local changes must be pushed before submission).')
    para(f"Agent model: {summary['model']}. Four LangGraph roles share the dialogue: orchestrator, scientist, general and executor. The orchestrator chooses each destination through a model call; only the executor receives the focusing tool. The user asks to focus the beam and supplies no experiment settings. The returned enhancement and images are retained directly by the harness.")
    para('The bench uses a 16 x 16 binary amplitude DMD and a 32 x 32 camera. A fixed complex Gaussian transmission matrix models the scattering medium. The camera intensity is |T Ein| squared, with Poisson shot noise (1000 photons per intensity unit) and Gaussian read noise (standard deviation 0.002). A macropixel shares one binary amplitude across its physical pixels. All experiments use the same physical matrix.')
    para('The target is centered at camera (16,16). The proximal background is an annulus, starting two pixels beyond the focus half-width, split into angular sectors. Enhancement is the mean focus intensity divided by the unweighted mean of the background sector means. Each proposal flips randomly selected macropixels. It is accepted when its noisy enhancement exceeds a freshly measured current enhancement. The optimizer receives only camera measurements. The final result is measured independently using the mean of 20 exposures. Noise can cause mistaken acceptances; no monotonic improvement is guaranteed.')
    data=[['Run','Macro','Focus','Radius','Regions','Iters','Flips','Seed','Initial','Final']]
    for i,r in enumerate(results,1):
        s=r['settings']; data.append([str(i)]+[str(s[k]) for k in ('macropixel_size','focus_size','background_radius','background_regions','iterations','flips','seed')]+[f"{r['initial_enhancement']:.2f}",f"{r['enhancement']:.2f}"])
    table=Table(data,hAlign='LEFT'); table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.lightgrey),('GRID',(0,0),(-1,-1),.4,colors.grey),('FONTSIZE',(0,0),(-1,-1),8),('LEFTPADDING',(0,0),(-1,-1),5)])); story.extend([table,Spacer(1,12)])
    best=max(results,key=lambda r:r['enhancement'])
    para(f"Best measured enhancement: {best['enhancement']:.3f}; initial: {best['initial_enhancement']:.3f}. Across the 20 final shots the sample standard deviation of the individual-shot enhancements was {best['enhancement_shot_std']:.3f}. This is shot variability, not uncertainty across independent optimization runs. Larger focus regions and different background definitions change the objective and are not directly comparable. The runs are illustrative rather than a statistically replicated benchmark.")
    para('The reported best run maximizes the measured local contrast among these experiments; it does not establish a global optimum or guarantee that the target is the brightest pixel on the entire camera. The raw tool results determine this table and the saved best result. The small language model can repeat configurations, propose invalid settings or misstate a result in its narrative; validation rejects invalid tool inputs, and numerical claims in that narrative do not replace the saved measurements.')
    if args.lecture_enhancement is None:
        para('Comparison with the supplied lab resource: <link href="https://photonicsai.com/wp-content/uploads/2026/08/Lab_work_2_focusing.pdf">Focusing Through a Scattering Medium</link>, sections 2.2-2.3, Figure 1 and Experiment Tasks 1-4. Its Figure 1 depicts diffuse speckle becoming a localized bright focus. Our before/after camera images show the corresponding camera-feedback objective, with residual speckle around the target. The resource uses a phase-only SLM to align the phases of segment contributions. This homework instead uses a binary amplitude DMD: the search switches contributions on or off, without changing their phases. Both use a static transmission matrix, a selected camera ROI, macropixel grids and repeated optimization, but the available control differs.')
        para('The lab asks for initial field-of-view background intensity and final optimized ROI intensity. Our enhancement instead uses the final proximal background, divided into sectors as required by the homework. These denominators differ, so enhancement coefficients cannot be directly equated. The supplied PDF contains a schematic focusing illustration and an empty results table, with no measured lecture enhancement or actual camera dataset. Therefore the comparison is qualitative; it does not claim quantitative agreement with a laboratory measurement. A separate lecture camera image, if provided, can be compared using matching ROI/background definitions.')
    else:
        para(f"Lecture enhancement supplied for comparison: {args.lecture_enhancement:.3f}. Simulation / lecture ratio: {best['enhancement']/args.lecture_enhancement:.3f}. This ratio is meaningful only if the focus and background definitions match; the simulated bench differs from the real setup.")
    for i,r in enumerate(results,1):
        story.append(PageBreak()); story.append(Paragraph(f'Experiment {i}',styles['Heading1']))
        para('Agent-chosen settings: '+', '.join(f'{k}={v}' for k,v in r['settings'].items()))
        story.append(Image(str(args.run/f'experiment_{i:02d}'/'camera.png'),width=490,height=157))
        para(f"Final enhancement {r['enhancement']:.3f}. Cyan marks the focus region; initial and final images share the same color scale. The final image averages 20 shots, whereas the initial image is a single noisy exposure.")
    SimpleDocTemplate(str(args.run/'task2_report.pdf')).build(story)
    print(args.run/'task2_report.pdf')

if __name__=='__main__': main()
