"""Binary amplitude DMD, fixed complex transmission matrix, noisy camera."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, model_validator

class Settings(BaseModel):
    macropixel_size: int = Field(ge=1, le=8)
    focus_size: int = Field(ge=1, le=5)
    background_radius: int = Field(ge=4, le=15)
    background_regions: int = Field(ge=2, le=8)
    iterations: int = Field(ge=10, le=2000)
    flips: int = Field(default=1, ge=1, le=16)
    seed: int = Field(default=42, ge=0)

    @model_validator(mode='after')
    def validate_geometry(self):
        if 16 % self.macropixel_size:
            raise ValueError('macropixel_size must divide the 16 x 16 DMD: 1, 2, 4, 8')
        if self.focus_size % 2 == 0:
            raise ValueError('focus_size must be odd')
        if self.flips > (16 // self.macropixel_size)**2:
            raise ValueError('flips exceeds the number of macropixels')
        return self

class Bench:
    def __init__(self, settings):
        self.s = settings
        # Same physical bench for every run; the agent seed controls search/noise only.
        physical = np.random.default_rng(2026)
        self.T = (physical.normal(size=(32*32, 16*16)) +
                  1j*physical.normal(size=(32*32, 16*16))) / np.sqrt(2*256)
        self.rng = np.random.default_rng(settings.seed)
        y, x = np.indices((32, 32))
        dx, dy = x-16, y-16
        h = settings.focus_size//2
        self.focus = (abs(dx)<=h) & (abs(dy)<=h)
        radius = np.hypot(dx, dy)
        ring = (radius >= h+2) & (radius <= settings.background_radius) & ~self.focus
        angle = (np.arctan2(dy, dx)+np.pi) / (2*np.pi)
        self.background = [ring & (np.minimum((angle*settings.background_regions).astype(int),
                                 settings.background_regions-1)==i)
                           for i in range(settings.background_regions)]
        if any(not m.any() for m in self.background):
            raise ValueError('empty background region')
        m = settings.macropixel_size
        # Aggregating columns exactly equals repeating each binary macropixel on the DMD.
        self.A = self.T.reshape(1024, 16//m, m, 16//m, m).sum(axis=(2,4)).reshape(1024,-1)

    def shot(self, field):
        intensity = abs(field.reshape(32,32))**2
        return self.rng.poisson(intensity*1000)/1000 + self.rng.normal(0, .002, (32,32))

    def metric(self, image):
        bg = np.array([image[m].mean() for m in self.background])
        return float(image[self.focus].mean()/max(float(bg.mean()), 1e-12)), bg.tolist()

    def run(self, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        bits = self.rng.integers(0,2,self.A.shape[1])
        if not bits.any(): bits[0] = 1
        field = self.A @ bits
        initial = self.shot(field)
        initial_eta, _ = self.metric(initial)
        score = initial_eta
        trace = [score]
        accepted = 0
        measurements = 1
        for _ in range(self.s.iterations):
            idx = self.rng.choice(len(bits), self.s.flips, replace=False)
            delta = 1-2*bits[idx]
            if bits.sum()+delta.sum() == 0:
                trace.append(score)
                continue
            candidate = field + self.A[:,idx] @ delta
            # Fresh paired measurements reduce drift from accepting an unusually lucky shot.
            current_eta, _ = self.metric(self.shot(field))
            candidate_eta, _ = self.metric(self.shot(candidate))
            measurements += 2
            if candidate_eta > current_eta:
                bits[idx] = 1-bits[idx]
                field = candidate
                score = candidate_eta
                accepted += 1
            else:
                score = current_eta
            trace.append(score)
        # Independent repeated exposures; report ratio-of-means and per-shot variability.
        shots = np.array([self.shot(field) for _ in range(20)])
        final = shots.mean(axis=0)
        eta, bg = self.metric(final)
        repeat_eta = [self.metric(im)[0] for im in shots]
        result = dict(settings=self.s.model_dump(), enhancement=eta,
                      initial_enhancement=initial_eta, background_means=bg,
                      enhancement_shot_std=float(np.std(repeat_eta, ddof=1)),
                      accepted=accepted, active_macropixels=int(bits.sum()),
                      measurements=measurements+20,
                      final_exposures=20, output=str(output.resolve()))
        np.savez_compressed(output/'arrays.npz', initial=initial, final=final,
                            pattern=bits.reshape(16//self.s.macropixel_size,-1), trace=trace,
                            focus=self.focus, background=np.array(self.background))
        fig, axes = plt.subplots(1,3,figsize=(12,3.5))
        vmax = max(initial.max(),final.max())
        for ax, im, title in zip(axes[:2], [initial,final], ['Initial noisy shot','Final: mean of 20 noisy shots']):
            artist = ax.imshow(im, cmap='inferno', vmin=0, vmax=vmax)
            ax.contour(self.focus, levels=[.5], colors='cyan', linewidths=.7)
            ax.set_title(title); ax.set_xlabel('camera x'); ax.set_ylabel('camera y')
        fig.colorbar(artist, ax=axes[:2], shrink=.7, label='Intensity (same scale)')
        axes[2].plot(trace); axes[2].set(xlabel='Iteration', ylabel='Measured enhancement', title=f'Final enhancement {eta:.2f}')
        fig.savefig(output/'camera.png',dpi=150,bbox_inches='tight'); plt.close(fig)
        (output/'result.json').write_text(json.dumps(result,indent=2))
        return result
